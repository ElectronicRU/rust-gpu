//! This algorithm is not intended to be an optimization, it is rather for legalization.
//! Specifically, spir-v disallows things like a `StorageClass::Function` pointer to a
//! `StorageClass::Input` pointer. Our frontend definitely allows it, though, this is like taking a
//! `&Input<T>` in a function! So, we inline all functions that take these "illegal" pointers, then
//! run mem2reg (see mem2reg.rs) on the result to "unwrap" the Function pointer.

use super::apply_rewrite_rules;
use super::{get_name, get_names};
use crate::linker::simple_passes::outgoing_edges;
use rspirv::dr::{Block, Function, Instruction, Module, ModuleHeader, Operand};
use rspirv::spirv::{FunctionControl, Op, StorageClass, Word};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_session::Session;
use std::mem::take;

type FunctionMap = FxHashMap<Word, usize>;

pub fn inline(sess: &Session, module: &mut Module) -> super::Result<()> {
    // This algorithm gets real sad if there's recursion - but, good news, SPIR-V bans recursion
    let functions: FxHashMap<_, _> = module
        .functions
        .iter()
        .enumerate()
        .map(|(idx, f)| (f.def_id().unwrap(), idx))
        .collect();
    let (disallowed_argument_types, disallowed_return_types) =
        compute_disallowed_argument_and_return_types(module);
    let should_inline: FxHashSet<Word> = functions
        .iter()
        .filter_map(|(&id, idx)| {
            if should_inline(
                &disallowed_argument_types,
                &disallowed_return_types,
                &module.functions[*idx],
            ) {
                Some(id)
            } else {
                None
            }
        })
        .collect();
    let postorder = compute_function_postorder(sess, module, &should_inline)?;
    let ptr_cache = module
        .types_global_values
        .iter()
        .filter(|inst| {
            inst.class.opcode == Op::TypePointer
                && inst.operands[0].unwrap_storage_class() == StorageClass::Function
        })
        .map(|inst| (inst.operands[1].unwrap_id_ref(), inst.result_id.unwrap()))
        .collect();

    let void = module
        .types_global_values
        .iter()
        .find(|inst| inst.class.opcode == Op::TypeVoid)
        .map(|inst| inst.result_id.unwrap())
        .unwrap_or(0);

    let mut inliner = Inliner {
        header: module.header.as_mut().unwrap(),
        types_global_values: &mut module.types_global_values,
        ptr_cache,
        void,
        functions: &functions,
        should_inline: &should_inline,
    };
    for index in postorder {
        inliner.inline_fn(&mut module.functions, index);
        fuse_trivial_branches(&mut module.functions[index]);
    }
    module
        .functions
        .retain(|f| f.def_id().map_or(true, |id| !should_inline.contains(&id)));
    // Drop OpName etc. for inlined functions
    module.debug_names.retain(|inst| {
        !inst.operands.iter().any(|op| {
            op.id_ref_any()
                .map_or(false, |id| should_inline.contains(&id))
        })
    });
    Ok(())
}

/// Topological sorting algorithm due to T. Cormen
fn compute_function_postorder(
    sess: &Session,
    module: &Module,
    excluded_roots: &FxHashSet<Word>,
) -> super::Result<Vec<usize>> {
    let func_to_index: FxHashMap<Word, usize> = module
        .functions
        .iter()
        .enumerate()
        .map(|(index, func)| (func.def_id().unwrap(), index))
        .collect();
    #[derive(Clone)]
    enum NodeState {
        NotVisited,
        Discovered,
        Finished,
    }
    let mut states = vec![NodeState::NotVisited; module.functions.len()];
    let mut has_recursion = false;
    let mut postorder = vec![];
    for index in 0..module.functions.len() {
        if let NodeState::NotVisited = states[index] {
            if !excluded_roots.contains(&module.functions[index].def_id().unwrap()) {
                visit(
                    sess,
                    module,
                    index,
                    &mut states[..],
                    &mut has_recursion,
                    &mut postorder,
                    &func_to_index,
                );
            }
        }
    }

    fn visit(
        sess: &Session,
        module: &Module,
        current: usize,
        states: &mut [NodeState],
        has_recursion: &mut bool,
        postorder: &mut Vec<usize>,
        func_to_index: &FxHashMap<Word, usize>,
    ) {
        states[current] = NodeState::Discovered;

        for next in calls(&module.functions[current], func_to_index) {
            match states[next] {
                NodeState::Discovered => {
                    let names = get_names(module);
                    let current_name =
                        get_name(&names, module.functions[current].def_id().unwrap());
                    let next_name = get_name(&names, module.functions[next].def_id().unwrap());
                    sess.err(&format!(
                        "module has recursion, which is not allowed: `{}` calls `{}`",
                        current_name, next_name
                    ));
                    *has_recursion = true;
                    break;
                }
                NodeState::NotVisited => {
                    visit(
                        sess,
                        module,
                        next,
                        states,
                        has_recursion,
                        postorder,
                        func_to_index,
                    );
                }
                NodeState::Finished => {}
            }
        }

        states[current] = NodeState::Finished;
        postorder.push(current)
    }

    fn calls<'a>(
        func: &'a Function,
        func_to_index: &'a FxHashMap<Word, usize>,
    ) -> impl Iterator<Item = usize> + 'a {
        func.all_inst_iter()
            .filter(|inst| inst.class.opcode == Op::FunctionCall)
            .map(move |inst| {
                *func_to_index
                    .get(&inst.operands[0].id_ref_any().unwrap())
                    .unwrap()
            })
    }
    if has_recursion {
        Err(rustc_errors::ErrorReported)
    } else {
        Ok(postorder)
    }
}

fn compute_disallowed_argument_and_return_types(
    module: &Module,
) -> (FxHashSet<Word>, FxHashSet<Word>) {
    let allowed_argument_storage_classes = &[
        StorageClass::UniformConstant,
        StorageClass::Function,
        StorageClass::Private,
        StorageClass::Workgroup,
        StorageClass::AtomicCounter,
    ];
    let mut disallowed_argument_types = FxHashSet::default();
    let mut disallowed_pointees = FxHashSet::default();
    let mut disallowed_return_types = FxHashSet::default();
    for inst in &module.types_global_values {
        match inst.class.opcode {
            Op::TypePointer => {
                let storage_class = inst.operands[0].unwrap_storage_class();
                let pointee = inst.operands[1].unwrap_id_ref();
                if !allowed_argument_storage_classes.contains(&storage_class)
                    || disallowed_pointees.contains(&pointee)
                    || disallowed_argument_types.contains(&pointee)
                {
                    disallowed_argument_types.insert(inst.result_id.unwrap());
                }
                disallowed_pointees.insert(inst.result_id.unwrap());
                disallowed_return_types.insert(inst.result_id.unwrap());
            }
            Op::TypeStruct => {
                let fields = || inst.operands.iter().map(|op| op.id_ref_any().unwrap());
                if fields().any(|id| disallowed_argument_types.contains(&id)) {
                    disallowed_argument_types.insert(inst.result_id.unwrap());
                }
                if fields().any(|id| disallowed_pointees.contains(&id)) {
                    disallowed_pointees.insert(inst.result_id.unwrap());
                }
                if fields().any(|id| disallowed_return_types.contains(&id)) {
                    disallowed_return_types.insert(inst.result_id.unwrap());
                }
            }
            Op::TypeArray | Op::TypeRuntimeArray | Op::TypeVector => {
                let id = inst.operands[0].id_ref_any().unwrap();
                if disallowed_argument_types.contains(&id) {
                    disallowed_argument_types.insert(inst.result_id.unwrap());
                }
                if disallowed_pointees.contains(&id) {
                    disallowed_pointees.insert(inst.result_id.unwrap());
                }
            }
            _ => {}
        }
    }
    (disallowed_argument_types, disallowed_return_types)
}

fn should_inline(
    disallowed_argument_types: &FxHashSet<Word>,
    disallowed_return_types: &FxHashSet<Word>,
    function: &Function,
) -> bool {
    let def = function.def.as_ref().unwrap();
    let control = def.operands[0].unwrap_function_control();
    control.contains(FunctionControl::INLINE)
        || function
            .parameters
            .iter()
            .any(|inst| disallowed_argument_types.contains(inst.result_type.as_ref().unwrap()))
        || disallowed_return_types.contains(&function.def.as_ref().unwrap().result_type.unwrap())
}

// This should be more general, but a very common problem is passing an OpAccessChain to an
// OpFunctionCall (i.e. `f(&s.x)`, or more commonly, `s.x.f()` where `f` takes `&self`), so detect
// that case and inline the call.
fn args_invalid(function: &Function, call: &Instruction) -> bool {
    for inst in function.all_inst_iter() {
        if inst.class.opcode == Op::AccessChain {
            let inst_result = inst.result_id.unwrap();
            if call
                .operands
                .iter()
                .any(|op| *op == Operand::IdRef(inst_result))
            {
                return true;
            }
        }
    }
    false
}

// Steps:
// Move OpVariable decls
// Rewrite return
// Renumber IDs
// Insert blocks

struct Inliner<'m, 'map> {
    header: &'m mut ModuleHeader,
    types_global_values: &'m mut Vec<Instruction>,
    ptr_cache: FxHashMap<Word, Word>,
    void: Word,
    functions: &'map FunctionMap,
    should_inline: &'map FxHashSet<Word>,
    // rewrite_rules: FxHashMap<Word, Word>,
}

impl Inliner<'_, '_> {
    fn id(&mut self) -> Word {
        let result = self.header.bound;
        self.header.bound += 1;
        result
    }

    fn ptr_ty(&mut self, pointee: Word) -> Word {
        if let Some(existing) = self.ptr_cache.get(&pointee) {
            return *existing;
        }
        let inst_id = self.id();
        self.types_global_values.push(Instruction::new(
            Op::TypePointer,
            None,
            Some(inst_id),
            vec![
                Operand::StorageClass(StorageClass::Function),
                Operand::IdRef(pointee),
            ],
        ));
        self.ptr_cache.insert(pointee, inst_id);
        inst_id
    }

    fn inline_fn(&mut self, functions: &mut [Function], index: usize) {
        let mut block_idx = 0;
        let mut caller = take(&mut functions[index]);
        while block_idx < caller.blocks.len() {
            // Since we process the functions in strict postorder, we never need to re-process a
            // block, and may even skip the inlined blocks.
            block_idx = self.inline_block(&mut caller, functions, block_idx);
        }
        functions[index] = caller;
    }

    fn inline_block(
        &mut self,
        caller: &mut Function,
        functions: &[Function],
        block_idx: usize,
    ) -> usize {
        // Find the first inlined OpFunctionCall
        let call = caller.blocks[block_idx]
            .instructions
            .iter()
            .enumerate()
            .filter(|(_, inst)| inst.class.opcode == Op::FunctionCall)
            .map(|(index, inst)| {
                (
                    index,
                    inst,
                    inst.operands[0].id_ref_any().unwrap(),
                    self.functions
                        .get(&inst.operands[0].id_ref_any().unwrap())
                        .unwrap(),
                )
            })
            .find(|(_, inst, id, _)| {
                self.should_inline.contains(&id) || args_invalid(caller, inst)
            });
        let (call_index, call_inst, _, callee_idx) = match call {
            None => {
                return block_idx + 1;
            }
            Some(call) => call,
        };
        let callee = &functions[*callee_idx];
        let call_result_type = {
            let ty = call_inst.result_type.unwrap();
            if ty == self.void {
                None
            } else {
                Some(ty)
            }
        };
        let call_result_id = call_inst.result_id.unwrap();
        // Rewrite parameters to arguments
        let call_arguments = call_inst
            .operands
            .iter()
            .skip(1)
            .map(|op| op.id_ref_any().unwrap());
        let callee_parameters = callee.parameters.iter().map(|inst| {
            assert!(inst.class.opcode == Op::FunctionParameter);
            inst.result_id.unwrap()
        });
        let mut rewrite_rules = callee_parameters.zip(call_arguments).collect();

        let return_variable = if call_result_type.is_some() {
            Some(self.id())
        } else {
            None
        };
        let return_jump = self.id();
        // Rewrite OpReturns of the callee.
        let mut inlined_blocks = get_inlined_blocks(callee, return_variable, return_jump);
        // Clone the IDs of the callee, because otherwise they'd be defined multiple times if the
        // fn is inlined multiple times.
        self.add_clone_id_rules(&mut rewrite_rules, &inlined_blocks);
        apply_rewrite_rules(&rewrite_rules, &mut inlined_blocks);

        // Split the block containing the OpFunctionCall into two, around the call.
        let mut post_call_block_insts = caller.blocks[block_idx]
            .instructions
            .split_off(call_index + 1);
        // pop off OpFunctionCall
        let call = caller.blocks[block_idx].instructions.pop().unwrap();
        assert!(call.class.opcode == Op::FunctionCall);

        if let Some(call_result_type) = call_result_type {
            // Generate the storage space for the return value: Do this *after* the split above,
            // because if block_idx=0, inserting a variable here shifts call_index.
            insert_opvariable(
                &mut caller.blocks[0],
                self.ptr_ty(call_result_type),
                return_variable.unwrap(),
            );
        }

        // Fuse the first block of the callee into the block of the caller. This is okay because
        // it's illegal to branch to the first BB in a function.
        let mut callee_header = inlined_blocks.remove(0).instructions;
        // TODO: OpLine handling
        let num_variables = callee_header.partition_point(|inst| inst.class.opcode == Op::Variable);
        caller.blocks[block_idx]
            .instructions
            .append(&mut callee_header.split_off(num_variables));
        // Move the OpVariables of the callee to the caller.
        insert_opvariables(&mut caller.blocks[0], callee_header);

        if let Some(call_result_type) = call_result_type {
            // Add the load of the result value after the inlined function. Note there's guaranteed no
            // OpPhi instructions since we just split this block.
            post_call_block_insts.insert(
                0,
                Instruction::new(
                    Op::Load,
                    Some(call_result_type),
                    Some(call_result_id),
                    vec![Operand::IdRef(return_variable.unwrap())],
                ),
            );
        }
        // Insert the second half of the split block.
        let continue_block = Block {
            label: Some(Instruction::new(Op::Label, None, Some(return_jump), vec![])),
            instructions: post_call_block_insts,
        };
        caller.blocks.insert(block_idx + 1, continue_block);

        let inlined_len = inlined_blocks.len();
        // Insert the rest of the blocks (i.e. not the first) between the original block that was
        // split.
        caller
            .blocks
            .splice((block_idx + 1)..(block_idx + 1), inlined_blocks);

        return block_idx + 1 + inlined_len;
    }

    fn add_clone_id_rules(&mut self, rewrite_rules: &mut FxHashMap<Word, Word>, blocks: &[Block]) {
        for block in blocks {
            for inst in block.label.iter().chain(&block.instructions) {
                if let Some(result_id) = inst.result_id {
                    let new_id = self.id();
                    let old = rewrite_rules.insert(result_id, new_id);
                    assert!(old.is_none());
                }
            }
        }
    }
}

fn get_inlined_blocks(
    function: &Function,
    return_variable: Option<Word>,
    return_jump: Word,
) -> Vec<Block> {
    let mut blocks = function.blocks.clone();
    for block in &mut blocks {
        let last = block.instructions.last().unwrap();
        if let Op::Return | Op::ReturnValue = last.class.opcode {
            if Op::ReturnValue == last.class.opcode {
                let return_value = last.operands[0].id_ref_any().unwrap();
                block.instructions.insert(
                    block.instructions.len() - 1,
                    Instruction::new(
                        Op::Store,
                        None,
                        None,
                        vec![
                            Operand::IdRef(return_variable.unwrap()),
                            Operand::IdRef(return_value),
                        ],
                    ),
                );
            } else {
                assert!(return_variable.is_none());
            }
            *block.instructions.last_mut().unwrap() =
                Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(return_jump)]);
        }
    }
    blocks
}

fn insert_opvariable(block: &mut Block, ptr_ty: Word, result_id: Word) {
    let index = block
        .instructions
        .partition_point(|inst| inst.class.opcode == Op::Variable);
    let inst = Instruction::new(
        Op::Variable,
        Some(ptr_ty),
        Some(result_id),
        vec![Operand::StorageClass(StorageClass::Function)],
    );
    block.instructions.insert(index, inst);
}

fn insert_opvariables(block: &mut Block, insts: Vec<Instruction>) {
    let index = block
        .instructions
        .partition_point(|inst| inst.class.opcode == Op::Variable);
    block.instructions.splice(index..index, insts);
}

fn fuse_trivial_branches(function: &mut Function) {
    let mut chain_list = compute_outgoing_1to1_branches(&function.blocks);

    for block_idx in 0..chain_list.len() {
        let mut next = chain_list[block_idx].take();
        loop {
            match next {
                None => {
                    // end of the chain list
                    break;
                }
                Some(x) if x == block_idx => {
                    // loop detected
                    break;
                }
                Some(next_idx) => {
                    let mut dest_insts = take(&mut function.blocks[next_idx].instructions);
                    assert_eq!(
                        function.blocks[next_idx].label_id().unwrap(),
                        function.blocks[block_idx]
                            .instructions
                            .last()
                            .unwrap()
                            .operands[0]
                            .unwrap_id_ref()
                    );
                    let self_insts = &mut function.blocks[block_idx].instructions;
                    assert_eq!(self_insts.last().unwrap().class.opcode, Op::Branch);
                    self_insts.pop(); // pop the branch
                    self_insts.append(&mut dest_insts);
                    next = chain_list[next_idx].take();
                }
            }
        }
    }
    function.blocks.retain(|b| !b.instructions.is_empty());
}

fn compute_outgoing_1to1_branches(blocks: &[Block]) -> Vec<Option<usize>> {
    let block_id_to_idx: FxHashMap<_, _> = blocks
        .iter()
        .enumerate()
        .map(|(idx, block)| (block.label_id().unwrap(), idx))
        .collect();
    #[derive(Clone)]
    enum NumIncoming {
        Zero,
        One(usize),
        TooMany,
    }
    let mut incoming = vec![NumIncoming::Zero; blocks.len()];
    for (source_idx, source) in blocks.iter().enumerate() {
        for dest_id in outgoing_edges(source) {
            let dest_idx = block_id_to_idx[&dest_id];
            incoming[dest_idx] = match incoming[dest_idx] {
                NumIncoming::Zero => NumIncoming::One(source_idx),
                _ => NumIncoming::TooMany,
            }
        }
    }

    let mut result = vec![None; blocks.len()];

    for (dest_idx, inc) in incoming.iter().enumerate() {
        if let &NumIncoming::One(source_idx) = inc {
            if blocks[source_idx].instructions.last().unwrap().class.opcode == Op::Branch {
                result[source_idx] = Some(dest_idx);
            }
        }
    }

    result
}
