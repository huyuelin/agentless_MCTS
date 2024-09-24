import random
from math import sqrt, log
import MCTS_action.localize_file_names
import MCTS_action.localize_function_names
import MCTS_action.localize_line_numbers
import MCTS_action.repair_with_rerank
import MCTS_action.self_evaluation_file_names
import MCTS_action.self_evaluation_function_names
import MCTS_action.self_evaluation_line_numbers
import MCTS_action.self_evaluation_patch
from agentless.util.utils import load_jsonl, setup_logger
import uuid

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.instance_id = state.instance_id
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0
        self.untried_actions = self.get_untried_actions()
        self.node_id = str(uuid.uuid4())  # 添加唯一标识符

    def get_untried_actions(self):
        return self.state.get_possible_actions()

    def select_child(self):
        # UCB1选择
        #return max(self.children, key=lambda c: c.value / c.visits + sqrt(2 * log(self.visits) / c.visits))
        
        # 修改后的UCT选择，考虑confidence
        exploration_weight = sqrt(2)
        return max(self.children, key=lambda c: 
                   (c.value / c.visits if c.visits > 0 else 0) +  # 利用项
                   exploration_weight * sqrt(log(self.visits) / c.visits if c.visits > 0 else float('inf')) +  # 探索项
                   (1 if c.state.confidence == 1 else 0))  # confidence 奖励

    def expand(self):
        action = self.untried_actions.pop(0)
        next_state = self.state.apply_action(action, self.node_id)
        child = MCTSNode(next_state, parent=self, action=action)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        #self.value += result[0] * self.state.confidence[0]
        result_value = result if isinstance(result, (int, float)) else result[0]
        confidence = self.state.confidence if isinstance(self.state.confidence, (int, float)) else self.state.confidence[0]
        # 只有当 confidence 为 1 时才更新 value
        if confidence == 1:
            self.value += result_value
        
class CodeRepairState:
    def __init__(self,instance_id,parent_node_id=None, line_numbers_node_id=None, file_names_node_id=None):
        self.instance_id = instance_id
        self.current_step = 0
        self.patch = None
        self.confidence = 0
        self.parent_node_id = parent_node_id
        self.line_numbers_node_id = line_numbers_node_id
        self.file_names_node_id = file_names_node_id

    def get_possible_actions(self):
        actions = [
            "localize_file_names",
            "localize_function_names",
            "localize_line_numbers",
            "repair_with_rerank"
        ]
        return actions[self.current_step:self.current_step+1]  # 只返回下一个可能的action

    def apply_action(self, action, node_id):
        new_state = CodeRepairState(self.instance_id, parent_node_id=node_id,
                                    line_numbers_node_id=self.line_numbers_node_id,
                                    file_names_node_id=self.file_names_node_id)
        new_state.current_step = self.current_step + 1

        if action == "localize_file_names":
            print(f"$$$$$$$$$$$$$$$$excuting localize_file_names in node {node_id}")
            localize_file_names_MCTS(node_id)
            new_state.file_names_node_id = node_id
            print(f"$$$$$$$$$$$$$$$$excuting self_evaluation_file_names_MCTS in node {node_id}")
            new_state.confidence = self_evaluation_file_names_MCTS(self.instance_id, node_id)
        elif action == "localize_function_names":
            print(f"$$$$$$$$$$$$$$$$excuting localize_function_names in node {node_id}")
            localize_function_names_MCTS(node_id, self.file_names_node_id)
            print(f"$$$$$$$$$$$$$$$$excuting self_evaluation_function_names_MCTS in node {node_id}")
            new_state.confidence = self_evaluation_function_names_MCTS(self.instance_id, node_id)
        elif action == "localize_line_numbers":
            print(f"$$$$$$$$$$$$$$$$excuting localize_line_numbers in node {node_id}")
            localize_line_numbers_MCTS(node_id, self.parent_node_id, self.file_names_node_id)
            new_state.line_numbers_node_id = node_id
            print(f"$$$$$$$$$$$$$$$$excuting self_evaluation_line_numbers_MCTS in node {node_id}")
            new_state.confidence = self_evaluation_line_numbers_MCTS(self.instance_id, node_id)
        elif action == "repair_with_rerank":
            print(f"$$$$$$$$$$$$$$$$excuting repair_with_rerank in node {node_id}")
            repair_with_rerank_MCTS(node_id, self.parent_node_id, self.file_names_node_id)
            print(f"$$$$$$$$$$$$$$$$excuting self_evaluation_patch_MCTS in node {node_id}")
            new_state.confidence = self_evaluation_patch_MCTS(self.instance_id, node_id, new_state.line_numbers_node_id)

        return new_state

    def is_terminal(self):
        return self.current_step == 4 or self.confidence == 1 #是否是and
    
    def get_result(self):
        return self.confidence

def get_action_confidence(action, node_id, instance_id):
    if action == "localize_file_names":
        file = f"self_eval_file_names/self_evaluation_file_names_{node_id}.jsonl"
    elif action == "localize_function_names":
        file = f"self_eval_function_names/self_evaluation_function_names_{node_id}.jsonl"
    elif action == "localize_line_numbers":
        file = f"self_eval_line_numbers/self_evaluation_line_numbers_{node_id}.jsonl"
    elif action == "repair_with_rerank":
        file = f"self_eval_patch/self_evaluation_patch_{node_id}.jsonl"
    else:
        return 0
    
    output_folder = "/home/wsl/AgentlessMCTS/Agentless/0822_MCTS/location_file"
    file = os.path.join(output_folder, file)
    
    data = load_jsonl(file)
    
    for item in data:
        if item["instance_id"] == instance_id:
            return item["confidence"]
    return 0

def mcts_code_repair(instance_id, iterations=15):
    root = MCTSNode(CodeRepairState(instance_id))

    for _ in range(iterations):
        node = root
        
        # Selection
        while node.untried_actions == [] and node.children != []:
            node = node.select_child()

        # Expansion
        if node.untried_actions != []:
            node = node.expand()

        # Simulation
        state = node.state
        while not state.is_terminal():
            action = state.get_possible_actions()[0]
            state = state.apply_action(action, node.node_id)

        # Backpropagation
        while node is not None:
            node.update(state.get_result())
            node = node.parent
            
    # # 选择最佳路径
    # best_path = []
    # best_overall_confidence = 0
    # node = root
    # while node.children:
    #     node = max(node.children, key=lambda c: c.visits * c.state.confidence)
    #     best_path.append(node.action)

    # return best_path, node.state
    
    # Find the path with the highest overall confidence
    best_path = []
    best_overall_confidence = 0
    best_node = None
    
    def dfs(node, current_path, current_confidence):
        nonlocal best_path, best_overall_confidence, best_node

        if not node.children:
            if current_confidence > best_overall_confidence:
                best_overall_confidence = current_confidence
                best_path = current_path.copy()
                best_node = node
            return
        
        for child in node.children:
            action_confidence = get_action_confidence(child.action, child.node_id, instance_id)
            new_confidence = current_confidence * action_confidence  #可能需要改变
            current_path.append(child.action)
            dfs(child, current_path, new_confidence)
            current_path.pop()
    dfs(root, [], 1)
    
    # Get the best patch
    best_patch = None
    if best_node and best_node.action == "repair_with_rerank":
        patch_file = f"/home/wsl/AgentlessMCTS/Agentless/0822_MCTS/repair_with_rerank/repair_with_rerank_{best_node.node_id}.jsonl"
        patches = load_jsonl(patch_file)
        for patch in patches:
            if patch["instance_id"] == instance_id:
                best_patch = patch["model_patch"]
                break

    return best_path, best_patch, best_overall_confidence,best_node.node_id
    
    
    

# 辅助函数（需要实现）
def localize_file_names_MCTS(node_id):
    MCTS_action.localize_file_names.main(output_file=f"localize_file_names_{node_id}.jsonl")
    pass

def localize_function_names_MCTS(node_id, file_names_node_id):
    MCTS_action.localize_function_names.main(input_file = f"localize_file_names_{file_names_node_id}.jsonl",
                                             output_file=f"localize_function_names_{node_id}.jsonl")
    pass

def localize_line_numbers_MCTS(node_id, parent_node_id,file_names_node_id):
    MCTS_action.localize_line_numbers.main(input_file = f"localize_function_names_{parent_node_id}.jsonl",
                                           structure_file=f"localize_file_names_{file_names_node_id}.jsonl",
                                           output_file=f"localize_line_numbers_{node_id}.jsonl")
    pass

def repair_with_rerank_MCTS(node_id, parent_node_id, file_names_node_id):
    MCTS_action.repair_with_rerank.main(input_file = f"localize_line_numbers_{parent_node_id}.jsonl",
                                         structure_file=f"localize_file_names_{file_names_node_id}.jsonl",
                                        output_file=f"repair_with_rerank_{node_id}.jsonl")
    pass

def self_evaluation_file_names_MCTS(instance_id, node_id):
    output_file = MCTS_action.self_evaluation_file_names.main(input_file = f"localize_file_names_{node_id}.jsonl", output_file=f"self_evaluation_file_names_{node_id}.jsonl")
    file_names_jsonl = load_jsonl(output_file)
    for file_name_jsonl in file_names_jsonl:
        if file_name_jsonl["instance_id"] == instance_id:
            return file_name_jsonl["confidence"]
    pass

def self_evaluation_function_names_MCTS(instance_id, node_id):
    output_file = MCTS_action.self_evaluation_function_names.main(input_file = f"localize_function_names_{node_id}.jsonl", output_file=f"self_evaluation_function_names_{node_id}.jsonl")
    function_names_jsonl = load_jsonl(output_file)
    for function_name_jsonl in function_names_jsonl:
        if function_name_jsonl["instance_id"] == instance_id:
            return function_name_jsonl["confidence"]
    pass

def self_evaluation_line_numbers_MCTS(instance_id, node_id):
    output_file = MCTS_action.self_evaluation_line_numbers.main(input_file = f"localize_line_numbers_{node_id}.jsonl", output_file=f"self_evaluation_line_numbers_{node_id}.jsonl")
    line_numbers_jsonl = load_jsonl(output_file)
    for line_number_jsonl in line_numbers_jsonl:
        if line_number_jsonl["instance_id"] == instance_id:
            return line_number_jsonl["confidence"]
    pass

def self_evaluation_patch_MCTS(instance_id, node_id, line_numbers_node_id):
    output_file = MCTS_action.self_evaluation_patch.main(input_file_patch = f"repair_with_rerank_{node_id}.jsonl", 
                                                         input_file_line_numbers=f"localize_line_numbers_{line_numbers_node_id}.jsonl",
                                                         output_file = f"self_evaluation_patch_{node_id}.jsonl")
    patches_jsonl = load_jsonl(output_file)
    for patch_jsonl in patches_jsonl:
        if patch_jsonl["instance_id"] == instance_id:
            return patch_jsonl["confidence"]
    pass

# 使用示例
instance_id = "astropy__astropy-8707"
#best_path, final_state = mcts_code_repair(instance_id)
best_path, final_state, confidence,best_node_id = mcts_code_repair(instance_id)
print("Best path:", best_path)
print("Final state:", final_state)
print("Confidence:", confidence)
print("Best node id:", best_node_id)