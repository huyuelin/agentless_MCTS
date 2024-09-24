import argparse
import concurrent.futures
import json
import os
from difflib import unified_diff
import re

from datasets import load_dataset
from tqdm import tqdm

from agentless.util.api_requests import num_tokens_from_messages
from agentless.util.model import make_model
from agentless.util.postprocess_data import (
    check_code_differ_by_just_empty_lines,
    check_syntax,
    extract_python_blocks,
    fake_git_repo,
    lint_code,
    parse_diff_edit_commands,
    parse_edit_commands,
    remove_empty_lines,
    split_edit_multifile_commands,
)
from agentless.util.preprocess_data import (
    get_full_file_paths_and_classes_and_functions,
    get_repo_structure,
    line_wrap_content,
    transfer_arb_locs_to_locs,
)
from agentless.util.utils import load_jsonl, setup_logger

function_names_self_evaluation_prompt = """ 
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---


Below are the file names that we have generated patches for:
--- BEGIN FILE NAMES ---
{file_names}
--- END FILE NAMES ---

Below are the function names that we have found for the issue:
--- BEGIN FUNCTION NAMES ---
{function_names}
--- END FUNCTION NAMES ---


Please generate `confidence number`  to judge whether the above file names can solve the above issue.
If you are confident that the patch can solve the issue, please generate `confidence number` as 1.
If you are not confident that the patch can solve the issue, please generate `confidence number` as 0.

Here is an example of output:
 
###"confidence": 1###



"""




def self_evaluation_process_with_llm(loc, args, swe_bench_data, prev_o):
    instance_id = loc["instance_id"]
    log_file = os.path.join(
        args.output_folder, "self_evalluation_logs", f"{instance_id}.log"
    )
    logger = setup_logger(log_file)
    found = False
    for o in prev_o:
        if o["instance_id"] == instance_id:
            found = True
            break

    if found:
        logger.info(f"skipping {instance_id} since self_evaluation already generated")
        return None

    logger.info(f"================ repairing {instance_id} ================")
    if len(loc["found_files"]) == 0 :
        return {
            "instance_id": instance_id,
            "raw_output": [""],
            "try_count": [0],
            "all_generations": [[]],
            "traj": [],
            "prev_content": [[]],
            "confidence": [[]],
        }
        
    pred_files = loc["found_files"][: args.top_n]
    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    problem_statement = bench_data["problem_statement"]
    # structure = get_repo_structure(
    #         instance_id, bench_data["repo"], bench_data["base_commit"], "playground"
    #     )
    raw_outputs, counts, all_generations, traj, prev_contents, confidence = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    raw_output = ""
    prompt_template = function_names_self_evaluation_prompt
    
    found_related_locs = loc["found_related_locs"]
    flattened_related_locs = [item for sublist in found_related_locs for item in sublist if item]
      
    message = prompt_template.format(
        problem_statement=problem_statement,
        file_names="\n".join(loc["found_files"]),
        function_names="\n".join(flattened_related_locs),
    ).strip()
    
    # message = prompt_template.format(
    #     problem_statement=problem_statement,
    #     file_names="\n".join("xxx"),
    #     function_names="\n".join("xxx"),
    # ).strip()
        
    logger.info(f"prompting with message:\n{message}")
    
    model = make_model(
        model=args.model,
        logger=logger,
        backend=args.backend,
        max_tokens=1024,
        temperature=0,
        batch_size=1,
    )
    if args.skip_greedy:
        greedy_traj = {
            "response": "",
            "usage": {
                "completion_tokens": 0,
                "prompt_tokens": 0,
            },
        }
    else:
        if args.mock:
            greedy_traj = {
                "response": "",
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, args.model),
                },
            }
        else:
            greedy_traj = model.codegen(message, num_samples=1)[0]
        
    sample_responses = []
    sample_responses.append(greedy_traj)
    # get temperature samples
    model = make_model(
        model=args.model,
        logger=logger,
        backend=args.backend,
        max_tokens=1024,
        temperature=0.8,
        batch_size=args.max_samples - 1,  # minus the 1 greedy sample
    )
    
    sample_trajs = []
    sample_responses.extend(sample_trajs)
    
    count = 0
    all_generations, counts, traj,  = [], [], [], 
    
    while count < args.max_samples:
        print(f"trying the {count + 1}-th sample ...")
        ret = sample_responses[count]
        count += 1
        traj.append({**ret, "prompt": message})
        
        raw_output = ret["response"]
        
        pattern = r'###"confidence":\s*([01])###'
        match = re.search(pattern, raw_output)
        confidence_number = int(match.group(1))
        confidence.append(confidence_number)
        
        logger.info(f"raw output:\n{raw_output}")
        all_generations.append(raw_output)
        counts.append(count)
        raw_outputs.append(raw_output)
        
    
    with open(args.output_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "raw_output": raw_outputs,
                    "all_generations": [all_generations],
                    "try_count": counts,
                    "traj": traj,
                    "confidence": confidence,
                }
            )
            + "\n"
        )  
        





def self_evaluation(args):
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    #swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    locs = load_jsonl(args.loc_file)
    prev_o = load_jsonl(args.output_file) if os.path.exists(args.output_file) else []
    
    with open(f"{args.output_folder}/used_locs.jsonl", "w") as f:
        for loc in locs:
            f.write(json.dumps(loc) + "\n")
    
    results = []
 
    for loc in tqdm(locs, total=len(locs)):
        instance_id = loc['instance_id']
        result = self_evaluation_process_with_llm(loc, args, swe_bench_data, prev_o)
        if result is not None:
            results.append(result)




def main(input_file,output_file):
    class Args:
        pass

    args = Args()

    # Set default values
    #args.loc_file = "/home/wsl/AgentlessMCTS/Agentless/0822_MCTS/location_function/loc_function_names.jsonl"  # 请替换为实际的文件路径
    args.loc_file_folder = "/home/wsl/AgentlessMCTS/Agentless/0822_MCTS/location_function"
    args.loc_file = os.path.join(args.loc_file_folder, input_file)
    
    args.top_n = 3
    args.loc_interval = False
    args.context_window = 10
    args.stop_at_n_unique_valid_samples = -1
    args.gen_and_process = False
    args.max_samples = 1
    args.select_id = -1
    args.model = "gpt-4o-2024-05-13"
    args.backend = "openai"
    args.output_folder = "/home/wsl/AgentlessMCTS/Agentless/0822_MCTS/self_eval_function_names"  # 请替换为实际的输出文件夹路径
    args.only_correct = False
    args.post_process = False
    args.add_space = False
    args.cot = False
    args.fine_grain_loc_only = False
    args.diff_format = False
    args.skip_greedy = False
    args.sticky_scroll = False
    args.num_threads = 1
    args.mock = False
    
    # 验证参数
    assert (not "deepseek" in args.model) or (
        args.backend == "deepseek"
    ), "Must specify `--backend deepseek` if using a DeepSeek model"
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    if not os.path.exists(os.path.join(args.output_folder, "self_evalluation_logs")):
        os.makedirs(os.path.join(args.output_folder, "self_evalluation_logs"))


    args.output_file = os.path.join(args.output_folder, output_file)
    
    self_evaluation(args)
    
    return  args.output_file
    




if __name__ == "__main__":
    main()
