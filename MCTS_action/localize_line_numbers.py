##localize.py import
import argparse
import concurrent.futures
import json
import os

from datasets import load_dataset
from tqdm import tqdm

from agentless.fl.FL import LLMFL
from agentless.util.preprocess_data import (
    filter_none_python,
    filter_out_test_files,
    get_full_file_paths_and_classes_and_functions,
    show_project_structure,
)
from agentless.util.utils import (
    load_existing_instance_ids,
    load_json,
    load_jsonl,
    setup_logger,
)
from get_repo_structure.get_repo_structure import (
    clone_repo,
    get_project_structure_from_scratch,
)

##FL.py import
from abc import ABC, abstractmethod

from agentless.repair.repair import construct_topn_file_context
from agentless.util.compress_file import get_skeleton
from agentless.util.postprocess_data import extract_code_blocks, extract_locs_for_files
from agentless.util.preprocess_data import (
    correct_file_paths,
    get_full_file_paths_and_classes_and_functions,
    get_repo_files,
    line_wrap_content,
    show_project_structure,
)

MAX_CONTEXT_LENGTH = 128000


class FL(ABC):
    def __init__(self, instance_id, structure, problem_statement, **kwargs):
        self.structure = structure
        self.instance_id = instance_id
        self.problem_statement = problem_statement

    @abstractmethod
    def localize_line_numbers_with_LLM(self, top_n=1, mock=False) -> tuple[list, list, list, list, any]:
        pass


class LocLineNumbers(FL):
    localize_line_number_prompt = """
Please review the following GitHub problem description and relevant files, and provide a set of locations that need to be edited to fix the issue.
The locations can be specified as class names, function or method names, or exact line numbers that require modification.

### GitHub Problem Description ###
{problem_statement}

### File Contents ###
{file_contents}

###

Please provide the class name, function or method name, or the exact line numbers that need to be edited.
### Examples:
```
full_path1/file1.py
line: 10
class: MyClass1
line: 51

full_path2/file2.py
function: MyClass2.my_method
line: 12

full_path3/file3.py
function: my_function
line: 24
line: 156
```

Return just the location(s)
"""

    def __init__(
        self,
        instance_id,
        structure,
        problem_statement,
        model_name,
        backend,
        logger,
        match_partial_paths,
        **kwargs,
    ):
        super().__init__(instance_id, structure, problem_statement)
        self.max_tokens = 300
        self.model_name = model_name
        self.backend = backend
        self.logger = logger
        self.match_partial_paths = match_partial_paths

    def _parse_model_return_lines(self, content: str) -> list[str]:
        if content:
            return content.strip().split("\n")
    
    def localize_line_numbers_with_LLM(
        self,
        file_names,
        coarse_locs,
        context_window: int,
        add_space: bool,
        sticky_scroll: bool,
        no_line_number: bool,
        temperature: float = 0.0,
        num_samples: int = 1,
        mock=False,
    ):
        
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model
        
        file_contents = get_repo_files(self.structure, file_names)
        topn_content, file_loc_intervals = construct_topn_file_context(
            coarse_locs,
            file_names,
            file_contents,
            self.structure,
            context_window=context_window,
            loc_interval=True,
            add_space=add_space,
            sticky_scroll=sticky_scroll,
            no_line_number=no_line_number,
        )
        
        message = self.localize_line_number_prompt.format(
            problem_statement=self.problem_statement, file_contents=topn_content
        )

        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)
        
        token_count = num_tokens_from_messages(message, "gpt-4o-2024-05-13")
        
        if num_tokens_from_messages(message, "gpt-4o-2024-05-13")> MAX_CONTEXT_LENGTH:
            reduction_ratio = 1 - (110000 / token_count)
            content_lines = topn_content.split('\n')
            total_lines = len(content_lines)
            reduced_lines = content_lines[int(len(content_lines) * reduction_ratio) // 2 :total_lines - int(len(content_lines) * reduction_ratio) // 2]
            topn_content = '\n'.join(reduced_lines)
            
            message = template.format(
            problem_statement=self.problem_statement, file_contents=topn_content
        )
            
        assert num_tokens_from_messages(message, "gpt-4o-2024-05-13") < 128000# 断言token数小于128000  
        
        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=temperature,
            batch_size=num_samples,
        )
        
        raw_trajs = model.codegen(message, num_samples=num_samples)

        # Merge trajectories
        raw_outputs = [raw_traj["response"] for raw_traj in raw_trajs]
        traj = {
            "prompt": message,
            "response": raw_outputs,
            "usage": {  # merge token usage
                "completion_tokens": sum(
                    raw_traj["usage"]["completion_tokens"] for raw_traj in raw_trajs
                ),
                "prompt_tokens": sum(
                    raw_traj["usage"]["prompt_tokens"] for raw_traj in raw_trajs
                ),
            },
        }
        
        model_found_locs_separated_in_samples = []

        for raw_output in raw_outputs:
            model_found_locs = extract_code_blocks(raw_output)
            model_found_locs_separated = extract_locs_for_files(
                model_found_locs, file_names
            )
            model_found_locs_separated_in_samples.append(model_found_locs_separated)

            self.logger.info(f"==== raw output ====")
            self.logger.info(raw_output)
            self.logger.info("=" * 80)
            print(raw_output)
            print("=" * 80)
            self.logger.info(f"==== extracted locs ====")
            for loc in model_found_locs_separated:
                self.logger.info(loc)
            self.logger.info("=" * 80)
        self.logger.info("==== Input coarse_locs")
        coarse_info = ""
        for fn, found_locs in coarse_locs.items():
            coarse_info += f"### {fn}\n"
            if isinstance(found_locs, str):
                coarse_info += found_locs + "\n"
            else:
                coarse_info += "\n".join(found_locs) + "\n"
        self.logger.info("\n" + coarse_info)
        if len(model_found_locs_separated_in_samples) == 1:
            model_found_locs_separated_in_samples = (
                model_found_locs_separated_in_samples[0]
            )

        return (
            model_found_locs_separated_in_samples,
            {"raw_output_loc": raw_outputs},
            traj,
            topn_content,
        )




def localize_line_numbers_instance(bug, args, swe_bench_data):
    function_names_json = load_jsonl(args.input_function_names)
    file_names_json = load_jsonl(args.input_file_names)
    
    instance_id = bug["instance_id"] 

    instance_id = bug["instance_id"]  
    
      
    log_file = os.path.join(
        args.output_folder, "localization_line_numbers_logs", f"{instance_id}.log"
    )
    if os.path.exists(log_file):
        # 文件存在,清空内容
        open(log_file, 'w').close()
    else:
        # 文件不存在,创建目录
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
    logger = setup_logger(log_file)
    logger.info(f"Localizing line numbers of  {instance_id}")
    
    for file_name_json in file_names_json:
        if file_name_json["instance_id"] == instance_id:
            structure = file_name_json["structrue"]
    
    for function_name_json in function_names_json:
        if function_name_json["instance_id"] == instance_id:
            found_related_locs = function_name_json["found_related_locs"] 
            found_files = function_name_json["found_files"]

    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    problem_statement = bench_data["problem_statement"]
    
    fl = LocLineNumbers(
            instance_id,
            structure,
            problem_statement,
            args.model,
            args.backend,
            logger,
            args.match_partial_paths,
        )
    
    
    coarse_found_locs = {}
    for i, found_file in enumerate(found_files):
            if len(found_related_locs) > i:
                coarse_found_locs[found_file] = found_related_locs[i]
                
    found_edit_locs = []
    additional_artifact_loc_edit_location = None
    edit_loc_traj = {}
    
    
    
    
    (
            found_edit_locs,
            additional_artifact_loc_edit_location,
            edit_loc_traj,
            topn_content,
        ) = fl.localize_line_numbers_with_LLM(
            found_files,
            coarse_found_locs,
            context_window=args.context_window,
            add_space=args.add_space,
            no_line_number=args.no_line_number,
            sticky_scroll=args.sticky_scroll,
            temperature=args.temperature,
            num_samples=args.num_samples,
        )
    
    additional_artifact_loc_edit_location = [additional_artifact_loc_edit_location]
    
    with open(args.output_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": bug["instance_id"],
                    "found_files": found_files,
                    "found_related_locs": found_related_locs,
                    "found_edit_locs": found_edit_locs,
                    "additional_artifact_loc_edit_location": additional_artifact_loc_edit_location,
                    "edit_loc_traj": edit_loc_traj,
                    "topn_content": topn_content,
                }
            )
            + "\n"
        )

def localize_line_numbers(args):
    swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    #swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    valid_instance_ids = [
        "astropy__astropy-8707",       
    ]
    
    
    for bug in swe_bench_data:
        
        if bug["instance_id"] not in valid_instance_ids:
                continue
            
        localize_line_numbers_instance(
                bug, args, swe_bench_data,
            )


def main(input_file,structure_file,output_file):
    class Args:
        pass
    
    args = Args()
    args.output_folder = '/home/wsl/AgentlessMCTS/Agentless/0822_MCTS/location_line' 
    #args.output_file = "loc_line_numbers.jsonl"
    args.output_file = output_file
    #args.input_function_names = "/home/wsl/AgentlessMCTS/Agentless/0822_MCTS/location_function/loc_function_names.jsonl"
    args.input_function_names_folder = "/home/wsl/AgentlessMCTS/Agentless/0822_MCTS/location_function"
    args.input_function_names = os.path.join(args.input_function_names_folder, input_file)
    #args.input_file_names = "/home/wsl/AgentlessMCTS/Agentless/0822_MCTS/location_file/loc_file_names.jsonl"#为了提取structure,未来将structure放入一个单独的jsonl中
    args.input_file_names_folder = "/home/wsl/AgentlessMCTS/Agentless/0822_MCTS/location_file"#为了提取structure
    args.input_file_names = os.path.join(args.input_file_names_folder, structure_file) 
    args.start_file = None
    args.file_level = False
    args.related_level = False
    args.fine_grain_line_level = False
    args.top_n = 3
    args.temperature = 0.0
    args.num_samples = 1
    args.compress = False
    args.merge = False
    args.add_space = False
    args.no_line_number = False
    args.sticky_scroll = False
    args.match_partial_paths = True
    args.context_window = 10
    args.num_threads = 1
    args.target_id = None
    args.skip_existing = False
    args.mock = False
    args.model = "gpt-4o-2024-05-13"
    args.backend = "openai"
    
    
    args.output_file = os.path.join(args.output_folder, args.output_file)
    
    if not os.path.exists(os.path.join(args.output_folder, "localization_line_numbers_logs")):
        os.makedirs(os.path.join(args.output_folder, "localization_line_numbers_logs"))
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        
    localize_line_numbers(args)
    
    


if __name__ == "__main__":
    main()

    
    