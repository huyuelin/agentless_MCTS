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
    def localize_function_names_with_LLM(self, top_n=1, mock=False) -> tuple[list, list, list, any]:
        pass


class LocFuncName(FL):
    localize_function_names_prompt = """
Please look through the following GitHub Problem Description and the Skeleton of Relevant Files.
Identify all locations that need inspection or editing to fix the problem, including directly related areas as well as any potentially related global variables, functions, and classes.
For each location you provide, either give the name of the class, the name of a method in a class, the name of a function, or the name of a global variable.

### GitHub Problem Description ###
{problem_statement}

### Skeleton of Relevant Files ###
{file_contents}

###

Please provide the complete set of locations as either a class name, a function name, or a variable name.
Note that if you include a class, you do not need to list its specific methods.
You can include either the entire class or don't include the class name and instead include specific methods in the class.
### Examples:
```
full_path1/file1.py
function: my_function_1
class: MyClass1
function: MyClass2.my_method

full_path2/file2.py
variable: my_var
function: MyClass3.my_method

full_path3/file3.py
function: my_function_2
function: my_function_3
function: MyClass4.my_method_1
class: MyClass5
```

Return just the locations.
"""

    file_content_in_block_template = """
### File: {file_name} ###
```python
{file_content}
```
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
    
    def localize_function_names_with_LLM(self,file_names,):
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model
        
        file_contents = get_repo_files(self.structure, file_names)
        compressed_file_contents = {
            fn: get_skeleton(code) for fn, code in file_contents.items()
        }
        
        contents = [
            self.file_content_in_block_template.format(file_name=fn, file_content=code)
            for fn, code in compressed_file_contents.items()
        ]
        
        file_contents = "".join(contents)
        
        message =self.localize_function_names_prompt.format(
            problem_statement=self.problem_statement, file_contents=file_contents
        )
        
        def message_too_long(message):
            return (
                num_tokens_from_messages(message, self.model_name) >= MAX_CONTEXT_LENGTH
            )
            
        while message_too_long(message) and len(contents) > 1:
            self.logger.info(f"reducing to \n{len(contents)} files")
            contents = contents[:-1]
            file_contents = "".join(contents)
            message = template.format(
                problem_statement=self.problem_statement, file_contents=file_contents
            )  # Recreate message
            
        if message_too_long(message):
            raise ValueError(
                "The remaining file content is too long to fit within the context length"
            )
        
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)
        
        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]
        
        model_found_locs = extract_code_blocks(raw_output)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file_names
        )
        
        self.logger.info(f"==== raw output ====")
        self.logger.info(raw_output)
        self.logger.info("=" * 80)
        self.logger.info(f"==== extracted locs ====")
        for loc in model_found_locs_separated:
            self.logger.info(loc)
        self.logger.info("=" * 80)
        
        print(raw_output)

        return model_found_locs_separated, {"raw_output_loc": raw_output}, traj
        
        
        
        
        
        

def localize_function_names_instance(bug, args, swe_bench_data):
    file_names_json = load_jsonl(args.input_file_names)
    
    instance_id = bug["instance_id"]  
      
    log_file = os.path.join(
        args.output_folder, "localization_function_names_logs", f"{instance_id}.log"
    )
    if os.path.exists(log_file):
        # 文件存在,清空内容
        open(log_file, 'w').close()
    else:
        # 文件不存在,创建目录
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
    logger = setup_logger(log_file)
    logger.info(f"Localizing function names of  {instance_id}")
    
    for file_name_json in file_names_json:
        if file_name_json["instance_id"] == instance_id:
            found_files = file_name_json["found_files"]
            structure = file_name_json["structrue"]
        
        
    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    problem_statement = bench_data["problem_statement"]
        
        
    fl = LocFuncName(
            instance_id,
            structure,
            problem_statement,
            args.model,
            args.backend,
            logger,
            args.match_partial_paths,
        )
    
    additional_artifact_loc_related = []
    found_related_locs = []
    related_loc_traj = {}

    (
        found_related_locs,
        additional_artifact_loc_related,
        related_loc_traj,
    ) = fl.localize_function_names_with_LLM(
        found_files,
    )
    
    with open(args.output_file, "a") as f:
        f.write(
            json.dumps(
                {   "instance_id": bug["instance_id"],
                    "found_files": found_files,
                    "found_related_locs": found_related_locs,
                    "additional_artifact_loc_related": additional_artifact_loc_related,
                    "related_loc_traj": related_loc_traj,
                }
            )
            + "\n"
        )



def localize_function_names(args):
    swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    #swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    valid_instance_ids = [
        "astropy__astropy-8707",       
    ]
    
    
    for bug in swe_bench_data:
        
        if bug["instance_id"] not in valid_instance_ids:
                continue
            
        localize_function_names_instance(
                bug, args, swe_bench_data,
            )







def main(input_file,output_file):
    class Args:
        pass
    
    args = Args()
    args.output_folder = '/home/wsl/AgentlessMCTS/Agentless/0822_MCTS/location_function' 
    #args.output_file = "loc_function_names.jsonl"
    args.output_file = output_file
    #args.input_file_names_folder = "/home/wsl/AgentlessMCTS/Agentless/0822_MCTS/location_file/loc_file_names.jsonl"
    args.input_file_names_folder = "/home/wsl/AgentlessMCTS/Agentless/0822_MCTS/location_file"
    args.input_file_names = os.path.join(args.input_file_names_folder, input_file)
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
    
    if not os.path.exists(os.path.join(args.output_folder, "localization_function_names_logs")):
        os.makedirs(os.path.join(args.output_folder, "localization_function_names_logs"))
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        
    localize_function_names(args)

    

if __name__ == "__main__":
    main()
