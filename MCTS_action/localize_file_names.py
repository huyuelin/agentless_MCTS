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
    def localize_file_names_with_LLM(self, top_n=1) -> tuple[list, list, list, any]:
        pass

class LocFileNames(FL):
    localize_file_names_prompt = """
Please look through the following GitHub problem description and Repository structure and provide a list of files that one would need to edit to fix the problem.

### GitHub Problem Description ###
{problem_statement}

###

### Repository Structure ###
{structure}

###

Please only provide the full path and return at most 1 files.
The returned files should be separated by new lines ordered by most to least important and wrapped with ```
For example:
```
file1.py
file2.py
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
        
    def localize_file_names_with_LLM(self, top_n=1, match_partial_paths=False)-> tuple[list, list, list, any]:
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model
        
        found_files = []
        message = self.localize_file_names_prompt.format(
            problem_statement=self.problem_statement,
            structure=show_project_structure(self.structure).strip(),
        ).strip()
        self.logger.info(f"prompting with message:\n{message}")
        print(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)
        print("=" * 80)
        
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
        model_found_files = self._parse_model_return_lines(raw_output)

        files, classes, functions = get_full_file_paths_and_classes_and_functions(
            self.structure
        )
        
        # sort based on order of appearance in model_found_files
        found_files = correct_file_paths(model_found_files, files, match_partial_paths)

        self.logger.info(raw_output)
        print(raw_output)
        
        return (
            found_files,
            {"raw_output_files": raw_output},
            traj,
        )
    

def localize_file_names_instance(bug, args, swe_bench_data):
    instance_id = bug["instance_id"]
         
    log_file = os.path.join(
        args.output_folder, "localization_file_names_logs", f"{instance_id}.log"
    )
    
    if os.path.exists(log_file):
        # 文件存在,清空内容
        open(log_file, 'w').close()
    else:
        # 文件不存在,创建目录
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = setup_logger(log_file)
    logger.info(f"Localizing file names of  {instance_id}")
    
    d = get_project_structure_from_scratch(
            bug["repo"], bug["base_commit"], bug["instance_id"], "playground"
        )
    
    logger.info(f"================ localize file names {instance_id} ================")
    
    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    problem_statement = bench_data["problem_statement"]
    structure = d["structure"]

    filter_none_python(structure)  # some basic filtering steps
    
    if not d["instance_id"].startswith("pytest"):
        filter_out_test_files(structure)
        
    found_files = []
    additional_artifact_loc_file = None
    file_traj = {}
    
    LCN = LocFileNames(
            d["instance_id"],
            structure,
            problem_statement,
            args.model,
            args.backend,
            logger,
            args.match_partial_paths,
        )
    
    found_files, additional_artifact_loc_file, file_traj = LCN.localize_file_names_with_LLM()
    
    with open(args.output_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": d["instance_id"],
                    "found_files": found_files,
                    "additional_artifact_loc_file": additional_artifact_loc_file,
                    "file_traj": file_traj,
                    "structrue": structure,
                }
            )
            + "\n"
        )
    
    
    

def localize_file_names(args):
    swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    #swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    valid_instance_ids = [
        "astropy__astropy-8707",       
    ]
    
    
    for bug in swe_bench_data:
        
        if bug["instance_id"] not in valid_instance_ids:
                continue
            
        localize_file_names_instance(
                bug, args, swe_bench_data,
            )

def main(output_file):
    class Args:
        pass
    
    args = Args()
    args.output_folder = '/home/wsl/AgentlessMCTS/Agentless/0822_MCTS/location_file' 
    #args.output_file = "loc_file_names.jsonl"
    args.output_file = output_file
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
    
    
    if not os.path.exists(os.path.join(args.output_folder, "localization_file_names_logs")):
        os.makedirs(os.path.join(args.output_folder, "localization_file_names_logs"))
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        
    localize_file_names(args)



if __name__ == "__main__":
    main()
