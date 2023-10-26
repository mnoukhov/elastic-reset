"""Modified from codeparrot human_eval script"""

import json
import multiprocessing
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    StoppingCriteria,
    StoppingCriteriaList,
)


EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]


@dataclass
class HumanEvalArguments:
    """
    Configuration for running evaluation on HumanEval dataset.
    """

    model_ckpt: Optional[str] = field(
        default="llama-7b-sft",
        metadata={"help": "Model name or path of model to be evaluated."},
    )
    tokenizer: Optional[str] = field(
        default="huggyllama/llama-7b",
        metadata={"help": "Tokenizer name or path of model to be evaluated."},
    )

    num_workers: Optional[int] = field(
        default=None, metadata={"help": "Number of workers used for code evaluation."}
    )
    num_tasks: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of human-eval tasks to run. If not included all tasks are evaluated."
        },
    )
    do_sample: Optional[bool] = field(
        default=True,
        metadata={"help": "Sample from the language model's output distribution."},
    )
    load_8bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Load mode in 8bit"},
    )
    temperature: Optional[float] = field(
        default=0.2, metadata={"help": "Sampling temperature used for generation."}
    )
    max_new_tokens: Optional[int] = field(
        default=256, metadata={"help": "Maximum number of newly generated tokens."}
    )
    top_k: Optional[int] = field(
        default=0, metadata={"help": "Top-k parameter used for generation."}
    )
    top_p: Optional[float] = field(
        default=0.95, metadata={"help": "Top-p parameter used for nucleus sampling."}
    )
    batch_size: Optional[int] = field(
        default=10, metadata={"help": "Number of generations to run in parallel."}
    )
    n_samples: Optional[int] = field(
        default=200,
        metadata={"help": "Number of completions to generate for each sample."},
    )
    seed: Optional[int] = field(
        default=1, metadata={"help": "Random seed used for evaluation."}
    )
    output_file: Optional[str] = field(
        default="eval_results.json",
        metadata={"help": "Random seed used for evaluation."},
    )
    HF_ALLOW_CODE_EVAL: Optional[str] = field(
        default="0",
        metadata={"help": "Allow `code_eval` to execute Python code on machine"},
    )
    device_int: Optional[int] = field(
        default=-1,
        metadata={
            "help": (
                "Determine which device to run the `text-generation` Pipeline on. -1 is CPU and any zero or positive"
                " number corresponds to which GPU device id to run on."
            )
        },
    )


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially.
    See compute_code for more details.
    """

    def __init__(self, tokenizer, dataset, n_tasks=None, n_copies=1):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.n_tasks = len(dataset) if n_tasks is None else n_tasks
        self.n_copies = n_copies

    def __iter__(self):
        prompts = []
        for task in range(self.n_tasks):
            # without strip, the model generate commented codes ...
            prompts.append(
                self.tokenizer.eos_token + self.dataset[task]["prompt"].strip()
            )
        outputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
        for task in range(self.n_tasks):
            for _ in range(self.n_copies):
                yield {
                    "ids": outputs.input_ids[task],
                    "task_id": task,
                    "input_len": outputs.attention_mask[task].sum(),
                }


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any(
                    [
                        stop_string in decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done)


def remove_last_block(string):
    """Remove the last block of the code containing EOF_STRINGS"""
    string_list = re.split("(%s)" % "|".join(EOF_STRINGS), string)
    # last string should be ""
    return "".join(string_list[:-2])


def complete_code(
    accelerator, model, tokenizer, dataloader, n_tasks, batch_size=20, **gen_kwargs
):
    """Generate multiple codes for each task in the dataset. This function leverage accelerator to distribute
    the processing to multiple GPUs.
    dataloader, a wrapper around a TokenizeDataset objectm is supposed to send all the prompts from
    the evalution dataset to the modelm as the following:
    [p_0_0, p_0_1, ..., p_0_nc-1, p_1_0, ..., p_nt-1_nc-1]
    where nc is the number of copies of the prompt, and nt is the number of tasks.
    nc is such that num_sample = nc * batch_size

    Parameters
    ----------
    accelerator: Accelerator

    model: transformers.PreTrainedModel
        Code generation model. AutoTokenizer.from_pretrained(model_ckpt), ex model_ckpt = "lvwerra/codeparrot"

    tokenizer: transformers.AutoTokenizer
        The tokenizer used to train model

    dataloader: DataLoader
        The dataloader is a wrapper around a TokenizeDataset object. It is designed to be used with multiple GPUs.

    n_tasks: int
        The number of tasks in the dataset. It is used to determine the length of the output.
        Should be aligned with the number of tasks in the TokenizeDataset.

    batch_size: int
        num_return_sequences per copy of the prompt such that num_sample = batch_size * n_copies

    gen_kwargs: dict
        Keyword arguments for the generation function of the model.

    Returns
    -------
    code_gens: list of list of str, of length n_tasks
        List of generated codes for each task.
        Each element is a list of generated codes for each task, with length num_samples
    """
    gen_token_dict = defaultdict(list)  # dict of list of generated tokens
    for step, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            gen_kwargs["stopping_criteria"][0].start_length = batch["ids"].shape[-1]
            generated_tokens = accelerator.unwrap_model(model).generate(
                input_ids=batch["ids"][:, : batch["input_len"]],
                num_return_sequences=batch_size,
                **gen_kwargs,
            )
            # each task is generated batch_size times
            generated_tasks = batch["task_id"].repeat(batch_size)
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens, generated_tasks = accelerator.gather(
                (generated_tokens, generated_tasks)
            )
            generated_tokens = generated_tokens.cpu().numpy()
            generated_tasks = generated_tasks.cpu().numpy()

            for task, generated_tokens in zip(generated_tasks, generated_tokens):
                gen_token_dict[task].append(generated_tokens)

    code_gens = [[] for _ in range(n_tasks)]
    for task, generated_tokens in gen_token_dict.items():
        for s in generated_tokens:
            gen_code = tokenizer.decode(
                s, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            code_gens[task].append(remove_last_block(gen_code))
    return code_gens


def main(args_dict=None):
    # Setup configuration
    parser = HfArgumentParser(HumanEvalArguments)
    if args_dict is None:
        args = parser.parse_args()
    else:
        args = parser.parse_dict(args_dict)[0]

    print(args)
    transformers.logging.set_verbosity_error()
    # enables code execution in code_eval metric
    os.environ["HF_ALLOW_CODE_EVAL"] = args.HF_ALLOW_CODE_EVAL
    # make sure tokenizer plays nice with multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.num_workers is None:
        args.num_workers = multiprocessing.cpu_count()

    # Use dataset load to feed to accelerate
    accelerator = Accelerator()
    set_seed(args.seed, device_specific=True)

    # Load model and tokenizer
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model_ckpt
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_ckpt,
        trust_remote_code=True,
        load_in_8bit=args.load_8bit,
        device_map="auto",
    )

    # Generation settings
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "stopping_criteria": StoppingCriteriaList(
            [EndOfFunctionCriteria(0, EOF_STRINGS, tokenizer)]
        ),
    }

    # Load evaluation dataset and metric
    human_eval = load_dataset("openai_humaneval")
    code_eval_metric = evaluate.load("code_eval")

    n_tasks = args.num_tasks if args.num_tasks is not None else len(human_eval["test"])
    n_copies = args.n_samples // args.batch_size

    human_eval_tokenized = TokenizedDataset(
        tokenizer, human_eval["test"], n_copies=n_copies, n_tasks=n_tasks
    )
    # do not confuse args.batch_size, which is actually the num_return_sequences
    human_eval_loader = DataLoader(human_eval_tokenized, batch_size=1)

    # Run a quick test to see if code evaluation is enabled
    try:
        _ = code_eval_metric.compute(references=[""], predictions=[[""]])
    except ValueError as exception:
        print(
            'Code evaluation not enabled. Read the warning below carefully and then use `--HF_ALLOW_CODE_EVAL="1"`'
            " flag to enable code evaluation."
        )
        raise exception

    model, human_eval_loader = accelerator.prepare(model, human_eval_loader)

    generations = complete_code(
        accelerator,
        model,
        tokenizer,
        human_eval_loader,
        n_tasks=n_tasks,
        batch_size=args.batch_size,
        **gen_kwargs,
    )

    if accelerator.is_main_process:
        references = []

        for task in tqdm(range(n_tasks)):
            test_func = human_eval["test"][task]["test"]
            entry_point = f"check({human_eval['test'][task]['entry_point']})"
            references.append("\n" + test_func + "\n" + entry_point)

        # Evaluate completions with "code_eval" metric
        pass_at_k, _ = code_eval_metric.compute(
            references=references, predictions=generations, num_workers=args.num_workers
        )
        print(f"Results: {pass_at_k}")

        # Save results to json file
        with open(args.output_file, "w") as fp:
            json.dump(pass_at_k, fp)


# For some reason the folliwng seems to be necessary sometimes for code_eval to work nice with multiprocessing
# https://stackoverflow.com/questions/60804599/python-multiprocessing-keeps-spawning-the-whole-script
if __name__ == "__main__":
    main()
