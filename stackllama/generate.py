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

# from accelerate import Accelerator
# from accelerate.utils import set_seed
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


def build_dataset(
    tokenizer,
    train_dataset,
    dataset_name="lvwerra/stack-exchange-paired",
    input_min_text_length=2,
    input_max_text_length=8,
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    # load imdb with datasets
    ds = load_dataset(dataset_name, data_dir="data/rl", split="train")
    original_columns = ds.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)

    ds.set_format(type="torch")
    return ds


@dataclass
class GenerateArguments:
    """
    Configuration for running evaluation on HumanEval dataset.
    """

    model: Optional[str] = field(
        default="llama-7b-se-rm",
        metadata={"help": "Model name or path of model to be evaluated."},
    )
    tokenizer: Optional[str] = field(
        default="huggyllama/llama-7b",
        metadata={"help": "Tokenizer name or path of model to be evaluated."},
    )

    num_workers: Optional[int] = field(
        default=None, metadata={"help": "Number of workers used for code evaluation."}
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
        default=0.9, metadata={"help": "Sampling temperature used for generation."}
    )
    max_new_tokens: Optional[int] = field(
        default=128, metadata={"help": "Maximum number of newly generated tokens."}
    )
    top_k: Optional[int] = field(
        default=0, metadata={"help": "Top-k parameter used for generation."}
    )
    top_p: Optional[float] = field(
        default=0.9, metadata={"help": "Top-p parameter used for nucleus sampling."}
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
    device_int: Optional[int] = field(
        default=-1,
        metadata={
            "help": (
                "Determine which device to run the `text-generation` Pipeline on. -1 is CPU and any zero or positive"
                " number corresponds to which GPU device id to run on."
            )
        },
    )


def main(args_dict=None):
    # Setup configuration
    parser = HfArgumentParser(GenerateArguments)
    if args_dict is None:
        args = parser.parse_args()
    else:
        args = parser.parse_dict(args_dict)[0]

    print(args)
    transformers.logging.set_verbosity_error()
    # make sure tokenizer plays nice with multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # if args.num_workers is None:
    #     args.num_workers = multiprocessing.cpu_count()

    # Use dataset load to feed to accelerate
    # accelerator = Accelerator()
    # set_seed(args.seed, device_specific=True)

    # Load model and tokenizer
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.pad_token = tokenizer.eos_token
    # current_device = Accelerator().local_process_index
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
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
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": 1.2,
        "eos_token_id": 100_000,
    }

    train_dataset = load_dataset(
        "lvwerra/stack-exchange-paired", data_dir="data/rl", split="train"
    )
    train_dataset = train_dataset.select(range(100))

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset(tokenizer, train_dataset)

    # def collator(data):
    #     return dict((key, [d[key] for d in data]) for key in data[0])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        # collate_fn=collator,
        shuffle=False,
    )

    for example in dataloader:
        # batch_inputs = torch.stack(batch["input_ids"])
        # print(example["query"])
        output = model.generate(
            example["input_ids"][0].unsqueeze(0).to("cuda"), **gen_kwargs
        )
        print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))


# For some reason the folliwng seems to be necessary sometimes for code_eval to work nice with multiprocessing
# https://stackoverflow.com/questions/60804599/python-multiprocessing-keeps-spawning-the-whole-script
if __name__ == "__main__":
    main()
