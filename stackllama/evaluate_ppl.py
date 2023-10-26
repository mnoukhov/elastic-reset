from dataclasses import dataclass, field
from typing import Optional

import evaluate
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from perplexity import batched_perplexity, eval_perplexity, unbatched_perplexity
from supervised_finetuning import (
    ConstantLengthDataset,
    chars_token_ratio,
    prepare_sample_text,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
)


@dataclass
class ScriptArguments:
    model: Optional[str] = field(
        default="llama-7b-sft",
        metadata={"help": "the model path"},
    )
    is_peft: Optional[bool] = field(default=False)
    tokenizer: Optional[str] = field(default="huggyllama/llama-7b")
    dataset_name: Optional[str] = field(default="lvwerra/stack-exchange-paired")
    subset: Optional[str] = field(default="data/evaluation")
    split: Optional[str] = field(default="train")  # there is no test data
    data_size: Optional[int] = field(default=4000)
    seq_length: Optional[int] = field(default=1024)
    stride: Optional[int] = field(default=512)
    seed: Optional[int] = field(default=0)
    batch_size: Optional[int] = field(default=2)
    bit8: Optional[bool] = field(default=True)


def main(args):
    if args.is_peft:
        config = PeftConfig.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            load_in_8bit=True,
            device_map={"": 0},
        )
        model = PeftModel.from_pretrained(model, args.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            load_in_8bit=args.bit8,
            device_map={"": 0},
        )

    # config = AutoConfig.from_pretrained(args.model_path)

    # with init_empty_weights():
    #     model = AutoModelForCausalLM.from_config(
    #         config,
    #         trust_remote_code=True,
    #     )
    #     model.tie_weights()

    # model = load_checkpoint_and_dispatch(model, config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
    )

    if args.data_size is not None:
        dataset = dataset.select(range(args.data_size))

    texts = [prepare_sample_text(sample) for sample in dataset]

    if args.batch_size == 1:
        ppl = unbatched_perplexity(
            texts, model, tokenizer, max_length=args.seq_length, stride=args.stride
        )
    else:
        ppl = batched_perplexity(
            texts,
            model,
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.seq_length,
            stride=args.stride,
        )

    print(f"Perplexity {ppl}")


def main_dict(dict_args):
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_dict(dict_args)[0]
    main(args)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
