# RLHF pipeline for the creation of StackLLaMa: a Stack exchange llama-7b model.
There were three main steps to the training process:
1. Supervised fine-tuning of the base llama-7b model to create llama-7b-se:
2. Reward modeling using dialog pairs from the SE dataset using the llama-7b-se to create llama-7b-se-rm:
3. RL fine-tuning of llama-7b-se with the llama-7b-se-rm reward model:

For all methods use `python run.py -e configs/` and choose the corresponding config


## Pretrained Models

My LoRA layers for the vanilla StackLLaMA are publicly available on huggingface as 
- [`mnoukhov/llama-7b-se-peft`](https://huggingface.co/mnoukhov/llama-7b-se-peft)
- [`mnoukhov/llama-7b-se-rm-peft`](https://huggingface.co/mnoukhov/llama-7b-se-rm-peft)
- [`mnoukhov/llama-7b-se-rl-peft`](https://huggingface.co/mnoukhov/llama-7b-se-rl-peft)

LoRA layers were using at all stages to reduce memory requirements. 
At each stage the peft adapter layers were merged with the base model, using: 
```shell
python examples/stack_llama/scripts/merge_peft_adapter.py --adapter_model_name=XXX --base_model_name=YYY --output_name=ZZZ
```

I used `huggyllama/llama-7b` as the base model. Note the order that models must be merged:

1. llama-7b-se = merge_peft_adapter llama-7b + llama-7b-se-peft
2. llama-7b-se-rm = merge_peft_adapter llama-7b-se + llama-7b-se-rm-peft
3. llama-7b-se-rl = merge_peft_adapter llama-7b-se + llama-7b-se-rl-peft
