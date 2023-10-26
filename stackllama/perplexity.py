"""Evaluate's perplexity but we pass in the model and tokenizer"""

import numpy as np
import torch
import torch.nn.functional as F
from evaluate import logging
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


def eval_perplexity(
    predictions,
    model,
    tokenizer,
    batch_size: int = 16,
    add_start_token: bool = True,
    device=None,
    max_length=None,
):
    if device is not None:
        assert device in [
            "gpu",
            "cpu",
            "cuda",
        ], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer(
        predictions,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 1)
        ), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor(
                [[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)
            ).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [
                    torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device),
                    attn_mask,
                ],
                dim=1,
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (
                loss_fct(shift_logits.transpose(1, 2), shift_labels)
                * shift_attention_mask_batch
            ).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return np.mean(ppls)
    # return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


def batched_perplexity(texts, model, tokenizer, batch_size, max_length, stride):
    device = model.device
    tokenized_inputs = tokenizer(texts, truncation=False)["input_ids"]
    all_token_ids = []
    for tokenized_input in tokenized_inputs:
        all_token_ids.extend(tokenized_input + [tokenizer.eos_token_id])

    text_len = len(all_token_ids)
    token_ids = torch.LongTensor(all_token_ids)
    lls = []
    n_samples = 0

    for i in tqdm(range(0, text_len, batch_size * stride)):
        begin_locs, end_locs, trg_lens = [], [], []
        for j in range(batch_size):
            j = i + j * stride
            if j >= text_len:
                break
            begin_loc = max(j + stride - max_length, 0)
            end_loc = min(j + stride, text_len)
            trg_len = end_loc - j  # may be different from stride on last loop

            begin_locs.append(begin_loc)
            end_locs.append(end_loc)
            trg_lens.append(trg_len)

        input_ids = [token_ids[b:e] for b, e in zip(begin_locs, end_locs)]
        target_end_locs = [sen.size(0) for sen in input_ids]
        input_ids = [
            F.pad(sen, (0, max_length - sen.size(0)), "constant", 0)
            for sen in input_ids
        ]  # we dont need attention mask as long as these padded token is not involved in loss calculation
        input_ids = torch.stack(input_ids, dim=0).to(device)

        target_ids = (
            torch.ones_like(input_ids) * -100
        )  # -100 is the default ingore_index value in torch.nn.CrossEntropyLoss
        for index, (b, e) in enumerate(zip(trg_lens, target_end_locs)):
            labels = input_ids[index, -b:e].clone()
            target_ids[index, -b:e] = labels

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * sum(trg_lens)

        n_samples += input_ids.size(0)
        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_locs[-1])
    return ppl


def unbatched_perplexity(texts, model, tokenizer, max_length, stride):
    tokenized_inputs = tokenizer(texts, truncation=False)["input_ids"]
    all_token_ids = []
    for tokenized_input in tokenized_inputs:
        all_token_ids.extend(tokenized_input + [tokenizer.eos_token_id])

    token_ids = torch.LongTensor(all_token_ids)

    seq_len = len(token_ids)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = token_ids[begin_loc:end_loc].unsqueeze(0).to("cuda")
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())

    return ppl
