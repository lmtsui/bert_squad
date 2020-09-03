# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""

import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm_notebook

from transformers import (
    AdamW,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)

from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor

def load_and_cache_examples(tokenizer, is_training=True):
    # Load data features from cache or dataset file
    cached_features_file = "cached_{}".format("train" if is_training else "dev")

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file):
        print("Loading features from cached file ", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        print("Creating features from dataset file")
        
        if is_training:
            examples = SquadV1Processor().get_train_examples('')
        else:
            examples = SquadV1Processor().get_dev_examples('')

        features, dataset = squad_convert_examples_to_features(
            examples,tokenizer,max_seq_length,
            doc_stride=128,
            max_query_length=64,
            is_training=is_training,
            return_dataset="pt")

        print("Saving features into cached file", cached_features_file)
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    return dataset, examples, features

def train(train_dataset, model, tokenizer):
    # Training
""" Train the model """
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=12)
epochs = 2
t_total = len(train_dataloader) * epochs

# Prepare optimizer and schedule (linear warmup and decay)
optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

# Train!
print("***** Running training *****")
print("  Num examples = ", len(train_dataset))
print("  Total optimization steps = ", t_total)

global_step = 1
tr_loss = 0.0
model.zero_grad()

for epoch in range(epochs):
    print('Epoch:{}'.format(epoch+1))
    epoch_iterator = tqdm_notebook(train_dataloader, desc="Iteration", disable=False)
    for step, batch in enumerate(epoch_iterator):

        model.train()
        batch = tuple(t.to(device) for t in batch)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }

        outputs = model(**inputs)
        # model outputs are always tuple in transformers (see doc)
        loss = outputs[0]

        loss.backward()

        tr_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        global_step += 1
        
        # Log metrics
        if global_step % 50 == 0:
            # Only evaluate when single GPU otherwise metrics may not average well
            print('Global step = {}, logging_loss = {}'.format(global_step,tr_loss))

        return global_step, tr_loss / global_step

def evaluate(model, tokenizer):
    # Evaluate
    dataset, examples, features = load_and_cache_examples(tokenizer, is_training=False)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=8)

    # Eval!
    print("***** Running evaluation *****")
    print("  Num examples = ", len(dataset))

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm_notebook(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]

            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            output = [output[i].detach().cpu().tolist() for output in outputs]
            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)
            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    print("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        n_best_size = 20,
        max_answer_length = 30,
        do_lower_case=False,
        output_prediction_file="predictions.json",
        output_nbest_file="nbest_predictions.json",
        output_null_log_odds_file=None,
        verbose_logging=False,
        version_2_with_negative=False,
        null_score_diff_threshold=0.0,
    )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)

    return results

def main():
   # Set seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda")

    config = BertConfig.from_pretrained('bert-base-cased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased',do_lower_case=True,)
    #the nn.module BertForQuestionAnswering has a single untrained layer qa_output: Linear(hidden_size,2) on top of the trained BERT-base.
    model = BertForQuestionAnswering.from_pretrained('bert-base-cased',config=config,)

    model.to(device)

    max_seq_length=384

    train_dataset = load_and_cache_examples(tokenizer, is_training=True)[0]

    # Training
    global_step, ave_loss = train(train_dataset, model, tokenizer)
    print(" global_step = %s, average loss = %s", global_step, tr_loss / global_step)

    # Save the trained model and the tokenizer
    output_dir = 'output/'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model checkpoint to %s", output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Load a trained model and vocabulary that you have fine-tuned
    model = BertForQuestionAnswering.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=True)
    model.to(device)

    # Evaluate
    results = evaluate(model, tokenizer)
    print("Results: {}".format(results))

    return result

def predict(max_seq_length=384,tokenizer,model,q,doc,device):
    indexed_tokens = tokenizer.encode(q,doc)
    attention_mask = [1]*len(indexed_tokens)
    seg_idx = indexed_tokens.index(102)+1
    segment_ids = [0]*seg_idx+[1]*(len(indexed_tokens)-seg_idx)

    #padding
    indexed_tokens += [0]*(max_seq_length-len(indexed_tokens))
    attention_mask += [0]*(max_seq_length-len(attention_mask))
    segment_ids += [0]*(max_seq_length-len(segment_ids))
    
    # for debugging
    # ind2word = {v:k for k,v in tokenizer.vocab.items()}
    # [ind2word[ind] for ind in indexed_tokens]

    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segment_tensor = torch.tensor([segment_ids]).to(device)
    attention_tensor = torch.tensor([attention_mask]).to(device)

    # Predict the start and end positions logits
    with torch.no_grad():
        start_logits, end_logits = model(tokens_tensor, token_type_ids=segment_tensor, attention_mask=attention_tensor)

    # get the highest prediction
    answer = tokenizer.decode(indexed_tokens[torch.argmax(start_logits):torch.argmax(end_logits)+1])
    return answer

if __name__ == "__main__":
    main()
