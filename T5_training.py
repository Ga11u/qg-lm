import polars
import transformers
from transformers import T5ForConditionalGeneration,T5TokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
import gc
import accelerate
from datasets import Dataset
import evaluate

MAX_TOKENS = 2**7

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5TokenizerFast.from_pretrained("t5-base")

"""
Loading the datasets (it is asumed that the data sets are in the same directory)
"""
train_df = polars.read_csv("train.csv",encoding='utf8-lossy',eol_char='\r') # utf8-lossy as there is some error with Utf-8 enconding in some rows
test_df = polars.read_csv("test.csv",eol_char='\r')
valid_df = polars.read_csv("valid.csv",eol_char='\r')

"""
Cleaning unnecessary columns, removing some data and creating the Dataset for training
"""
train_df=train_df.select(["claim_reviewed","question"])
test_df=test_df.select(["claim_reviewed","question"])
valid_df=valid_df.select(["claim_reviewed","question"])

train_df=train_df.drop_nulls(["claim_reviewed","question"])
test_df=test_df.drop_nulls(["claim_reviewed","question"])
valid_df=valid_df.drop_nulls(["claim_reviewed","question"])

train_ds=Dataset.from_pandas(train_df.to_pandas())
test_ds=Dataset.from_pandas(test_df.to_pandas())
valid_ds=Dataset.from_pandas(valid_df.to_pandas())


def answer_question_generator(batch):
    claims = [claim + '</s>' for claim in batch["claim_reviewed"]]
    questions = [question + '</s>' for question in batch["question"]]
    input_encodings  = tokenizer(claims,padding='max_length',truncation=True,return_tensors="pt",max_length=MAX_TOKENS)
    target_encodings = tokenizer(text_target = questions, padding='max_length',truncation=True,return_tensors="pt",max_length=MAX_TOKENS) 
    return {"input_ids": input_encodings["input_ids"], 
           "attention_mask": input_encodings["attention_mask"], 
           "labels": target_encodings["input_ids"]}

train_data = train_ds.map(answer_question_generator,batched=True)
train_data = train_data.remove_columns(["claim_reviewed","question"])
test_data = test_ds.map(answer_question_generator, batched=True)
test_data = test_data.remove_columns(["claim_reviewed","question"])
valid_data = valid_ds.map(answer_question_generator, batched=True)
valid_data = valid_data.remove_columns(["claim_reviewed","question"])


def compute_metrics(eval_preds):
  bleu = evaluate.load('bleu')
  rouge = evaluate.load('rouge')
  labels_ids = eval_preds.label_ids
  pred_ids = eval_preds.predictions
  pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
  label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
  bleu_output = bleu.compute(predictions=pred_str, references=label_str)
  rouge_output = rouge.compute(predictions=pred_str, references=label_str,use_stemmer=True)
  return {
        'bleu': round(bleu_output["bleu"], 4),
        'rouge1': round(rouge_output['rouge1'], 4),
        'rouge2': round(rouge_output['rouge2'], 4),
        'rougeL': round(rouge_output['rougeL'], 4)
    }


training_args = Seq2SeqTrainingArguments(
    output_dir="results",
    warmup_steps=500,
    #label_smoothing_factor=0.1,
    predict_with_generate=True,
    evaluation_strategy="steps",
    save_strategy = "steps",
    learning_rate=1e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    auto_find_batch_size = True,
    fp16 = True,
    fp16_full_eval = True,
    metric_for_best_model = "bleu",
    generation_max_length = MAX_TOKENS,
    load_best_model_at_end = True
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_data,
    eval_dataset=valid_data,
    compute_metrics = compute_metrics,
)

torch.cuda.empty_cache()
gc.collect()

trainer.train()
