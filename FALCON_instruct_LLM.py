from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import einops

model = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
    )

claim = "The current prime minister of Norway is Jonas Gahr Støre."
question_generation = f"Give me a question to this answer: {claim}"

data_generation = """You are asked to come up with one pair of question and answer.

Here are the requirements:
1. The answer must be at least 1 sentence long.
2. The question and answer must be in English.
3. The answer must be newsworthy.
4. The answer should provide additional information to reply the question.
5. The answer must be correct.
"""


sequences = pipeline(
    claim, #change this with data_generation for generating data
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=5,
    eos_token_id=tokenizer.eos_token_id,
    return_full_text=False
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
