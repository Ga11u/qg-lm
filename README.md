# LM for question genration
## Dependencies
- transformers
- datasets
- evaluate
- accelerate
- rouge_score
- polars
- einops

You can install them with:
```sh
pip install transformers datasets evaluate accelerate rouge_score polars einops
```

## Files

- `data_exploration.py` : Provides a glimpse of the data and data quality
- `T5_training.py` : Fine tunes a T5-small pre-trained model
- `BART_training.py` : Fine tunes a BART-base pre-trained model
-  `FALCON_instruct_LM.py`: FALCON as LM can be used for both question generation and data generation. It actually seems to works quite well for generating questions with the provided promp (not evaluated).

## Results

Training results were evaluated with BLEU [0,1] and ROUGE [0,100].

|Model|BLEU|ROUGE 1|ROUGE 2|ROUGE L|
|-----|-----|---|-----|----|
|T5|0|0.027400 | 0.011600 | 0.024600 |
|BART|0.092000| 0.338100| 0.14360 | 0.310400 |
