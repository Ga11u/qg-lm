# LM for question genration
## Dependencies
- transformers
- datasets
- evaluate
- accelerate
- rouge_score
- polars

You can install them with:
```sh
pip install transformers datasets evaluate accelerate rouge_score polars
```

## Files

- `data_exploration.py` : Provides a glimpse of the data and data quality
- `T5_training.py` : Fine tunes a T5-small pre-trained model
- `BART_training.py` : Fine tunes a BART-base pre-trained model
