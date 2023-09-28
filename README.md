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

## Results

Training results were evaluated with BLUE [0,1] and ROUGE [0,100]. In both cases for BLUE the results were < 0.22 and ROUGE < 0.37 .
