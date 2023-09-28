import polars

"""
train.csv has some latin1 encodings that are outside standard utf8
"""
train_df = polars.read_csv("train.csv",encoding='utf8-lossy',eol_char='\r')
test_df = polars.read_csv("test.csv",eol_char='\r')
valid_df = polars.read_csv("valid.csv",eol_char='\r')

def explore(df):
  print("Shape:", df.shape)
  print("Columns:", df.columns)

print("Train DF")
explore(train_df)
print("----------")
print("Test DF")
explore(test_df)
print("----------")
print("Validation DF")
explore(valid_df)

def explore_claims_questions(df):
  print("Number of claims:",df["claim_reviewed"].len())
  print("Number of unique claims:",df["claim_reviewed"].n_unique())
  print("Number of empty claims", train_df["claim_reviewed"].null_count())
  n_chars = df["claim_reviewed"].unique().str.n_chars()
  print(f"Median, max, min chars per claim: [{n_chars.median()}, {n_chars.max()}, {n_chars.min()}]")
  print("Number of questions:",df["question"].len())
  print("Number of unique questions:",df["question"].n_unique())
  print("Number of empty questions", train_df["question"].null_count())
  print("Ratio of quetions per claim:",df["question"].n_unique()/df["claim_reviewed"].n_unique())
  n_chars = df["question"].unique().str.n_chars()
  print(f"Median, max, min chars per question: [{n_chars.median()}, {n_chars.max()}, {n_chars.min()}]")

print("Train DF")
explore_claims_questions(train_df)
print("----------")
print("Test DF")
explore_claims_questions(test_df)
print("----------")
print("Validation DF")
explore_claims_questions(valid_df)

"""
If we analyse the statistics we can see that the data contains some rare cases like:
- Questions that come from an empty claims
- Claims that do not have questions
- Very long claims and very short claims (1 word or 2 words)
- Long questions that came from very short claims
- Questions that do not have any relation with the claim
"""

res = train_df.sort(polars.col("claim_reviewed").str.n_chars(),descending=[True])
print("Examples of extrem cases in the data set")
print(res["claim_reviewed"][0])
print(res["claim_reviewed"][-1],"--", res["question"][-1])
print(res["claim_reviewed"][-3],"--", res["question"][-3])


