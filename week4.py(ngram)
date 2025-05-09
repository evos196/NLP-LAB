!pip install nltk
import nltk
from nltk import ngrams
from collections import Counter
import re

def clean_text(text):
  cleaned_text = re.sub(r'[^a-zA-Z0-9\s]',' ',text).lower()
  return cleaned_text

def ngram_analysis(text,n):
  cleaned_text = clean_text(text)
  words = cleaned_text.split()
  ngrams_list = list(ngrams(words,n))
  ngrams_count = Counter(ngrams_list)
  return ngrams_count

if __name__ == "__main__":
  text = input("Enter the text")

  n=2
  result=ngram_analysis(text,n)

  print(f"{n}-Gram Analysis:")
  for ngram, count in result.items():
    print(f"{ngram}:{count} times")
