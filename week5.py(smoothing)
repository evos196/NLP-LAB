import nltk
from nltk import ngrams
from collections import Counter
import re

def preprocess_text(text):
  text=re.sub(r'[^\w\s]','',text)
  text=text.lower()
  return text
def generate_bigrams(tokens):
  return list(zip(tokens,tokens[1:]))
def calculate_bigram_prob(corpus):
  bigrams=generate_bigrams(corpus)
  bigram_counts=Counter(bigrams)
  vocabulary_size=len(set(corpus))
  bigram_probs={}
  for bigram in bigram_counts:
    bigram_probs[bigram]=(bigram_counts[bigram]+1)/(corpus.count(bigram[0])+vocabulary_size)
  return bigram_probs
def bigram_smoothing(text):
  preprocessed_text=preprocess_text(text)
  tokens=preprocessed_text.split()
  bigram_prob=calculate_bigram_prob(tokens)
  print("Bigram Probs:")
  for bigram,prob in bigram_prob.items():
    print(f"{bigram}:{prob:.4f}")
if __name__=="__main__":
  text=input("Enter the text")
  bigram_smoothing(text)
  preprocess_text(text)
