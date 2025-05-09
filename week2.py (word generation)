!pip install nltk
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import random
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
def generate_words(text,num_words=10):
  words=word_tokenize(text.lower())
  stop_words=set(stopwords.words('english'))
  words=[word for word in words if word.isalnum() and word not in stop_words]
  freq_dist=FreqDist(words)
  generate_words=[]
  for _ in range(num_words):
    generate_words.append(random.choice(list(freq_dist.keys())))
  return generate_words
if __name__=="__main__":
  text="Natural language is a field of Artificial Intelligence which focuses on interaction between computers and human using natural language"
  generated_words= generate_words(text,num_words=5)
  print("Generated words:",generated_words)
