!pip install nltk
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def word_analysis(text):
  words=word_tokenize(text)
  word=[word.lower() for word in words]
  stop_words=set(stopwords.words('english'))
  words=[word for word in words if word.isalnum and word not in stop_words]
  freq_dist=FreqDist(words)

  print("total words:",len(words))
  print("Unique words:",len(freq_dist))
  print("Most common words:")
  print(freq_dist.most_common(10))

if __name__ == "__main__":
  text="Natural language processing (NLP) is a subfield of computer science and artificial intelligence (AI) that uses machine learning to enable computers to understand and communicate with human language."
  word_analysis(text)
