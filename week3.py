import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import wordnet
from nltk.stem import  WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('punkt_tab')
def morphological_analysis(text):
  words=word_tokenize(text)
  stop_words=set(stopwords.words('english'))
  words=[word for word in words if word.isalnum() and word not in stop_words]
  pos_tags=pos_tag(words)
  lemmatizer=WordNetLemmatizer()
  lemmatized_words=[lemmatizer.lemmatize(word,get_wordnet_pos(pos)) for word,pos in pos_tags]
  print("Orginal words:",words)
  print("Lemmatized words:",lemmatized_words)
def get_wordnet_pos(treebank_tag):
  if treebank_tag.startswith('J'):
    return 'a'
  elif treebank_tag.startswith('V'):
    return 'v'
  elif treebank_tag.startswith('N'):
    return 'n'
  elif treebank_tag.startswith('R'):
    return 'r'
  else:
    return 'n'
if __name__=="__main__":
  text=input("Enter the text:")
  morphological_analysis(text)
  for word,pos in pos_tag(word_tokenize(text)):
    print(f"Word:{word},POS TAG:{pos}, Wordnet POS Tag:{get_wordnet_pos(pos)}")
    
