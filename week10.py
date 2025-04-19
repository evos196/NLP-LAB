import nltk
from nltk import pos_tag, RegexpParser
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')
nltk.download('punkt_tab')

def chunker(text):
  words = word_tokenize(text)
  tagged_words = pos_tag(words)
  chunk_grammar =r"""
    NP: {<DT>?<JJ>*<NN>}
    PP: {<IN><NP>}
    VP: {<VB.*><NP|PP>+$}
  """
  chunk_parser = RegexpParser(chunk_grammar)
  chunked_text = chunk_parser.parse(tagged_words)
  return chunked_text

text = "The quick brown fox jumps over the lazy dog"
result = chunker(text)
print(result)
