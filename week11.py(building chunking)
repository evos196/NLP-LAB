import nltk
from nltk import RegexpParser
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Download the necessary resource
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')
nltk.download('punkt_tab')

text ="The quick brown fox jumps over the lazy dog"
words = word_tokenize(text)
tagged_words = pos_tag(words)

chunk_grammar = r"""
  NP: {<DT>?<JJ>*<NN>}
  PP: {<IN><NP>}
  VP: {<VB.*><NP|PP>+$}
  CLAUSE: {<NP><VP>}
"""

chunk_parser = RegexpParser(chunk_grammar)

chunks = chunk_parser.parse(tagged_words)

print(chunks)
