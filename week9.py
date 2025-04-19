import spacy
def pos_tagger_spacy(text):
  nlp = spacy.load("en_core_web_sm")
  doc = nlp(text)

  tagged_words = [(token.text,token.pos_) for token in doc]
  return tagged_words

if __name__ == "__main__":
  text = input("Enter the text")
  tagged_result = pos_tagger_spacy(text)

  print("Input Text",text)
  print("\n POS Tagged :")
  for word, pos in tagged_result:
    print(f"{word}: {pos}")
