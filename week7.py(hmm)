import nltk

from nltk.tag import hmm

from nltk.corpus import treebank

nltk.download('treebank')

nltk.download('universal_tagset')

data = treebank.tagged_sents(tagset='universal')

train_data = data[:3500]

test_data = data[3500:]

trainer = hmm.HiddenMarkovModelTrainer()

hmm_tagger = trainer.train_supervised(train_data)

sentence = "The cat sat on the mat".split()

tags = hmm_tagger.tag(sentence)

print("Tagged Sentence ",tags)
