import nltk
import numpy as np
import string

class LanguageGenerator:
    def __init__(self,texts, n=5, topk=10000):
        c = nltk.FreqDist(nltk.ngrams(texts.lower(), n)).most_common(topk)

        self.n = n
        self.grams = ["".join(x[0]) for x in c]
        self.counts = np.array([x[1] for x in c]) # saving the probabilities as numpy array
        self.counts = self.counts / self.counts.sum()

    def generate(self,text):
        """Generate the next character in the sequence"""

        # Get last n-1 characters of the input
        prefix = text[-(self.n-1):]

        # Find all the ngrams starting with the prefix
        indices = [i for i, gram in enumerate(self.grams) if gram.startswith(prefix)]

        if len(indices) == 0:
            # If there are no ngrams starting with the ngrams
            character = np.random.choice(list(string.ascii_lowercase))
            text = text + character
        else:
            probs = self.counts[indices]
            probs = probs / probs.sum() # Renormalize

            # Sample
            selection = np.random.choice(indices, p=probs)
            #selection = np.argmax(indices)
            #print(selection)

            text = text + self.grams[selection][-1]
        return text

    def generate_n(self, text, n = 100):
        for _ in range(n):
            text = self.generate(text)
        return text

english_text = nltk.corpus.gutenberg.raw(nltk.corpus.gutenberg.fileids())#[:200000]
print(english_text[1:3000])

langen = LanguageGenerator(english_text, n=5, topk=100000)
text = langen.generate_n(english_text, 1000)
print(text)
