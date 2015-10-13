from corpus import Document, NamesCorpus, ReviewCorpus
from maxent import MaxEnt
from unittest import TestCase, main
from random import shuffle, seed
import sys


class BagOfWords(Document):
    def features(self):
        """Trivially tokenized words."""
        return self.data.split()

class Review(Document):
    def raw_features(self):
        """Return raw (dense) features"""
        # return [word for word in self.data.split() if len(word)>2] + ['***BIAS TERM***']
        return [word[:4] for word in stripPunctuation(self.data).lower().split()] + ['***BIAS TERM***']

def stripPunctuation(text):
    from unicodedata import category
    return ''.join(ch for ch in text if category(ch)[0] != 'P')

class Name(Document):
    def raw_features(self):
        name = self.data
        return ['First=%s' % name[0], 'Last=%s' % name[-1]] + ['***BIAS TERM***']

def accuracy(classifier, test, verbose=sys.stderr):
    correct = [classifier.classify(x) == x.label for x in test]
    if verbose:
        print >> verbose, "%.2d%% " % (100 * sum(correct) / len(correct)),
    return float(sum(correct)) / len(correct)

class MaxEntTest(TestCase):
    u"""Tests for the MaxEnt classifier."""

    # def split_names_corpus(self, document_class=Name):
    #     """Split the names corpus into training, dev, and test sets"""
    #     names = NamesCorpus("names/*.txt", 0, document_class=document_class)
    #     self.assertEqual(len(names), 5001 + 2943) # see names/README
    #     seed(hash("names"))
    #     shuffle(names)
    #     return (names[:5000], names[5000:6000], names[6000:], 
    #         names.label_set, names.features_set)

    # def test_names_nltk(self):
    #     """Classify names using NLTK features"""
    #     train, dev, test, labels, features = self.split_names_corpus()
    #     classifier = MaxEnt()
    #     classifier.train(train, labels, features, dev)
    #     acc = accuracy(classifier, test)
    #     self.assertGreater(acc, 0.70)

    def split_review_corpus(self, document_class):
        """Split the yelp review corpus into training, dev, and test sets"""
        training_size = 10000
        dev_size = 1000
        test_size = 3000
        total_records = training_size + dev_size + test_size
        reviews = ReviewCorpus('yelp_reviews.json', total_records, document_class=document_class)
        seed(hash("reviews"))
        shuffle(reviews)
        return (reviews[:training_size], reviews[training_size:training_size+test_size],
        reviews[training_size+test_size:total_records],
            reviews.label_set, reviews.features_set)

    def test_reviews_bag(self):
        """Classify sentiment using bag-of-words"""
        train, dev, test, labels, features = self.split_review_corpus(Review)
        classifier = MaxEnt()
        classifier.train(train, labels, features, dev)
        classifier.save("mikes_model")

        class2 = MaxEnt()
        class2.load("mikes_model")
        self.assertGreater(accuracy(classifier, test), 0.55)

if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)
