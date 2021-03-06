# -*- mode: Python; coding: utf-8 -*-

"""For the purposes of classification, a corpus is defined as a collection
of labeled documents. Such documents might actually represent words, images,
etc.; to the classifier they are merely instances with features."""

from abc import ABCMeta, abstractmethod
from csv import reader as csv_reader
from glob import glob
import json
from os.path import basename, dirname, split, splitext
import numpy
from random import randint

class Document(object):
    """A document completely characterized by its features."""

    max_display_data = 10 # limit for data abbreviation

    def __init__(self, data, label=None, source=None):
        self.data = data
        self.label = label
        self.source = source
        self.feature_vector = None

    def __repr__(self):
        return ("<%s: %s>" % (self.label, self.abbrev()) if self.label else
                "%s" % self.abbrev())

    def abbrev(self):
        return (self.data if len(self.data) < self.max_display_data else
                self.data[0:self.max_display_data] + "...")

    def features(self):
        """A list of features that characterize this document."""
        return self.feature_vector

class Corpus(object):
    """An abstract collection of documents."""

    __metaclass__ = ABCMeta
    features_set = {}
    label_set = {}

    def __init__(self, datafiles, requested_records, document_class=Document):
        self.documents = []
        self.datafiles = glob(datafiles)
        for datafile in self.datafiles:
            self.load(datafile, requested_records, document_class)

        """Perform an initial pass over training data to gether labels and features"""
        self.features_set = {}
        self.label_set = {}
        for doc in self.documents:
            if not doc.label in self.label_set:
                self.label_set[doc.label] = len(self.label_set)
            for feat in doc.raw_features():
                if not feat in self.features_set:
                    self.features_set[feat] = len(self.features_set)

        print "Feature set size: %d" % len(self.features_set)
        """Now we have the full set of features and labels with indexes"""
        """Set the features vector for each document"""
        for doc in self.documents:
            doc.feature_vector = numpy.zeros( len(self.features_set) )
            for f in doc.raw_features():
                doc.feature_vector[self.features_set[f]] = 1


    # Act as a mutable container for documents.
    def __len__(self): return len(self.documents)
    def __iter__(self): return iter(self.documents)
    def __getitem__(self, key): return self.documents[key]
    def __setitem__(self, key, value): self.documents[key] = value
    def __delitem__(self, key): del self.documents[key]

    @abstractmethod
    def load(self, datafile, requested_records, document_class):
        """Make labeled document instances for the data in a file."""
        pass

class PlainTextFiles(Corpus):
    """A corpus contained in a collection of plain-text files."""

    def load(self, datafile, requested_records, document_class):
        """Make a document from a plain-text datafile. The document is labeled
        using the last component of the datafile's directory."""
        label = split(dirname(datafile))[-1]
        with open(datafile, "r") as file:
            data = file.read()
            self.documents.append(document_class(data, label, datafile))

class PlainTextLines(Corpus):
    """A corpus in which each document is a line in a datafile."""

    def load(self, datafile, requested_records, document_class):
        """Make a document from each line of a plain text datafile.
        The document is labeled using the datafile name, sans directory
        and extension."""
        label = splitext(basename(datafile))[0]
        with open(datafile, "r") as file:
            for line in file:
                data = line.strip()
                self.documents.append(document_class(data, label, datafile))


class NamesCorpus(PlainTextLines):
    """A collection of names, labeled by gender. See names/README for
    copyright and license."""

    def __init__(self, datafiles, requested_records, document_class=Document):
        super(NamesCorpus, self).__init__(datafiles, 0, document_class)

class ReviewCorpus(Corpus):
    """Yelp dataset challenge. A collection of business reviews."""

    def load(self, datafile, requested_records, document_class):
        """Make a document from each row of a json-formatted Yelp reviews"""
        """Only make total_records many documents to save memory"""
        with open(datafile, "r") as file:
            total_records = sum(1 for line in file)
            file.seek(0)
            for line in file:
                review = json.loads(line)
                label = review['sentiment']
                data = review['text']
                """Prevents overloading memory by loading in more records than we will use"""
                if requested_records > 0:
                    if randint(0,total_records) > requested_records*1.05:
                        continue        
                self.documents.append(document_class(data, label, datafile))

        print "\nsending %d records back, %d requested" % (len(self.documents), requested_records)


