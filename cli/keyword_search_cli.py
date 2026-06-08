#!/usr/bin/env python3

import argparse
import json
import math
import os
import string 
import pickle

from typing import Dict
from collections import Counter
from nltk.stem import PorterStemmer


CACHE_DIR = "cache"
BM25_K1 = 1.5
BM25_B = 0.75

os.makedirs(CACHE_DIR, exist_ok=True)
stemmer = PorterStemmer()


class InvertedIndex:

    def __init__(self):
        self.index = dict()
        self.docmap = dict()
        self.term_frequencies: Dict[int, Counter] = dict()
        self.doc_lengths = dict()
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pk1")

    def __add_document(self, doc_id, text):
        tokens = preprocess(text)
        self.term_frequencies.setdefault(doc_id, Counter())
        self.doc_lengths[doc_id] = len(set(tokens))
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)
            self.term_frequencies[doc_id][token] += 1
        
    def __get_avg_doc_length(self) -> float:
        total_length = sum(self.doc_lengths.values())
        num_docs = len(self.doc_lengths)
        average_length = total_length / num_docs if num_docs > 0 else 0.0
        return average_length


    def get_documents(self, term: str):
        if term.lower() in self.index:
            return sorted(self.index[term.lower()])
        return []

    def build(self) -> None:
        with open('./data/movies.json', 'r') as f:
            data = json.load(f)
            movies = data["movies"]
        for m in movies:
            self.docmap[m['id']] = m
            self.__add_document(m['id'], f"{m['title']} {m['description']}")
    
    def get_tf(self, doc_id, term):
        tokens = preprocess(term)
        if len(tokens) > 1:
            raise Exception("Error: more than one token for term frequency search")

        return self.term_frequencies[doc_id][tokens[0]]

    def get_bm25_idf(self, term: str) -> float:
        tokens = preprocess(term)
        if len(tokens) > 1 or len(tokens) < 1:
            raise Exception("Error: more than one token for idf")
        n = len(self.docmap.keys())
        df = len(self.get_documents(tokens[0]))
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)


    def save(self):
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        try:
            with open("cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)
            with open("cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)
            with open("cache/term_frequencies.pkl", "rb") as f:
                self.term_frequencies = pickle.load(f)
            with open(self.doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)

        except FileNotFoundError:
            raise Exception("File Doens't exist")


    
def remove_stop(ls):
    with open('./data/stopwords.txt', 'r') as f:
        stop_words = f.read().split()
    return [w for w in ls if w not in set(stop_words)]


def tokenize(text: str):
    return [stemmer.stem(token) for token in text.split( ) if token]

def preprocess(text):
    translator = str.maketrans('', '', string.punctuation)
    return remove_stop(tokenize(text.lower().translate(translator)))


def bm25_idf_command(term: str):
    index = InvertedIndex()
    index.load()
    return index.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1):
    index = InvertedIndex()
    index.load()
    return index.get_bm25_tf(doc_id, term, k1)


    

def search_movies(query: str):
    print(f"Searching for: {query}")
    index = InvertedIndex()
    index.load()
    result = []

    for token in preprocess(query):
        if len(result) >= 5:
            break
        for t in index.get_documents(token):
            if len(result) >= 5:
                break
            if t in result:
                continue
            result.append(t)
            

    for v in result:
       print(index.docmap[v]['id'], index.docmap[v]['title'])

def get_term_freq(doc_id: int, term: str):
    index = InvertedIndex()
    index.load()
    return index.get_tf(doc_id, term)

def get_idf(term: str):
    index = InvertedIndex()
    index.load()
    tokens = preprocess(term)
    if len(tokens) > 1:
        raise Exception("Error: more than one token for term frequency search")
    total_doc_count = len(index.docmap.keys())
    term_match_doc_count = len(index.get_documents(tokens[0]))
    return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

def get_tf_idf(doc_id, term: str):
    idf = get_idf(term)
    tf = get_term_freq(doc_id, term) 
    tf_idf = idf * tf
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")
    

def build_index():
    index = InvertedIndex()
    index.build()
    index.save()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    subparsers.add_parser("build")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a specified 'term'")
    tf_parser.add_argument("doc_id", type=int, help="Document Id")
    tf_parser.add_argument("term", type=str, help="term")

    idf_parser = subparsers.add_parser("idf", help="Get Inverse Document Freq")
    idf_parser.add_argument("term", type=str, help="Term for the idf")

    idf_tf_parser = subparsers.add_parser("tfidf", help="Get term frequency for a specified 'term'")
    idf_tf_parser.add_argument("doc_id", type=int, help="Document Id")
    idf_tf_parser.add_argument("term", type=str, help="term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
      "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")


    args = parser.parse_args()

    match args.command:
        case "search":
           search_movies(args.query)
        case "build":
           build_index() 
        case "tf":
            freq = get_term_freq(args.doc_id, args.term)
            print(f"Term appeared {freq} times")
        case "tfidf":
            get_tf_idf(args.doc_id, args.term)
        case "idf":
            idf = get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
