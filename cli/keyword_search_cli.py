#!/usr/bin/env python3

import argparse
import json
import os
import string 
import pickle
from nltk.stem import PorterStemmer



stemmer = PorterStemmer()

os.makedirs("cache", exist_ok=True)




class InvertedIndex:

    def __init__(self):
        self.index = dict()
        self.docmap = dict()

    def __add_document(self, doc_id, text):
        tokens = preprocess(text)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)

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

    
    def save(self):
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

    def load(self):
        try:
            with open("cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)
            with open("cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)


        except FileNotFoundError:
            raise Exception("File Doens't exist")

        



    
def remove_stop(ls):
    with open('./data/stopwords.txt', 'r') as f:
        stop_words = f.read().split()

    for word in stop_words:
        if word in ls:
            ls.remove(word)
    return ls


def tokenize(text: str):
    return [stemmer.stem(token) for token in text.split( ) if token]

def preprocess(text):
    translator = str.maketrans('', '', string.punctuation)
    return remove_stop(tokenize(text.lower().translate(translator)))


def match(base, keyword): 
    for word in keyword:
        for sub in base:
            if word in sub:
                return True
    return False
    

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

    args = parser.parse_args()

    match args.command:
        case "search":
           search_movies(args.query)
        case "build":
           build_index() 

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
