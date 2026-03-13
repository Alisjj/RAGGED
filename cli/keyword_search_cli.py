#!/usr/bin/env python3

import argparse
import json
import string 
from nltk.stem import PorterStemmer



stemmer = PorterStemmer()

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
    result = []
    with open('./data/movies.json', 'r') as f:
        data = json.load(f)
        movies = data["movies"]
        for m in movies:
            if match(preprocess(m["title"]), preprocess(query)):
                result.append(m["title"])

        for v in result:
            print(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
           search_movies(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
