import argparse
from distutils.command.config import config
import sys
import random
import json

from dataset import Dataset
from model import Model
from train import *

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", type=str, default="")
    parser.add_argument("-s", "--save", type=str, default="")
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=4)
    args = parser.parse_args()
    return args

def writeLyrics(lyricPredictions: list, configOptions: dict, wpl: int):
    stanzas = configOptions['numStanzas']
    linesPerStanza = configOptions['numLinesPerStanza']
    wordsPerLine = wpl

    i = 0
    for _ in range(stanzas):
        for _ in range(linesPerStanza):
            for _ in range(wordsPerLine):
                sys.stdout.write(lyrics[i] + " ")
                i += 1
            sys.stdout.write("\n")
        sys.stdout.write("\n")
    
    
if __name__ == "__main__":
    args = getArgs()

    with open("poetics.json", "r") as cf:
        opts = json.load(cf)

    stanzas = opts["numStanzas"]
    linesPerStanza = opts["numLinesPerStanza"]
    wordsPerLine = random.randint(
        opts["numWordsPerLine"][0], opts["numWordsPerLine"][1]
    )
    nextWords = stanzas * linesPerStanza * wordsPerLine

    dataset = Dataset(args)

    if len(args.load) != 0:
        model = torch.load(args.load)
    else:
        model = Model(dataset)
        train(dataset, model, args)

    lyrics = predict(dataset, model, text="stare into the mirror", next_words=nextWords)

    writeLyrics(lyrics, opts, wordsPerLine)
