import argparse
import sys
import random
import json

from dataset import Dataset
from model import Model
from train import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load', type=str, default='')
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--sequence-length', type=int, default=4)
    args = parser.parse_args()

    with open('poetics.json', 'r') as cf:
        opts = json.load(cf)
    
    stanzas = opts['numStanzas']
    linesPerStanza = opts['numLinesPerStanza']
    wordsPerLine = random.randint(opts['numWordsPerLine'][0], opts['numWordsPerLine'][1])

    nextWords = stanzas * linesPerStanza * wordsPerLine

    dataset = Dataset(args)
    
    if len(args.load) != 0:
        model = torch.load(args.load)
    else:
        model = Model(dataset)
        train(dataset, model, args)

    lyrics = predict(dataset, model, text='stare into the mirror', next_words=nextWords)

    i = 0
    for stanza in range(stanzas):
        for line in range(linesPerStanza):
            for word in range(wordsPerLine):
                sys.stdout.write(lyrics[i] + " ")
                i += 1
            sys.stdout.write('\n')
        sys.stdout.write('\n')