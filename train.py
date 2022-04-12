import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader


def train(dataset, model, args):
    model.train()

    loader = DataLoader(dataset, batch_size=args.batch_size)
    loss = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.max_epochs):
        h, c = model.init_state(args.sequence_length)

        for batch, (x, y) in enumerate(loader):
            opt.zero_grad()

            y_pred, (h, c) = model(x, (h, c))
            loss = nn.CrossEntropyLoss(y_pred.transpose(1, 2), y)

            h = h.detach()
            c = c.detach()

            loss.backward()
            opt.step()

            # TODO: use logging framework to log this
            # TODO: add a verbose mode toggle
            print({"epoch": epoch, "batch": batch, "loss": loss.item()})

    torch.save(model, "models/first-model.tfld")


def predict(dataset, model, text, nextWords=240):
    model.eval()

    seed = text.split(" ")
    words = seed
    h, c = model.init_state(len(seed))

    for i in range(0, nextWords):
        x = torch.tensor([[dataset.wordToIdx[w] for w in seed[i:]]])
        yPred, (h, c) = model(x, (h, c))

        logits = yPred[0][-1]
        p = torch.nn.functional.softmax(logits, dim=0).detach().numpy()
        wordIdx = np.random.choice(len(logits), p=p)
        words.append(dataset.idxToWord[wordIdx])

    return words
