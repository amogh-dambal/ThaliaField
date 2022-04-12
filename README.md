
<img src="img/thalia-v1.jpg" width=200px height=200px/>

# Thalia Fieldstone
## The Thunderfield AI Project
Thalia Fieldstone is a passionate songwriter trying her best to write poignant, 
yet passionate indie songs for Thunderfield, an interdisciplinary music
collective of independent artists based in Austin, TX. Her main songwriting acumen
is a Recurrent Neural Network (RNN) that uses a simple LSTM (long short term 
memory) architecture based on the word embedding model of computation built 
using `pytorch` in Python 3.7.2. 

Her critics have called her "derivative of her peers" and "nonsensical"; they
consider her more of a dedicated Thunderfield fan rather than an actual member. 

## Instructions
1. Download and unpack the source code into a folder labeled `thunderfield`
2. Modify the 4-word seed in the `main.py` source file to whatever you'd like.
    - You may modify Thalia's default configurations using the configuration settings found in `poetics.json`. 
3. Run Thalia using the command `python -l models/first-model.tfld main.py`

## Credits
Much of the initial code for Thalia v1.0.0 was sourced from 
[this tutorial](https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html)
on LSTM architectures in `pytorch`. Subsequent versions of Thalia
will aim to incorporate more and more of our own code and less tutorial code,
including as we experiment with different hyperparameters and architectures.
