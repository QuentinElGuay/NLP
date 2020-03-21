import numpy as np
import matplotlib.pyplot as plt

import string
import random
import re
import requests
import os


def create_mapping(letters2):
    letters1 = list(string.ascii_lowercase)

    mapping = {}
    for k, v in zip(letters1, letters2):
        mapping[k] = v

    return mapping

letters2 = list(string.ascii_lowercase)
random.shuffle(letters2)
true_mapping = create_mapping(letters2)

### the language model

# initialize Markov matrix
M = np.ones((26, 26))

# initial state distribution
pi = np.zeros(26)


def update_transition(char1, char2):
    # use function ord to convert ascii to integer: ord('a') = 97, ord('b') = 98, etc.
    i = ord(char1) - 97
    j = ord(char2) - 97
    M[i][j] += 1


def update_pi(char):
    # use function ord to convert ascii to integer: ord('a') = 97, ord('b') = 98, etc.
    pi[ord(char) - 97] += 1

def get_word_prob(word):
    i = ord(word[0]) - 97
    logp = np.log(pi[i])

    for char in word[1:]:
        j = ord(char) - 97
        logp += np.log(M[i][j])
        i = j

    return logp

def get_sequence_prob(words):
    if type(words) == str:
        words = words.split()

    logp = 0
    for word in words:
        logp += get_word_prob(word)

    return logp

### create a markov model based on an English dataset
# is an edit of https://www.gutenberg.org/ebooks/2701
# (I removed the front and back matter)

# download the file
if not os.path.exists('moby_dick.txt'):
  print("Downloading moby dick...")
  r = requests.get('https://lazyprogrammer.me/course_files/moby_dick.txt')
  with open('moby_dick.txt', 'w') as f:
    f.write(r.content.decode())

regex = re.compile('[^a-zA-Z]')

for line in open('moby_dick.txt'):
    line.rstrip()

    if line:
        line = regex.sub(' ', line) # replace non alpha characters with space
        
        # split line into tokens
        tokens = line.lower().split()

        for token in tokens:
            char0 = token[0]
            update_pi(char0)
        
            for char1 in token[1:]:
                update_transition(char0, char1)
                char0 = char1

pi /= pi.sum()
M /= M.sum(axis=1, keepdims=True)


### encode a message

# this is a random excerpt from Project Gutenberg's
# The Adventures of Sherlock Holmes, by Arthur Conan Doyle
# https://www.gutenberg.org/ebooks/1661

original_message = '''I then lounged down the street and found,
as I expected, that there was a mews in a lane which runs down
by one wall of the garden. I lent the ostlers a hand in rubbing
down their horses, and received in exchange twopence, a glass of
half-and-half, two fills of shag tobacco, and as much information
as I could desire about Miss Adler, to say nothing of half a dozen
other people in the neighbourhood in whom I was not in the least
interested, but whose biographies I was compelled to listen to.
'''

def decode_message(text, cypher_map):
    encoded_chars = []
    for char in text:
        encoded_char = char
        if encoded_char in cypher_map:
            encoded_char = cypher_map[char]
        encoded_chars.append(encoded_char)

    return ''.join(encoded_chars)

# encode text
def encode_message(text, cypher_map):
    # replace non-alpha characters
    text = regex.sub(' ', text)
    text = text.lower().rstrip()

    return decode_message(text, cypher_map)

encoded_message = encode_message(original_message, true_mapping)
print(encoded_message)

### run an evolutionary algorithm to decode the message
def next_generation(dna_pool, nb_children):
    """
    Create a new generation of offsprings from the dna pool.
    Parameters:
        dna_pool (list(str)): surving DNAs of the current generation
        nb_children (int): number of offsprings per survivor
    Returns:
        list(str): new dna_pool composed of the survivors and the offsprings
    """
    offsprings = []

    for dna in dna_pool:
        for _ in range(nb_children):
            copy = dna.copy()
            i = random.randint(0, len(copy) -1)
            j = random.randint(0, len(copy) -1)

            tmp = copy[i]
            copy[i] = copy[j]
            copy[j] = tmp

            offsprings.append(copy)
    
    return offsprings + dna_pool

### Init randomly the DNA pool
dna_pool = []
for _ in range(20):
    dna = list(string.ascii_lowercase)
    random.shuffle(dna)
    dna_pool.append(dna)


NUMBER_ITERATIONS = 1000

scores = np.zeros(NUMBER_ITERATIONS)
best_score = float('-inf')
best_dna = None
best_map = None

for i in range(NUMBER_ITERATIONS):
    if i > 0:
        dna_pool = next_generation(dna_pool, 3)

    dna2score = {}
    for dna in dna_pool:
        current_map = create_mapping(dna)
        score = get_sequence_prob(decode_message(encoded_message, current_map))
        dna2score[''.join(dna)] = score

        # record the best so far
        if score > best_score:
            best_dna = dna
            best_map = current_map
            best_score = score

    scores[i] = np.mean(list(dna2score.values()))

    if i % 200 == 0:
        print("iter:", i, "score:", scores[i], "best so far:", best_score)


    # keep the best 5 dna
    # also turn them back into list of single chars
    sorted_dna = sorted(dna2score.items(), key=lambda x: x[1], reverse=True)
    dna_pool = [list(k) for k, v in sorted_dna[:5]]

print('Best score {}'.format(best_score))
print(original_message)
print(decode_message(encoded_message, best_map))