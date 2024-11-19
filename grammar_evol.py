# -*- coding: utf-8 -*-
"""
Approximating Real Grammars from Context-free Languages using Grammar Evolution

@author: Carlos Velazquez Fernandez
@version: 2.0
"""

import nltk
from nltk.tokenize import word_tokenize
import random



def create_lexicon(sentences):
    '''
    Given a list/set of setences, returns the lexicon.
    The lexicon is a dictionary consisting of all the POS tags assigned
    to every word in the sentences.
    
    Example of lexicon: {'sun':{'NOUN'}, 'that':{'ADP', 'DET'}}
    '''
    lexicon = {}
    
    for s in sentences:
        sentence_lex = nltk.pos_tag(s, tagset='universal')
        for w in sentence_lex:
            if w[0] not in lexicon:
                lexicon[w[0]] = {w[1]}
            elif w[0] in lexicon and w[1] not in lexicon[w[0]]:
                updatedLex = lexicon[w[0]]
                updatedLex.add(w[1])
                lexicon[w[0]] = updatedLex    
                
    return lexicon



def initialize_population(start, nonterminal, preterminal, n_individuals):
    
    population = []
    
    for i in range(n_individuals):
        
        rules = generate_rules(start, nonterminal, preterminal, 5, 2)
        population.append(rules)

        
    return population





def generate_rules(start, nonterminal, preterminal, max_children, max_length):
    """
    Summary
    ----------
    Randomly generates a set of rules for a context free grammar.
    Special characteristics:
        - From the first symbol it is only possible to go to non-terminal symbols.
        - All leaf nodes are preterminal symbols.
        - Each node has a number max_children of children of size max_length.
        - A symbol cannot map only to itself (Ex: NP -> NP)
        

    Parameters
    ----------
    start: starting symbol (Ex: S).
    
    nonterminal: set of non-terminal symbols (Ex: {NP, VP}).
    
    preterminal: set of preterminal symbols (Ex: {NOUN, ADP, ADV}).
    
    max_children: maximum number of children per starting/non-terminal symbol.
    
    max_length: maximum length of each child (max number of symbols it can contain).
    

    Returns
    -------
    results: dictionary of sets with the rules. (Ex: {'S': {'NP', 'NP.N'})
    """
    
    rules = {}

    # Creation of rules for starting and nonterminal symbols
    for symbol in [start] + [s for s in nonterminal]:
        
        # Select number of children
        n_children = random.randint(1, max_children)
        
        # Add children
        if symbol == start:
            symbols = [s for s in nonterminal]
        else:
            symbols = [s for s in preterminal.union(nonterminal)]
        symbol_rules = set()
        
        for i in range(n_children):
            
            child = ""
        
            # Select length of child
            child_length = random.randint(1, max_length)
            
            for j in range(child_length):
                if child_length == 1 and symbol in symbols:
                    symbols.remove(symbol)
                child_comp = random.choice(symbols)
                child = child + "." + child_comp
                
            symbol_rules.add(child[1:])
            
        rules[symbol] = symbol_rules
        
    return rules






# List of well-constructed sentences
with open("dataset/eng/correct.txt", "r") as file:
    good_sentences = file.read()
good_sentences = good_sentences.split('\n')


# List of grammatically incorrect sentences with syntactical errors
with open("dataset/eng/wrong.txt", "r") as file:
    bad_sentences = file.read()
bad_sentences = bad_sentences.split('\n')



# Load nltk models
nltk.download('averaged_perceptron_tagger_eng')

# Tokenization
good_tokenized = [word_tokenize(s) for s in good_sentences]
bad_tokenized = [word_tokenize(s) for s in bad_sentences]

# Pre-processing (removing punctuation marks and lower-casing all the words)
good_preprocessed = [[w.lower() for w in s if w.isalpha()] for s in good_tokenized]
bad_preprocessed = [[w.lower() for w in s if w.isalpha()] for s in bad_tokenized]

# Lexicon (dict of words with their possible (universal) POS tags)
sentences = good_preprocessed + bad_preprocessed
lexicon = create_lexicon(sentences)

# Terminal symbols or vocabulary (set of words of the grammar)
terminal = set(w for w in lexicon.keys())

# Preterminal symbols (symbols whose rules only go to terminal symbols)
preterminal = set()
for l in lexicon.values():
        for e in l:
            preterminal.add(e)
            
# Non-terminal symbols
nonterminal = {'NP', 'VP'}

# Start symbol
start = 'S'



# GENETIC PROGRAMMING

# Initialization of the population
population = initialize_population(start, nonterminal, preterminal, n_individuals=2)
for index,i in enumerate(population):
    print(f'{index}: {i}')



