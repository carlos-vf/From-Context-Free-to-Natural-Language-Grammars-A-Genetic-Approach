# -*- coding: utf-8 -*-
"""
Approximating Real Grammars from Context-free Languages using Grammar Evolution

@author: Carlos Velazquez Fernandez
@version: 2.0
"""


import random
import YAEP
from YAEP import earley
from YAEP import utils

import grammar_tools as gt




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
    Dictionary of sets with the rules. (Ex: {'S': {('VP'), (NP, N)})
                                             S -> VP | NP N
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
            
            child = []
            
            # Select length of child
            child_length = random.randint(1, max_length)
            
            for j in range(child_length):
                if child_length == 1 and symbol in symbols:
                    symbols.remove(symbol)
                child_comp = random.choice(symbols)
                child.append(child_comp)
                
            symbol_rules.add(tuple(child))
            
        rules[symbol] = symbol_rules
        
    return rules




def compute_fitnesses(population, lexicon, correct_examples, wrong_examples):
    
    fitnesses = []
    
    for individual in population:
        fitnesses.append(fitness(individual, lexicon, correct_examples, wrong_examples))
        
    return fitnesses

    
    
def fitness(individual, lexicon, correct_examples, wrong_examples):
    '''
    TP + TN
    '''
    # Compute the whole grammar of the CFL
    grammar = gt.get_full_grammar(individual, lexicon)
    
    # Format the grammar in the input form for the library
    formated_grammar = gt.format_grammar(grammar)
    
    tp = 0
    tn = 0

    # Try to parse every correct sentence
    for sentence in correct_examples:
        #try:
        sentence = ['the', 'sun', 'sets']
        formated_grammar = {'S':[['NP']], 'NP':[['DET', 'NOUN', 'VERB']], 'DET':[['the']], 'NOUN':[['sun']], 'VERB':[['sets']]}
        chart = earley.Earley().earley_parse(sentence, formated_grammar)
        parsed = [s for s in chart[-1] if s.left == 'S']
# =============================================================================
#         except Exception:
#             print(sentence)
#             print(grammar)
# =============================================================================
    
        if len(parsed) != 0:
            tp += 1
        
            
    
    # Try to parse every wrong sentence
    for sentence in wrong_examples:
        chart = earley.Earley().earley_parse(sentence, formated_grammar)
        parsed = [s for s in chart[-1] if s.left == 'S']
    
        if len(parsed) == 0:
            tn += 1
            
    #print("TRUE POSITIVES: ", tp)
    #print("TRUE NEGATIVES: ", tn, "\n")
            
    return tp+tn
    




###############################################################################
################################## DATASET ####################################
###############################################################################

print("\nLoading dataset...")

# List of well-constructed sentences
with open("dataset/eng/correct.txt", "r") as file:
    good_sentences = file.read()
good_sentences = good_sentences.split('\n')


# List of grammatically incorrect sentences with syntactical errors
with open("dataset/eng/wrong.txt", "r") as file:
    bad_sentences = file.read()
bad_sentences = bad_sentences.split('\n')

print("Done!\n")


###############################################################################
######################## DEFINITION OF THE GRAMMAR ############################
###############################################################################

print("Defining grammar...")

# Tokenization
good_tokenized = gt.tokenize(good_sentences)
bad_tokenized = gt.tokenize(bad_sentences)

# Pre-processing (removing punctuation marks and lower-casing all the words)
good_preprocessed = gt.preprocess(good_tokenized)
bad_preprocessed = gt.preprocess(bad_tokenized)

# Lexicon (dict of words with their possible (universal) POS tags)
sentences = good_preprocessed + bad_preprocessed
lexicon = gt.create_lexicon(sentences)

# Terminal symbols or vocabulary (set of words of the grammar)
terminal = gt.get_terminal(lexicon)

# Preterminal symbols (symbols whose rules only go to terminal symbols)
preterminal = gt.get_preterminal(lexicon)
            
# Non-terminal symbols
nonterminal = {'NP', 'VP'}

# Start symbol
start = 'S'

print("Done!\n")



###############################################################################
############################ GRAMMAR EVOLUTION ################################
###############################################################################

print("Starting evolutionary algorithm...\n")

# Initialization of the population
population = initialize_population(start, nonterminal, preterminal, n_individuals=1)

# Fitness of the individuals
fitnesses = compute_fitnesses(population, lexicon, good_preprocessed, bad_preprocessed)

for index,i in enumerate(population):
    pass
    #print(f'{index}: {i}')
    #print(f'{index} fitness: {fitnesses[i]}')



