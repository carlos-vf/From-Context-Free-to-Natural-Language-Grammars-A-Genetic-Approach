# -*- coding: utf-8 -*-
"""
Approximating Real Grammars from Context-free Languages using Grammar Evolution

@author: Carlos Velazquez Fernandez
@version: 3.0
"""


import random
import grammar_tools as gt
import statistics
import earleyparser as ep
from collections import Counter
import numpy as np
import neptune
import time


def initialize_population(start, nonterminal, preterminal, lexicon, n_individuals, max_rules, max_symbols):
    
    population = []
    
    for i in range(n_individuals):
        
        rules = generate_rules(start, nonterminal, preterminal, max_rules, max_symbols)
        
        # Compute the whole grammar of the CFL
        grammar = gt.get_full_grammar(rules, lexicon)
        formated_grammar = gt.format_grammar(grammar)
        
        population.append(formated_grammar)

        
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




def compute_fitnesses(population, correct_examples, wrong_examples):
    
    fitnesses = []
    
    for individual in population:
        f = fitness(individual, correct_examples, wrong_examples)
        fitnesses.append(f)

        
    return fitnesses

    
    
def fitness(individual, correct_examples, wrong_examples):
    """
    Summary
    ----------
    Computes the fitness for an individual. The individual (grammar) operates over a 
    dataset of examples (sentences). The fitness funtions is the sum of well-formed sentences
    that can be parsed given the grammar of the individual and the bad-formed sentences
    that cannot be parsed given the same grammar.
    
    fitness = true positives + true negarives


    Parameters
    ----------
    individual: the grammar
    
    correct_examples: set of (tokenized) well-formed sentences
    
    wrong_examples: set of (tokenized) bad-formed sentences
    

    Returns
    -------
    fitness score (float).
    """
    
    tp = 0
    tn = 0

    # Try to parse every correct sentence
    for sentence in correct_examples:
        parsed = ep.parse(sentence, individual)

        if parsed != None:
            tp += 1
        
        
    # Try to parse every wrong sentence
    for sentence in wrong_examples:
        parsed = ep.parse(sentence, individual)

        if parsed == None:
            tn += 1
      
            
    return tp+tn
    


def selection(population, fitness, t_size):
    
    n = len(population)
    
    # Select tournament participants
    participants = np.random.choice(n, t_size)
    
    # Tournament
    winners = []
    for i in range(int(t_size/2)):
        
        indiv_a_index = participants[i]
        indiv_b_index = participants[t_size-i-1]
        
        indiv_a_fintess = fitness[indiv_a_index]
        indiv_b_fintess = fitness[indiv_b_index]
        
        indiv_a = population[indiv_a_index]
        indiv_b = population[indiv_b_index]
        
        if indiv_a_fintess >= indiv_b_fintess:
            winners.append(indiv_a)
        
        else:
            winners.append(indiv_b)

                 
    return winners
        
           
    
def compute_crossovers(population, n_children, symbols, max_rules):
    
    new_population = []
    
    for i in range(0, len(population), 2):
        for j in range(n_children):
            
            new_individual = crossover(population[i], population[i+1], symbols, max_rules)
            new_population.append(new_individual)
        
    return new_population
        
        
    
def crossover(parent_a, parent_b, symbols, max_rules):
    
    new_individual = {}
    
    for key in parent_a.keys():
        
        # If it is a non-terminal (or starting) symbol
        if key in symbols:
            
            a_rule = parent_a[key]
            b_rule = parent_b[key]
            
            possible_rules = a_rule + b_rule
            possible_rules =[list(tup) for tup in set(tuple(sublist) for sublist in possible_rules)]
            #n_children = min(np.random.randint(len(possible_rules)) + 1, max_rules)
            n_children = np.random.randint(len(possible_rules)) + 1
            
            new_rules = random.sample(possible_rules, n_children)
            new_individual[key] = new_rules
        
        # Otherwise, the rules are copied (preterminal rules cannot be mutated)
        else:
            new_individual[key] = parent_a[key]
            
    return new_individual
    
    

def compute_mutations(population, probability, start, nonterminal, preterminal):
    
    new_population = []
    
    for i in population:
        new_indiv = mutation(i, probability, start, nonterminal, preterminal)
        new_population.append(new_indiv)
        
    return new_population



def mutation(individual, probability, start, nonterminal, preterminal):
    
    new_indiv = {}
    
    for key, rules in individual.items():
        
        # For the starting symbol
        if key == start:
            new_rules = []
            
            for rule in rules:
                new_rule = []
                
                for symbol in rule:
                    
                    mutation = np.random.choice(2, 1, p=[1-probability, probability])[0]
                    
                    if mutation:
                        new_set = sorted(nonterminal)
                        new_set.remove(symbol)
                        new_symbol = random.sample(new_set, 1)[0]
                        
                    else:
                        new_symbol = symbol
                        
                    new_rule.append(new_symbol)
                new_rules.append(new_rule)

            new_indiv[key] = new_rules
                    
                        
        # For non terminals
        elif key in nonterminal:
            new_rules = []
            
            for rule in rules:
                new_rule = []
                
                for symbol in rule:
                    
                    mutation = np.random.choice(2, 1, p=[1-probability, probability])[0]
                    
                    if mutation:
                        new_set = sorted(nonterminal.union(preterminal))
                        new_set.remove(symbol)
                        new_symbol = random.sample(new_set, 1)[0]
                        
                    else:
                        new_symbol = symbol
                        
                    new_rule.append(new_symbol)
                new_rules.append(new_rule)

            new_indiv[key] = new_rules
        
        # Otherwise
        else:
            new_indiv[key] = rules
            
    return new_indiv
    
    
    

    
###############################################################################
################################## NEPTUNE ####################################
###############################################################################

neptune_sync = True

if neptune_sync:
    run = neptune.init_run(
        project="carlos-vf/Evolutive-Context-Free-Grammars",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOGI4MjZjOC05MGJmLTQyNmEtOWFjYS04MDU4MTRhMjdiMDMifQ==",
    )  # your credentials



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

# Paramters
n_individuals = 120
max_rules = 10
max_symbols = 3
p_mutation = 0.1
max_iter = 100

if neptune_sync:
    params = {  "n_individuals": n_individuals,
                "max_rules" : max_rules,
                "max_symbols" : max_symbols,
                "p_mutation" : p_mutation}
    run["parameters"] = params
    run["sys/tags"].add([f"{i}={j}" for (i,j) in params.items()])


print("Starting evolutionary algorithm...\n")


# Initialization of the population
print("Initialazing population...\n")
population = initialize_population(start, nonterminal, preterminal, lexicon, n_individuals, max_rules, max_symbols)
results = []
best_individual = [0,0,0] # [individual, fitness, iteration]
i = 0

while(best_individual[1] < len(sentences) and i < max_iter):

    t1 = time.time()

    # Fitness of the individuals
    #print("Computing fitness...\n")
    fitnesses = compute_fitnesses(population, good_preprocessed, bad_preprocessed)
    best_fitness = max(fitnesses)
    if best_fitness > best_individual[1]:
        index_max = max(range(len(fitnesses)), key=fitnesses.__getitem__)
        best_individual = [population[index_max], best_fitness, i]

    
    # Tournament selection
    #print("Creating tournament...\n")
    t_size = int(n_individuals)
    selected_individuals = selection(population, fitnesses, t_size)
    
    
    # Crossover
    #print("Performing crossover...\n")
    children_per_cross = 4
    new_children = compute_crossovers(selected_individuals, children_per_cross, set(start).union(set(nonterminal)), max_rules)
    

    # Mutation
    #print("Applying mutation...\n")
    population = compute_mutations(new_children, p_mutation, start, nonterminal, preterminal)


    # Report
    avg_fitness = statistics.mean(fitnesses)
    results.append(avg_fitness)
    best_indiv_size = len([item for sublist in best_individual[0].keys() for item in sublist])
    t = time.time() - t1

    print(f'Iteration {i}')
    print(f'Average fitness = {avg_fitness}')
    print(f'Fitness of each individual = {Counter(fitnesses)}')
    print(f"Results until now: {results}")
    print(f'Best individual: {best_individual}')
    
    if neptune_sync:
        run["eval/avg_fitness"].append(avg_fitness)
        run["eval/best_fitness"].append(best_fitness)
        run["eval/best_iteration"].append(best_individual[2])
        run["eval/best_indiv_size"].append(best_indiv_size)
        run["eval/iter_time"].append(t)

        # TODO
        # time of iteration
        # Parallelize?

    i += 1

if neptune_sync:
    run.stop()  




