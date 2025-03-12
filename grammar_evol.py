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
import nltk

from multiprocessing import Pool
from functools import partial


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
    
    partial_function = partial(fitness, correct_examples=correct_examples, wrong_examples=wrong_examples)
    
    with Pool() as pool:
        fitnesses = pool.map(partial_function, population)
        
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

    parents = [[population[i],population[i+1]] for i in range(0, len(population), 2)]
    partial_function = partial(crossover, symbols=symbols, max_rules=max_rules, n_children=n_children)
    
    with Pool() as pool:
        new_population = pool.map(partial_function, parents)

    new_population = [indiv for sublist in new_population for indiv in sublist]
        
    return new_population
        
        
    
def crossover(parents, symbols, max_rules, n_children):
    
    parent_a = parents[0]
    parent_b = parents[1]
    children = []

    for i in range(n_children):

        new_individual = {}
        
        for key in parent_a.keys():
            
            # If it is a non-terminal (or starting) symbol
            if key in symbols:
                
                a_rule = parent_a[key]
                b_rule = parent_b[key]
                
                possible_rules = a_rule + b_rule
                possible_rules =[list(tup) for tup in set(tuple(sublist) for sublist in possible_rules)]
                #n_rules = min(np.random.randint(len(possible_rules)) + 1, max_rules)
                n_rules = np.random.randint(len(possible_rules)) + 1
                
                new_rules = random.sample(possible_rules, n_rules)
                new_individual[key] = new_rules
            
            # Otherwise, the rules are copied (preterminal rules cannot be mutated)
            else:
                new_individual[key] = parent_a[key]

        children.append(new_individual)
            
    return children
    
    

def compute_mutations(population, probability, start, nonterminal, preterminal):

    partial_function = partial(mutation, probability=probability, start=start, nonterminal=nonterminal, preterminal=preterminal)
    
    with Pool() as pool:
        new_population = pool.map(partial_function, population)
        
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
                        mut_type = random.sample({"add", "del", "mod"}, 1)[0]

                        # Symbol modification
                        if mut_type == "mod":
                            new_set = sorted(nonterminal)
                            new_set.remove(symbol)
                            new_symbols = [random.sample(new_set, 1)[0]]

                        # Symbol addition
                        elif mut_type == "add":
                            new_symbols = [symbol, random.sample(sorted(nonterminal), 1)[0]]

                        # Symbol deletion
                        else:
                            new_symbols = []
                        
                    else:
                        new_symbols = [symbol]
                    
                    for s in new_symbols:
                        new_rule.append(s)

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
                        mut_type = random.sample({"add", "del", "mod"}, 1)[0]

                        # Symbol modification
                        if mut_type == "mod":
                            new_set = sorted(nonterminal.union(preterminal))
                            new_set.remove(symbol)
                            new_symbols = [random.sample(new_set, 1)[0]]

                        # Symbol addition
                        elif mut_type == "add":
                            new_symbols = [symbol, random.sample(sorted(nonterminal.union(preterminal)), 1)[0]]

                        # Symbol deletion
                        else:
                            new_symbols = []

                    else:
                        new_symbols = [symbol]
                    
                    for s in new_symbols:
                        new_rule.append(s)

                new_rules.append(new_rule)

            new_indiv[key] = new_rules
        
        # Otherwise
        else:
            new_indiv[key] = rules
            
    return new_indiv
    
    
    

   
###############################################################################
################################## NEPTUNE ####################################
###############################################################################
if __name__ == '__main__':

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
    lang = "eng"

    # List of well-constructed sentences
    with open(f"dataset/{lang}/correct.txt", "r", encoding="utf-8") as file:
        good_sentences = file.read()
    good_sentences = good_sentences.split('\n')


    # List of grammatically incorrect sentences with syntactical errors
    with open(f"dataset/{lang}/wrong.txt", "r", encoding="utf-8") as file:
        bad_sentences = file.read()
    bad_sentences = bad_sentences.split('\n')

    print("Done!\n")


    ###############################################################################
    ######################## DEFINITION OF THE GRAMMAR ############################
    ###############################################################################

    print("Defining grammar...")

    # Tokenization
    nltk.download('punkt_tab')
    nltk.download('universal_tagset')
    good_tokenized = [gt.tokenize(s) for s in good_sentences]
    bad_tokenized = [gt.tokenize(s) for s in bad_sentences]

    # Pre-processing (removing punctuation marks and lower-casing all the words)
    good_preprocessed = [gt.preprocess(s) for s in good_tokenized]
    bad_preprocessed = [gt.preprocess(s) for s in bad_tokenized]

    # Lexicon (dict of words with their possible (universal) POS tags)
    sentences = good_sentences + bad_sentences
    lexicon = gt.create_lexicon(sentences, lang)
    print(lexicon)

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
    max_iter = 25

    if neptune_sync:
        params = {  "n_individuals": n_individuals,
                    "max_rules" : max_rules,
                    "max_symbols" : max_symbols,
                    "p_mutation" : p_mutation,
                    "dataset" : lang}
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
        best_indiv_size = len([item for sublist in best_individual[0].values() for item in sublist])
        avg_indiv_size = []
        for indiv in population:
            avg_indiv_size.append(len([item for sublist in indiv.values() for item in sublist]))
        avg_indiv_size = statistics.mean(avg_indiv_size)
        t = time.time() - t1

        print(f'Iteration {i}')
        print(f'Average fitness = {avg_fitness}')
        print(f'Fitness of each individual = {Counter(fitnesses)}')
        print(f"Results until now: {results}")
        print(f'Best individual: {best_individual}')
        
        if neptune_sync:

            # Best individual up to now
            run["eval/best_individual/fitness"].append(best_fitness)
            run["eval/best_individual/size"].append(best_indiv_size)

            # Current population
            run["eval/population/avg_fitness"].append(avg_fitness)
            run["eval/population/avg_size"].append(avg_indiv_size)

            # Iteration
            run["eval/iter/best"].append(best_individual[2])
            run["eval/iter/time"].append(t)

        i += 1

    if neptune_sync:
        run.stop()  




