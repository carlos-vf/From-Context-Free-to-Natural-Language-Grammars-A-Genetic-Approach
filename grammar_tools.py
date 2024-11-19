# -*- coding: utf-8 -*-
"""
Grammar tools library

@author: Carlos Velazquez Fernandez
@version: 1.0
"""

import nltk
from nltk.tokenize import word_tokenize
import copy


def tokenize(sentences):
    """
    Summary
    ----------
    Tokenize a list of sentences. 
    Ex: ["Hello world!", ["No way"]].
        

    Parameters
    ----------
    sentences: list of strings.
    

    Returns
    -------
    List of tokenized sentences. 
    Ex: [["Hello", "world", "!"], ["No", "way"]].
    """
    
    return [word_tokenize(s) for s in sentences]
    

    
def preprocess(sentences):
    """
    Summary
    ----------
    Preprocesses a list of sentences. 
    Ex: [["Hello", "world", "!"], ["No", "way"]]
    
    This preprocessing:
        - Removes punctuation marks
        - Lowercases all words
        

    Parameters
    ----------
    sentences: list of tokenized sentences.
    

    Returns
    -------
    List of preprocessed tokenized sentences. 
    Ex: [["hello", "world"], ["no", "way"]]
    """
    
    return [[w.lower() for w in s if w.isalpha()] for s in sentences]



def create_lexicon(sentences):
    """
    Summary
    ----------
    Given a list/set of tokenized setences, returns the lexicon.
    The lexicon is a dictionary consisting of all the POS tags assigned
    to every word in the sentences (mapping between terminal and preterminal symbols).


    Parameters
    ----------
    sentences: list of tokenized sentences.
    

    Returns
    -------
    Dictionary with the lexicon of the grammar.
    Ex: {'NOUN':{'sun', 'moon'}, 'ADP':{'that'}, 'DET':{'that'}}
    """
    
    # Load nltk models
    nltk.download('averaged_perceptron_tagger_eng')
    
    # Get lexicon
    lexicon = {}
                
    for s in sentences:
        sentence_lex = nltk.pos_tag(s, tagset='universal')
        for w in sentence_lex:
            if w[1] not in lexicon:
                lexicon[w[1]] = {w[0]}
            elif w[1] in lexicon and w[0] not in lexicon[w[1]]:
                updatedLex = lexicon[w[1]]
                updatedLex.add(w[0])
                lexicon[w[1]] = updatedLex    
                
    return lexicon



def get_terminal(lexicon):
    """
    Summary
    ----------
    Given a lexicon, retrieves the terminal symbols (vocabulary) of the grammar.
    Ex: {"the":{"DET"}, "that":{"DET", "ADP"}}


    Parameters
    ----------
    lexicon: lexicon of the grammar.
    

    Returns
    -------
    Set of terminal symbols.
    Ex: {"the", "that"}
    """
    
    terminal = set()
    for l in lexicon.values():
            for e in l:
                terminal.add(e)
                
    return terminal


def get_preterminal(lexicon):
    """
    Summary
    ----------
    Given a lexicon, retrieves the preterminal symbols of the grammar.
    Preterminal symbols are those who only map to terminal symbols.
    Ex: {"the":{"DET"}, "that":{"DET", "ADP"}}


    Parameters
    ----------
    lexicon: lexicon of the grammar.
    

    Returns
    -------
    Set of preterminal symbols.
    Ex: {"DET", "ADP"}
    """

    return set(w for w in lexicon.keys())



def get_full_grammar(rules, lexicon):
    """
    Summary
    ----------
    Merges the non-terminal rules and the lexicon mapping into one single
    dictionary uniforming their formats.


    Parameters
    ----------
    rules: dictionary of non-terminal rules of the grammar
    
    lexicon: dictionary containing the lexicon of the grammar
    

    Returns
    -------
    Dictionary with the complete grammar
    Ex: {'DET': {('the',), ('some',), 'NP': {('DET', 'ADV'), ('DET', 'CONJ')}
    """
    
    grammar = {}
    for (key, values) in lexicon.items():
        grammar[key] = set((w,) for w in values)
    grammar.update(rules)
    
    return grammar




def format_grammar(grammar):
    
    new_grammar = {}
    
    for (key, values) in grammar.items():
        new_values = []
        for v in values:
            components = []
            for c in v:
                components.append(c)
            new_values.append(components)
        new_grammar[key] = new_values
        
    return new_grammar
    
    

    
    
    
    
    
    
    
    
    
    
    