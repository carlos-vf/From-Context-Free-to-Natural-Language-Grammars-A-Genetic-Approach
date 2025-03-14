# -*- coding: utf-8 -*-
"""
Grammar tools library

@author: Carlos Velazquez Fernandez
@version: 1.0
"""

import nltk
from nltk.tokenize import word_tokenize
import spacy


def tokenize(sentence):
    """
    Summary
    ----------
    Tokenize a sentence. 
    Ex: "Hello world!".
        

    Parameters
    ----------
    sentence: string.
    

    Returns
    -------
    Tokenized sentence. 
    Ex: ["Hello", "world", "!"]
    """
    
    return word_tokenize(sentence)
    

    
def preprocess(sentence):
    """
    Summary
    ----------
    Preprocesses a sentence. 
    Ex: ["Hello", "world", "!"]
    Ex: "Hello world!"
    
    This preprocessing:
        - Removes punctuation marks
        - Lowercases all words
        

    Parameters
    ----------
    sentence: a tokenized sentence or a string.
    

    Returns
    -------
    Preprocessed sentence.
    Ex: ["hello", "world"]
    Ex: "hello world"
    """

    if isinstance(sentence, list):
        return [w.lower() for w in sentence if w.isalpha()]
    
    else:
        return ''.join(char for char in sentence.lower() if char.isalnum() or char.isspace())



def create_lexicon(sentences, lang):
    """
    Summary
    ----------
    Given a list/set of tokenized setences, returns the lexicon.
    The lexicon is a dictionary consisting of all the POS tags assigned
    to every word in the sentences (mapping between terminal and preterminal symbols).


    Parameters
    ----------
    sentences: list of tokenized sentences.

    lang: language code (eng/esp).
    

    Returns
    -------
    Dictionary with the lexicon of the grammar.
    Ex: {'NOUN':{'sun', 'moon'}, 'ADP':{'that'}, 'DET':{'that'}}
    """
    
    # Get lexicon
    lexicon = {}
                
    for s in sentences:

        sentence = preprocess(s)

        if lang == "eng":
            # Load nltk model (english)
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            sentence_lex = nltk.pos_tag(word_tokenize(sentence), tagset='universal')
            
        elif lang == "esp":
            # Load spacy model (spanish)
            nlp = spacy.load("es_core_news_sm", quiet=True)
            doc = nlp(sentence)
            sentence_lex = [[token.text, token.pos_] for token in doc]

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
    
    

    
    
    
    
    
    
    
    
    
    
    