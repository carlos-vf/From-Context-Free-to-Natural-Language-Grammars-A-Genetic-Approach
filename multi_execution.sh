#!/bin/bash

# Parameters
n_individuals=80
max_iter=31
init_rules=10
init_symbols=3
p_mutation=0.1
bloat=0.0025
elite=0.1
n_nonterminal=2


for lang in "eng"; do

    # Vary nonterminals
    for n in 1 2 3 4 5 6; do
        python grammar_evol.py --lang "$lang" --n_individuals "$n_individuals" --max_iter "$max_iter" --init_rules "$init_rules" --init_symbols "$init_symbols" --p_mutation "$p_mutation" --bloat "$bloat" --elite "$elite" --n_nonterminal "$n"
    done

    # Vary elite
    for e in 0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20; do
        python grammar_evol.py --lang "$lang" --n_individuals "$n_individuals" --max_iter "$max_iter" --init_rules "$init_rules" --init_symbols "$init_symbols" --p_mutation "$p_mutation" --bloat "0" --elite "$e" --n_nonterminal "$n_nonterminal"
    done

    # Vary bloat
    for b in 0.000 0.002 0.004 0.006 0.008 0.010 0.012 0.014 0.016 0.018 0.020; do
        python grammar_evol.py --lang "$lang" --n_individuals "$n_individuals" --max_iter "$max_iter" --init_rules "$init_rules" --init_symbols "$init_symbols" --p_mutation "$p_mutation" --bloat "$b" --elite "$elite" --n_nonterminal "$n_nonterminal"
    done

done
