# From Context-Free to Natural Language Grammars: A Genetic Approach

## Summary

Grammar is a fundamental aspect of natural language, defining the structure and rules that govern linguistic expressions. Understanding and modeling real-world grammars is crucial for applications in natural language processing, artificial intelligence, and computational linguistics. 

This project explores the use of Genetic Programming to evolve a population of context-free grammars (CFGs), iteratively refining them to approximate the structure of real grammars (such as English or Spanish). By applying evolutionary principles such as selection, crossover, and mutation, the system progressively improves the grammars' ability to generate sentences that resemble real structures.


## Execution

The main file, `grammar_evol.py`, consists on the genetic algorithm itself. It can executed from the command line with the following parameters:
- `lang`: Language of the dataset (only `eng` or `esp`). Default: `eng`.
- `n_individuals`: Number of individuals per generation. Default: `80`.
- `max_iter`: Number of generations. Default: `5`.
- `init_ruleses`: Maximum number of rules per grammar. Default: `10`.
- `init_symbols`: Maximim number of symbols which can be mapped per rule. Default: `3`. 
- `p_mutation`: Probability of mutation. Default: `0.1`.
- `bloat`: Bloat penalty. Default: `0.0025`.
- `elite`: Elite factor. Default: `0.1`.
- `n_nonterminal`: Number of nonterminal symbols. Default: `4`. 


To start with default params:

```
python grammar_evol.py
```