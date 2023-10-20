#Gram�ticas Probab. Independ. del Contexto

import nltk
from nltk.grammar import PCFG
from nltk import InsideChartParser

# Definici�n de la gram�tica probabil�stica
pcfg_grammar = PCFG.fromstring("""
    S -> NP VP [1.0]
    NP -> Det N [0.6] | Pronoun [0.4]
    Det -> 'the' [0.8] | 'a' [0.2]
    N -> 'cat' [0.4] | 'dog' [0.4] | 'bat' [0.2]
    VP -> V NP [0.7] | VP PP [0.3]
    V -> 'chased' [0.6] | 'saw' [0.4]
    Pronoun -> 'I' [0.4] | 'you' [0.3] | 'it' [0.2] | 'cats' [0.1]
    PP -> P NP [1.0]
    P -> 'with' [0.6] | 'in' [0.4]
""")

# Oraci�n de ejemplo
sentence = "I saw the cat with a dog"

# Tokenizaci�n de la oraci�n
tokens = nltk.word_tokenize(sentence)

# Creaci�n de un analizador sint�ctico PCFG
parser = InsideChartParser(pcfg_grammar)

# An�lisis sint�ctico de la oraci�n
for tree in parser.parse(tokens):
    tree.pretty_print()

# Probabilidad de la oraci�n
print("Probabilidad de la oraci�n:", pcfg_grammar.probabilistic_parse(tokens).logprob())

