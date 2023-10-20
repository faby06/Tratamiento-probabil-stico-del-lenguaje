#Gram�ticas Probabil�sticas Lexicalizadas

# Definici�n manual de una L-PCFG simple
lpcfg_grammar = {
    "S": [("NP VP", 1.0)],
    "NP": [
        ("Det N", 0.6),
        ("Pronoun", 0.4)
    ],
    "VP": [
        ("V NP", 0.7),
        ("VP PP", 0.3)
    ],
    "PP": [("P NP", 1.0)],
    "Det": {
        "the": 0.8,
        "a": 0.2
    },
    "N": {
        "cat": 0.4,
        "dog": 0.4,
        "bat": 0.2
    },
    "V": {
        "chased": 0.6,
        "saw": 0.4
    },
    "Pronoun": {
        "I": 0.4,
        "you": 0.3,
        "it": 0.2,
        "cats": 0.1
    },
    "P": {
        "with": 0.6,
        "in": 0.4
    }
}

# Oraci�n de ejemplo
sentence = "I saw the cat with a dog"

# Tokenizaci�n de la oraci�n
tokens = sentence.split()

# Funci�n para calcular la probabilidad de una oraci�n
def calculate_sentence_probability(grammar, tokens):
    probability = 1.0
    for token in tokens:
        if token in grammar and isinstance(grammar[token], dict):
            continue  # Es una palabra con probabilidades en lugar de una categor�a
        if token in grammar:
            probability *= grammar[token]
        else:
            return 0.0  # Si una palabra no est� en la gram�tica, probabilidad cero
    return probability

# Calcular y mostrar la probabilidad de la oraci�n
probability = calculate_sentence_probability(lpcfg_grammar, tokens)
print("Probabilidad de la oracion:", probability)

