#Extracción de Información

import nltk
from nltk import ne_chunk
from nltk.tokenize import word_tokenize, sent_tokenize

# Oración de ejemplo
text = "Barack Obama was born in Honolulu, Hawaii. He served as the 44th President of the United States."

# Tokenización de oraciones y palabras
sentences = sent_tokenize(text)
words = [word_tokenize(sentence) for sentence in sentences]

# Etiquetado de entidades nombradas (NER)
tagged = [nltk.pos_tag(word) for word in words]
entities = [ne_chunk(tag) for tag in tagged]

# Función para extraer nombres de personas y lugares
def extract_info(tree):
    info = {"person": [], "location": []}
    for subtree in tree:
        if type(subtree) == nltk.Tree:
            if subtree.label() == 'GPE':  # Lugar
                info["location"].append(" ".join([token for token, pos in subtree.leaves()]))
            elif subtree.label() == 'PERSON':  # Persona
                info["person"].append(" ".join([token for token, pos in subtree.leaves()]))
    return info

# Realizar la extracción de información
info = extract_info(entities)

# Imprimir la información extraída
print("Nombres de personas:", info["person"])
print("Lugares:", info["location"])
