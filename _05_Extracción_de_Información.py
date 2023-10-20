#Extracci�n de Informaci�n

import nltk
from nltk import ne_chunk
from nltk.tokenize import word_tokenize, sent_tokenize

# Oraci�n de ejemplo
text = "Barack Obama was born in Honolulu, Hawaii. He served as the 44th President of the United States."

# Tokenizaci�n de oraciones y palabras
sentences = sent_tokenize(text)
words = [word_tokenize(sentence) for sentence in sentences]

# Etiquetado de entidades nombradas (NER)
tagged = [nltk.pos_tag(word) for word in words]
entities = [ne_chunk(tag) for tag in tagged]

# Funci�n para extraer nombres de personas y lugares
def extract_info(tree):
    info = {"person": [], "location": []}
    for subtree in tree:
        if type(subtree) == nltk.Tree:
            if subtree.label() == 'GPE':  # Lugar
                info["location"].append(" ".join([token for token, pos in subtree.leaves()]))
            elif subtree.label() == 'PERSON':  # Persona
                info["person"].append(" ".join([token for token, pos in subtree.leaves()]))
    return info

# Realizar la extracci�n de informaci�n
info = extract_info(entities)

# Imprimir la informaci�n extra�da
print("Nombres de personas:", info["person"])
print("Lugares:", info["location"])
