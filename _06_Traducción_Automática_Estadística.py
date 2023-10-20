#Traducci�n Autom�tica Estad�stica

from googletrans import Translator

# Texto que deseas traducir
texto_a_traducir = "Hello, how are you?"

# Crea una instancia del traductor
translator = Translator()

# Idioma de destino
idioma_destino = "es"  # Por ejemplo, espa�ol

# Realiza la traducci�n
traduccion = translator.translate(texto_a_traducir, dest=idioma_destino)

# Imprime el resultado
print(f"Texto original: {texto_a_traducir}")
print(f"Traducci�n ({idioma_destino}): {traduccion.text}")


