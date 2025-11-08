import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import unicodedata

class Preprocesador:
    def __init__(self, idioma='es'):
        self.idioma = idioma
        self.stopwords = self._cargar_stopwords(idioma)
        self.stemmer = SnowballStemmer('spanish') if idioma == 'es' else SnowballStemmer('english')

    def _cargar_stopwords(self, idioma):
        try:
            if idioma == 'es':
                return set(stopwords.words('spanish'))
            else:
                return set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            return self._cargar_stopwords(idioma)

    def limpiar_texto(self, texto):
        if not isinstance(texto, str) or not texto.strip():
            return "texto_base"
        
        # Minúsculas
        texto = texto.lower()
        
        # Eliminar URLs, menciones, hashtags
        texto = re.sub(r'http\S+', '', texto)
        texto = re.sub(r'@\w+', '', texto)
        texto = re.sub(r'#\w+', '', texto)
        
        # Eliminar puntuación pero mantener acentos españoles
        texto = re.sub(r'[^\w\sáéíóúñü]', ' ', texto)
        
        # Eliminar números solos
        texto = re.sub(r'\b\d+\b', ' ', texto)
        
        # Normalizar espacios
        texto = re.sub(r'\s+', ' ', texto).strip()
        
        return texto if texto else "texto_base"

    def preprocesar(self, texto, usar_stopwords=False, usar_stemming=False):
        texto_limpio = self.limpiar_texto(texto)
        
        if texto_limpio == "texto_base":
            return "texto_base"
            
        tokens = texto_limpio.split()
        
        if not tokens:
            return "texto_base"
        
        # Aplicar stopwords si se solicita
        if usar_stopwords and self.stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
            if not tokens:  # Si se eliminan todos, mantener algunos
                tokens = texto_limpio.split()[:3]  # Primeros 3 tokens como fallback
        
        # Aplicar stemming si se solicita
        if usar_stemming and self.stemmer:
            try:
                tokens = [self.stemmer.stem(t) for t in tokens]
            except:
                pass  # Mantener tokens originales si hay error
        
        texto_final = ' '.join(tokens)
        return texto_final if texto_final.strip() else "texto_base"

    def diagnosticar_preprocesamiento(self, texto_original, usar_stopwords, usar_stemming):
        """Función de diagnóstico para ver qué pasa con el texto"""
        #print(f"Texto original: {texto_original[:100]}...")
        
        texto_limpio = self.limpiar_texto(texto_original)
        #print(f"Después de limpiar: {texto_limpio[:100]}...")
        
        tokens = texto_limpio.split()
        #print(f"Tokens después de limpiar: {len(tokens)}")
        #if tokens:
            #print(f"Primeros 3 tokens: {tokens[:3]}")
        
        if usar_stopwords:
            tokens_sin_stopwords = [t for t in tokens if t not in self.stopwords]
            #print(f"Tokens después de stopwords: {len(tokens_sin_stopwords)}")
            tokens = tokens_sin_stopwords
        
        if usar_stemming and tokens:
            tokens_stemmed = [self.stemmer.stem(t) for t in tokens]
            #print(f"Tokens después de stemming: {len(tokens_stemmed)}")
            tokens = tokens_stemmed
        
        resultado = ' '.join(tokens) if tokens else "VACÍO"
        #print(f"Resultado final: {resultado}")
        #print("-" * 50)
        
        return resultado