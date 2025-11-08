# main_gpu_optimized.py
import json
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')  # Para no mostrar gráficas en pantalla
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Verificar GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Usando dispositivo: {device}")
print(f"🎯 GPU disponible: {torch.cuda.get_device_name(0)}")
print(f"💾 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

class MLP_GPU_Optimized(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, inicializacion='xavier'):
        super(MLP_GPU_Optimized, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
        # Inicialización de pesos
        if inicializacion == 'xavier':
            nn.init.xavier_uniform_(self.hidden.weight)
            nn.init.xavier_uniform_(self.output.weight)
        elif inicializacion == 'normal':
            nn.init.normal_(self.hidden.weight, std=0.01)
            nn.init.normal_(self.output.weight, std=0.01)
        
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.output.bias)
    
    def forward(self, x):
        x = self.hidden(x)
        x = torch.sigmoid(x)
        x = self.output(x)
        return self.sigmoid(x)

def cargar_datos_jsonl(archivo):
    """Carga datos desde archivo JSONL optimizado"""
    datos = []
    try:
        with open(archivo, 'r', encoding='utf-8') as f:
            for linea in f:
                linea = linea.strip()
                if linea:
                    try:
                        datos.append(json.loads(linea))
                    except json.JSONDecodeError:
                        continue
        return datos
    except Exception as e:
        print(f"❌ Error cargando {archivo}: {e}")
        return []

def cargar_datos_completos():
    """Carga todos los datos necesarios optimizado"""
    print("📂 Cargando datos...")
    
    datos_ent = cargar_datos_jsonl('data/hateval_es_train.json')
    datos_pru = cargar_datos_jsonl('data/hateval_es_test.json')
    datos_all = cargar_datos_jsonl('data/hateval_es_all.json')
    
    if not datos_all and datos_ent and datos_pru:
        datos_all = datos_ent + datos_pru
    
    def extraer_textos_etiquetas(datos):
        textos = []
        etiquetas = []
        
        for d in datos:
            texto = d.get('text', '')
            klass = d.get('klass', 0)
            
            if texto and isinstance(texto, str) and texto.strip():
                textos.append(texto.strip())
                etiquetas.append(int(klass))
                
        return textos, etiquetas
    
    X_ent, y_ent = extraer_textos_etiquetas(datos_ent)
    X_pru, y_pru = extraer_textos_etiquetas(datos_pru)
    X_all, y_all = extraer_textos_etiquetas(datos_all)
    
    print(f"✅ Datos cargados:")
    print(f"   - Entrenamiento: {len(X_ent)} textos")
    print(f"   - Prueba: {len(X_pru)} textos")
    print(f"   - Completo: {len(X_all)} textos")
    
    return X_ent, y_ent, X_pru, y_pru, X_all, y_all

class PreprocesadorOptimizado:
    def __init__(self, idioma='es'):
        self.idioma = idioma
        
    def preprocesar_texto(self, texto, usar_stopwords=False, usar_stemming=False):
        """Preprocesamiento optimizado sin NLTK"""
        if not isinstance(texto, str) or not texto.strip():
            return ""
            
        # Minúsculas
        texto = texto.lower()
        
        # Eliminar URLs, menciones, hashtags
        import re
        texto = re.sub(r'http\S+', '', texto)
        texto = re.sub(r'@\w+', '', texto)
        texto = re.sub(r'#\w+', '', texto)
        
        # Eliminar puntuación pero mantener acentos
        texto = re.sub(r'[^\w\sáéíóúñü]', ' ', texto)
        
        # Eliminar números
        texto = re.sub(r'\b\d+\b', ' ', texto)
        
        # Normalizar espacios
        texto = re.sub(r'\s+', ' ', texto).strip()
        
        return texto

def entrenar_modelo_gpu(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, lr=0.01):
    """Entrenamiento optimizado para GPU"""
    model.to(device)
    
    # Convertir datos a tensores PyTorch
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(device)
    
    # Dataset y DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # Validación
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        
        train_losses.append(epoch_train_loss / len(train_loader))
        val_losses.append(val_loss.item())
        
        if epoch % 50 == 0:
            print(f"   Época {epoch}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")
    
    return train_losses, val_losses

def evaluar_modelo_gpu(model, X_test, y_test):
    """Evaluación optimizada para GPU"""
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        predictions = model(X_test_tensor)
        predictions_cpu = predictions.cpu().numpy().flatten()
    
    y_pred_binary = (predictions_cpu > 0.5).astype(int)
    
    precision = precision_score(y_test, y_pred_binary, zero_division=0)
    recall = recall_score(y_test, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, zero_division=0)
    
    return precision, recall, f1, predictions_cpu

def ejecutar_configuracion_optimizada(config, X_ent, y_ent, X_pru, y_pru):
    """Ejecuta una configuración optimizada para GPU"""
    
    # Preprocesamiento
    preprocesador = PreprocesadorOptimizado()
    
    X_ent_limpio = [preprocesador.preprocesar_texto(t) for t in X_ent]
    X_pru_limpio = [preprocesador.preprocesar_texto(t) for t in X_pru]
    
    # Vectorización
    if config['pesado_terminos'] == 'tf':
        vectorizador = CountVectorizer(ngram_range=config['ngramas'], max_features=5000)
    else:
        vectorizador = TfidfVectorizer(ngram_range=config['ngramas'], max_features=5000)
    
    X_ent_vec = vectorizador.fit_transform(X_ent_limpio).toarray()
    X_pru_vec = vectorizador.transform(X_pru_limpio).toarray()
    
    print(f"   Dimensionalidad: {X_ent_vec.shape[1]} features")
    
    # Crear modelo
    modelo = MLP_GPU_Optimized(
        input_size=X_ent_vec.shape[1],
        hidden_size=config['neuronas_ocultas'],
        inicializacion=config['inicializacion']
    )
    
    # Entrenar
    inicio_entrenamiento = time.time()
    train_losses, test_losses = entrenar_modelo_gpu(
        modelo, X_ent_vec, y_ent, X_pru_vec, y_pru,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lr=config['lr']
    )
    tiempo_entrenamiento = time.time() - inicio_entrenamiento
    
    # Evaluar
    precision, recall, f1, predicciones = evaluar_modelo_gpu(modelo, X_pru_vec, y_pru)
    
    resultado = {
        'config': config,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'tiempo_entrenamiento': tiempo_entrenamiento,
        'predicciones': predicciones,
        'vectorizador': vectorizador,
        'modelo': modelo
    }
    
    return resultado

# main_final.py
def main_optimized():
    print("🎯 PRÁCTICA 2 - CLASIFICACIÓN DE DISCURSO DE ODIO")
    print("🚀 OPTIMIZADO PARA GPU NVIDIA RTX 3060")
    print("=" * 60)
    
    # Monitoreo GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"💾 Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Cargar datos
    X_ent, y_ent, X_pru, y_pru, X_all, y_all = cargar_datos_completos()
    
    if not X_ent or not X_pru:
        print("❌ Error: No se pudieron cargar los datos")
        return
    
    resultados = []
    
    # Ejecutar configuraciones
    for i, config in enumerate(CONFIGURACIONES_GPU):
        print(f"\n{'='*50}")
        print(f"⚡ CONFIGURACIÓN {i+1}/{len(CONFIGURACIONES_GPU)}")
        print(f"🧠 Neuronas: {config['neuronas_ocultas']} | 📚 LR: {config['lr']}")
        print(f"🔧 Preproc: {config['preprocesamiento']} | 📊 Ngramas: {config['ngramas']}")
        print(f"{'='*50}")
        
        try:
            inicio = time.time()
            resultado = ejecutar_configuracion_optimizada(config, X_ent, y_ent, X_pru, y_pru)
            tiempo_total = time.time() - inicio
            
            resultado['tiempo_total'] = tiempo_total
            resultados.append(resultado)
            
            print(f"✅ Resultados - F1: {resultado['f1']:.4f} | "
                  f"Precision: {resultado['precision']:.4f} | Recall: {resultado['recall']:.4f}")
            print(f"⏱️ Tiempo entrenamiento: {resultado['tiempo_entrenamiento']:.2f}s | "
                  f"Tiempo total: {tiempo_total:.2f}s")
            
            # Limpiar memoria GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Error en configuración {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generar reportes completos
    generar_reporte_completo_gpu(resultados, CONFIGURACIONES_GPU, X_pru, y_pru)
    
    print(f"\n🎉 PRÁCTICA COMPLETADA!")
    print(f"📁 Reportes guardados en: resultados/")
    print(f"📊 Total configuraciones ejecutadas: {len(resultados)}")

if __name__ == '__main__':
    main_optimized()