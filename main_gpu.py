import json
import numpy as np
import os
import time
from src.preprocesamiento import Preprocesador
from src.representaciones import crear_vectorizador
from src.mlp_gpu import MLP_GPU
from src.entrenamiento_gpu import entrenar_mlp_gpu, evaluar_modelo_gpu
from src.visualizacion import graficar_perdidas, graficar_metricas_comparativas, generar_tabla_resultados
from src.monitor_gpu import GPUMonitor
from configs import CONFIGURACIONES
from src.analisis_errores import analizar_errores_detallados
from src.preprocesamiento import Preprocesador
from src.reporte_analitico import generar_reporte_completo


def cargar_datos_jsonl(archivo):
    """Carga datos desde archivo JSONL"""
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
        print(f" {len(datos)} registros cargados desde {archivo}")
        return datos
    except Exception as e:
        print(f" Error cargando {archivo}: {e}")
        return []

def cargar_datos_completos():
    """Carga todos los datos necesarios"""
    print(" Cargando datos...")
    
    datos_ent = cargar_datos_jsonl('data/hateval_es_train.json')
    datos_pru = cargar_datos_jsonl('data/hateval_es_test.json')
    datos_all = cargar_datos_jsonl('data/hateval_es_all.json')
    
    if not datos_all and datos_ent and datos_pru:
        print(" Combinando train y test para crear dataset completo...")
        datos_all = datos_ent + datos_pru
    
    def extraer_textos_etiquetas(datos):
        textos = []
        etiquetas = []
        textos_vacios = 0
        
        for d in datos:
            texto = d.get('text', '')
            klass = d.get('klass', 0)
            
            if texto and isinstance(texto, str) and texto.strip():
                textos.append(texto.strip())
                etiquetas.append(int(klass))
            else:
                textos_vacios += 1
                
        if textos_vacios > 0:
            print(f"  Se omitieron {textos_vacios} textos vacíos/inválidos")
            
        return textos, etiquetas
    
    X_ent, y_ent = extraer_textos_etiquetas(datos_ent)
    X_pru, y_pru = extraer_textos_etiquetas(datos_pru)
    X_all, y_all = extraer_textos_etiquetas(datos_all)
    
    print(f" Resumen de datos:")
    print(f"   - Entrenamiento: {len(X_ent)} textos")
    print(f"   - Prueba: {len(X_pru)} textos")
    print(f"   - Completo: {len(X_all)} textos")
    
    return X_ent, y_ent, X_pru, y_pru, X_all, y_all


def ejecutar_configuracion(config, X_ent, y_ent, X_pru, y_pru, textos_pru_original=None):
    """Ejecuta una configuración específica mejorada"""
    
    # Configurar preprocesamiento
    preprocesador = Preprocesador(idioma='es')
    
    preprocesamiento_type = config.get('preprocesamiento', 'normalizar')
    if preprocesamiento_type == 'normalizar':
        usar_stopwords, usar_stemming = False, False
    elif preprocesamiento_type == 'normalizar_sin_stopwords':
        usar_stopwords, usar_stemming = True, False
    elif preprocesamiento_type == 'normalizar_sin_stopwords_stemming':
        usar_stopwords, usar_stemming = True, True
    else:
        usar_stopwords, usar_stemming = False, False
    
    print(f"Preprocesamiento: {preprocesamiento_type}")
    
    # Preprocesar textos
    X_ent_limpio = [preprocesador.preprocesar(t, usar_stopwords, usar_stemming) for t in X_ent]
    X_pru_limpio = [preprocesador.preprocesar(t, usar_stopwords, usar_stemming) for t in X_pru]
    
    # Verificar que no todos los textos sean "texto_base"
    textos_unicos = set(X_ent_limpio)
    if len(textos_unicos) <= 1 and "texto_base" in textos_unicos:
        print("ADVERTENCIA: Demasiados textos convertidos a 'texto_base'")
        # Usar texto mínimo procesado como fallback
        X_ent_limpio = [preprocesador.limpiar_texto(t) for t in X_ent]
        X_pru_limpio = [preprocesador.limpiar_texto(t) for t in X_pru]
    
    # Vectorización con parámetros optimizados
    try:
        vectorizador = crear_vectorizador(
            tipo=config['pesado_terminos'],
            ngram_range=config['ngramas']
        )
        
        X_ent_vec = vectorizador.fit_transform(X_ent_limpio).toarray()
        X_pru_vec = vectorizador.transform(X_pru_limpio).toarray()
        
        print(f"Dimensionalidad: {X_ent_vec.shape[1]} features")
        
        if X_ent_vec.shape[1] == 0:
            raise ValueError("No se generaron features")
            
    except Exception as e:
        print(f"Error en vectorización: {e}")
        # Vectorizador de respaldo
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizador = CountVectorizer(
            ngram_range=config['ngramas'],
            min_df=2,
            max_df=0.95,
            max_features=5000
        )
        X_ent_vec = vectorizador.fit_transform(X_ent_limpio).toarray()
        X_pru_vec = vectorizador.transform(X_pru_limpio).toarray()
        print(f"Dimensionalidad con respaldo: {X_ent_vec.shape[1]} features")
    
    # Crear y entrenar modelo
    modelo = MLP_GPU(
        input_size=X_ent_vec.shape[1],
        hidden_size=config['neuronas_ocultas'],
        output_size=1,
        inicializacion=config['inicializacion']
    )
    
    y_ent_arr = np.array(y_ent).reshape(-1, 1)
    y_pru_arr = np.array(y_pru).reshape(-1, 1)
    
    print("Entrenando modelo...")
    train_losses, test_losses = entrenar_mlp_gpu(
        modelo, X_ent_vec, y_ent_arr, X_pru_vec, y_pru_arr,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lr=config['lr']
    )
    
    # Evaluar
    precision, recall, f1 = evaluar_modelo_gpu(modelo, X_pru_vec, y_pru_arr)
    
    resultado = {
        'config': config,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'vectorizador': vectorizador,
        'modelo': modelo
    }
    
    return resultado


def diagnosticar_problema_preprocesamiento():
    """Diagnóstico rápido del problema de preprocesamiento"""
    
    
    preprocesador = Preprocesador(idioma='es')
    
    # Cargar algunos datos de ejemplo
    X_ent, y_ent, X_pru, y_pru, X_all, y_all = cargar_datos_completos()
    
    print("=" * 80)
    print("DIAGNÓSTICO DE PREPROCESAMIENTO")
    print("=" * 80)
    
    # Probar diferentes configuraciones de preprocesamiento
    configuraciones_prueba = [
        ('normalizar', False, False),
        ('normalizar_sin_stopwords', True, False),
        ('normalizar_sin_stopwords_stemming', True, True)
    ]
    
    for config_name, stopwords, stemming in configuraciones_prueba:
        print(f"\nCONFIGURACIÓN: {config_name}")
        print(f"Stopwords: {stopwords}, Stemming: {stemming}")
        
        # Probar con los primeros 3 textos
        for i in range(3):
            if i < len(X_ent):
                resultado = preprocesador.diagnosticar_preprocesamiento(
                    X_ent[i], stopwords, stemming
                )
    
    print("=" * 80)

def main():
    print("PRÁCTICA 2 - CLASIFICACIÓN DE DISCURSO DE ODIO")
    print("=" * 60)
    
    
    # Cargar datos
    X_ent, y_ent, X_pru, y_pru, X_all, y_all = cargar_datos_completos()
    
    if not X_ent or not X_pru:
        print("Error: No se pudieron cargar los datos")
        return
    
    print(f"Datos cargados: {len(X_ent)} entrenamiento, {len(X_pru)} prueba")
    
    # Ejecutar configuraciones
    resultados = []
    
    for i, config in enumerate(CONFIGURACIONES):
        print(f"\n{'='*50}")
        print(f"CONFIGURACIÓN {i+1}/{len(CONFIGURACIONES)}")
        print(f"Neuronas: {config['neuronas_ocultas']}, LR: {config['lr']}")
        print(f"Preproc: {config['preprocesamiento']}, Ngramas: {config['ngramas']}")
        print(f"{'='*50}")
        
        try:
            inicio = time.time()
            resultado = ejecutar_configuracion(config, X_ent, y_ent, X_pru, y_pru)
            tiempo = time.time() - inicio
            
            resultado['tiempo_ejecucion'] = tiempo
            resultados.append(resultado)
            
            print(f"Resultados - F1: {resultado['f1']:.4f}, Precision: {resultado['precision']:.4f}, Recall: {resultado['recall']:.4f}")
            print(f"Tiempo: {tiempo:.2f}s")
            
        except Exception as e:
            print(f"Error en configuración {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generar reportes
    generar_reporte_completo(resultados, CONFIGURACIONES, X_all, y_all, X_ent, y_ent, X_pru, y_pru)
    
    print("\nPRÁCTICA COMPLETADA")
    print("Revise los archivos en la carpeta 'resultados/'")

if __name__ == '__main__':
    main()