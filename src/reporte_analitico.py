import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from src.preprocesamiento import Preprocesador
from src.representaciones import crear_vectorizador
from src.mlp_gpu import MLP_GPU
from src.entrenamiento_gpu import entrenar_mlp_gpu, evaluar_modelo_gpu

def generar_reporte_completo(resultados, configuraciones, X_all, y_all, X_ent, y_ent, X_pru, y_pru):
    """Genera reporte completo en archivos TXT"""
    
    # Identificar las 3 mejores configuraciones
    mejores_indices = np.argsort([r['f1'] for r in resultados])[-3:][::-1]
    
    # Realizar validación cruzada para las 3 mejores
    resultados_cv = validacion_cruzada_top3(mejores_indices, configuraciones, X_all, y_all)
    
    # Generar reporte principal
    generar_reporte_principal(resultados, configuraciones, mejores_indices, resultados_cv)
    
    # Generar análisis comparativo
    generar_analisis_comparativo(resultados, configuraciones)
    
    # Generar análisis de hiperparámetros
    generar_analisis_hiperparametros(resultados, configuraciones)
    
    print("Reporte completo generado en la carpeta resultados/")

def validacion_cruzada_top3(mejores_indices, configuraciones, X_all, y_all):
    """Realiza validación cruzada 5-fold para las 3 mejores configuraciones"""
    resultados_cv = {}
    
    for idx in mejores_indices:
        config = configuraciones[idx]
        print(f"Validacion cruzada para Configuracion {idx+1}")
        
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
            print(f"  Fold {fold+1}/5...")
            
            X_train, X_val = [X_all[i] for i in train_idx], [X_all[i] for i in val_idx]
            y_train, y_val = [y_all[i] for i in train_idx], [y_all[i] for i in val_idx]
            
            # Preprocesamiento
            preprocesador = Preprocesador(idioma='es')
            X_train_limpio = [preprocesador.preprocesar(t) for t in X_train]
            X_val_limpio = [preprocesador.preprocesar(t) for t in X_val]
            
            # Vectorización
            vectorizador = crear_vectorizador(config['pesado_terminos'], config['ngramas'])
            X_train_vec = vectorizador.fit_transform(X_train_limpio).toarray()
            X_val_vec = vectorizador.transform(X_val_limpio).toarray()
            
            # Modelo y entrenamiento
            modelo = MLP_GPU(
                input_size=X_train_vec.shape[1],
                hidden_size=config['neuronas_ocultas'],
                output_size=1,
                inicializacion=config['inicializacion']
            )
            
            y_train_arr = np.array(y_train).reshape(-1, 1)
            y_val_arr = np.array(y_val).reshape(-1, 1)
            
            # Entrenamiento rápido para CV
            train_losses, val_losses = entrenar_mlp_gpu(
                modelo, X_train_vec, y_train_arr, X_val_vec, y_val_arr,
                epochs=50,
                batch_size=config['batch_size'],
                lr=config['lr']
            )
            
            # Evaluación
            precision, recall, f1 = evaluar_modelo_gpu(modelo, X_val_vec, y_val_arr)
            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        resultados_cv[idx] = {
            'config': config,
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'precision_mean': np.mean(precision_scores),
            'precision_std': np.std(precision_scores),
            'recall_mean': np.mean(recall_scores),
            'recall_std': np.std(recall_scores),
            'f1_scores': f1_scores
        }
        
        print(f"  F1-score CV: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    
    return resultados_cv

def generar_reporte_principal(resultados, configuraciones, mejores_indices, resultados_cv):
    """Genera el reporte principal con todos los resultados"""
    
    with open('resultados/reporte_analitico_completo.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("REPORTE COMPLETO - PRÁCTICA 2: CLASIFICACIÓN DE DISCURSO DE ODIO\n")
        f.write("=" * 100 + "\n\n")
        
        
        f.write("RESUMEN \n")
        f.write("-" * 50 + "\n")
        f.write(f"Total de configuraciones evaluadas: {len(configuraciones)}\n")
        f.write(f"Mejor F1-score obtenido: {max([r['f1'] for r in resultados]):.4f}\n")
        f.write(f"Peor F1-score obtenido: {min([r['f1'] for r in resultados if r['f1'] > 0]):.4f}\n")
        f.write(f"F1-score promedio: {np.mean([r['f1'] for r in resultados if r['f1'] > 0]):.4f}\n\n")
        
        # Top 3 configuraciones
        f.write("TOP 3 CONFIGURACIONES (POR F1-SCORE)\n")
        f.write("-" * 50 + "\n")
        for i, idx in enumerate(mejores_indices):
            config = configuraciones[idx]
            res = resultados[idx]
            f.write(f"{i+1}. CONFIGURACIÓN {idx+1} - F1: {res['f1']:.4f}\n")
            f.write(f"   Neuronas ocultas: {config['neuronas_ocultas']}\n")
            f.write(f"   Inicialización: {config['inicializacion']}\n")
            f.write(f"   Pesado: {config['pesado_terminos']}\n")
            f.write(f"   N-gramas: {config['ngramas']}\n")
            f.write(f"   Preprocesamiento: {config['preprocesamiento']}\n")
            f.write(f"   Learning rate: {config['lr']}\n")
            f.write(f"   Batch size: {config['batch_size']}\n")
            f.write(f"   Precision: {res['precision']:.4f}, Recall: {res['recall']:.4f}\n")
            f.write(f"   Tiempo ejecución: {res['tiempo_ejecucion']:.2f}s\n\n")
        
        # Validación cruzada
        f.write("VALIDACIÓN CRUZADA (5-FOLD) - TOP 3 CONFIGURACIONES\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Config':^8} {'F1-Score':^12} {'Precision':^12} {'Recall':^12} {'Estabilidad':^15}\n")
        f.write("-" * 70 + "\n")
        for idx in mejores_indices:
            res_cv = resultados_cv[idx]
            estabilidad = "Alta" if res_cv['f1_std'] < 0.05 else "Media" if res_cv['f1_std'] < 0.1 else "Baja"
            f.write(f"{idx+1:^8} {res_cv['f1_mean']:.4f} ± {res_cv['f1_std']:.4f}  "
                   f"{res_cv['precision_mean']:.4f} ± {res_cv['precision_std']:.4f}  "
                   f"{res_cv['recall_mean']:.4f} ± {res_cv['recall_std']:.4f}  "
                   f"{estabilidad:^15}\n")
        f.write("\n")
        
        # Tabla completa de resultados
        f.write("TABLA COMPLETA DE RESULTADOS - TODAS LAS CONFIGURACIONES\n")
        f.write("=" * 120 + "\n")
        f.write(f"{'Config':^6} {'Neuronas':^10} {'Inicial':^10} {'Pesado':^8} {'Ngramas':^12} {'Preproc':^25} {'LR':^6} {'Batch':^6} {'F1':^8} {'Precision':^10} {'Recall':^8} {'Tiempo':^10}\n")
        f.write("-" * 120 + "\n")
        
        for i, (config, res) in enumerate(zip(configuraciones, resultados)):
            preproc_nombre = config['preprocesamiento']
            if len(preproc_nombre) > 25:
                preproc_nombre = preproc_nombre[:22] + "..."
                
            f.write(f"{i+1:^6} {config['neuronas_ocultas']:^10} {config['inicializacion']:^10} "
                   f"{config['pesado_terminos']:^8} {str(config['ngramas']):^12} {preproc_nombre:^25} "
                   f"{config['lr']:^6} {config['batch_size']:^6} {res['f1']:^8.3f} {res['precision']:^10.3f} "
                   f"{res['recall']:^8.3f} {res['tiempo_ejecucion']:^10.1f}\n")

def generar_analisis_comparativo(resultados, configuraciones):
    """Genera análisis comparativo entre diferentes configuraciones"""
    
    with open('resultados/analisis_comparativo.txt', 'w', encoding='utf-8') as f:
        f.write("ANÁLISIS COMPARATIVO DETALLADO\n")
        f.write("=" * 80 + "\n\n")
        
        # Análisis por tipo de parámetro
        f.write("1. ANÁLISIS POR NEURONAS OCULTAS\n")
        f.write("-" * 40 + "\n")
        neuronas_analysis = {}
        for config, res in zip(configuraciones, resultados):
            if res['f1'] > 0:  # Excluir configuraciones fallidas
                n = config['neuronas_ocultas']
                if n not in neuronas_analysis:
                    neuronas_analysis[n] = []
                neuronas_analysis[n].append(res['f1'])
        
        for n in sorted(neuronas_analysis.keys()):
            f1_values = neuronas_analysis[n]
            f.write(f"  {n} neuronas: F1 promedio = {np.mean(f1_values):.4f} "
                   f"(min: {np.min(f1_values):.4f}, max: {np.max(f1_values):.4f})\n")
        
        mejor_neuronas = max(neuronas_analysis, key=lambda x: np.mean(neuronas_analysis[x]))
        f.write(f"  MEJOR: {mejor_neuronas} neuronas\n\n")
        
        # Análisis por inicialización
        f.write("2. ANÁLISIS POR INICIALIZACIÓN\n")
        f.write("-" * 40 + "\n")
        init_analysis = {}
        for config, res in zip(configuraciones, resultados):
            if res['f1'] > 0:
                init = config['inicializacion']
                if init not in init_analysis:
                    init_analysis[init] = []
                init_analysis[init].append(res['f1'])
        
        for init in init_analysis.keys():
            f1_values = init_analysis[init]
            f.write(f"  {init}: F1 promedio = {np.mean(f1_values):.4f} "
                   f"(min: {np.min(f1_values):.4f}, max: {np.max(f1_values):.4f})\n")
        
        mejor_init = max(init_analysis, key=lambda x: np.mean(init_analysis[x]))
        f.write(f"  MEJOR: {mejor_init}\n\n")
        
        # Análisis por pesado de términos
        f.write("3. ANÁLISIS POR PESADO DE TÉRMINOS\n")
        f.write("-" * 40 + "\n")
        peso_analysis = {}
        for config, res in zip(configuraciones, resultados):
            if res['f1'] > 0:
                peso = config['pesado_terminos']
                if peso not in peso_analysis:
                    peso_analysis[peso] = []
                peso_analysis[peso].append(res['f1'])
        
        for peso in peso_analysis.keys():
            f1_values = peso_analysis[peso]
            f.write(f"  {peso}: F1 promedio = {np.mean(f1_values):.4f} "
                   f"(min: {np.min(f1_values):.4f}, max: {np.max(f1_values):.4f})\n")
        
        mejor_peso = max(peso_analysis, key=lambda x: np.mean(peso_analysis[x]))
        f.write(f"  MEJOR: {mejor_peso}\n\n")
        
        # Análisis por n-gramas
        f.write("4. ANÁLISIS POR N-GRAMAS\n")
        f.write("-" * 40 + "\n")
        ngram_analysis = {}
        for config, res in zip(configuraciones, resultados):
            if res['f1'] > 0:
                ngram = str(config['ngramas'])
                if ngram not in ngram_analysis:
                    ngram_analysis[ngram] = []
                ngram_analysis[ngram].append(res['f1'])
        
        for ngram in ngram_analysis.keys():
            f1_values = ngram_analysis[ngram]
            f.write(f"  {ngram}: F1 promedio = {np.mean(f1_values):.4f} "
                   f"(min: {np.min(f1_values):.4f}, max: {np.max(f1_values):.4f})\n")
        
        mejor_ngram = max(ngram_analysis, key=lambda x: np.mean(ngram_analysis[x]))
        f.write(f"  MEJOR: {mejor_ngram}\n\n")
        
        # Análisis por preprocesamiento
        f.write("5. ANÁLISIS POR PREPROCESAMIENTO\n")
        f.write("-" * 40 + "\n")
        preproc_analysis = {}
        for config, res in zip(configuraciones, resultados):
            if res['f1'] > 0:
                preproc = config['preprocesamiento']
                if preproc not in preproc_analysis:
                    preproc_analysis[preproc] = []
                preproc_analysis[preproc].append(res['f1'])
        
        for preproc in preproc_analysis.keys():
            f1_values = preproc_analysis[preproc]
            f.write(f"  {preproc}: F1 promedio = {np.mean(f1_values):.4f} "
                   f"(min: {np.min(f1_values):.4f}, max: {np.max(f1_values):.4f})\n")
        
        mejor_preproc = max(preproc_analysis, key=lambda x: np.mean(preproc_analysis[x]))
        f.write(f"  MEJOR: {mejor_preproc}\n\n")
        
        # Análisis por learning rate
        f.write("6. ANÁLISIS POR LEARNING RATE\n")
        f.write("-" * 40 + "\n")
        lr_analysis = {}
        for config, res in zip(configuraciones, resultados):
            if res['f1'] > 0:
                lr = config['lr']
                if lr not in lr_analysis:
                    lr_analysis[lr] = []
                lr_analysis[lr].append(res['f1'])
        
        for lr in sorted(lr_analysis.keys()):
            f1_values = lr_analysis[lr]
            f.write(f"  {lr}: F1 promedio = {np.mean(f1_values):.4f} "
                   f"(min: {np.min(f1_values):.4f}, max: {np.max(f1_values):.4f})\n")
        
        mejor_lr = max(lr_analysis, key=lambda x: np.mean(lr_analysis[x]))
        f.write(f"  MEJOR: {mejor_lr}\n\n")
        
        # Análisis por batch size
        f.write("7. ANÁLISIS POR BATCH SIZE\n")
        f.write("-" * 40 + "\n")
        batch_analysis = {}
        for config, res in zip(configuraciones, resultados):
            if res['f1'] > 0:
                batch = config['batch_size']
                if batch not in batch_analysis:
                    batch_analysis[batch] = []
                batch_analysis[batch].append(res['f1'])
        
        for batch in sorted(batch_analysis.keys()):
            f1_values = batch_analysis[batch]
            f.write(f"  {batch}: F1 promedio = {np.mean(f1_values):.4f} "
                   f"(min: {np.min(f1_values):.4f}, max: {np.max(f1_values):.4f})\n")
        
        mejor_batch = max(batch_analysis, key=lambda x: np.mean(batch_analysis[x]))
        f.write(f"  MEJOR: {mejor_batch}\n")

def generar_analisis_hiperparametros(resultados, configuraciones):
    """Genera análisis detallado del impacto de cada hiperparámetro"""
    
    with open('resultados/analisis_hiperparametros.txt', 'w', encoding='utf-8') as f:
        f.write("ANÁLISIS DETALLADO DE HIPERPARÁMETROS\n")
        f.write("=" * 80 + "\n\n")
        
        # Crear DataFrame para análisis
        datos = []
        for i, (config, res) in enumerate(zip(configuraciones, resultados)):
            if res['f1'] > 0:  # Solo configuraciones exitosas
                datos.append({
                    'config': i+1,
                    'neuronas': config['neuronas_ocultas'],
                    'inicializacion': config['inicializacion'],
                    'pesado': config['pesado_terminos'],
                    'ngramas': str(config['ngramas']),
                    'preprocesamiento': config['preprocesamiento'],
                    'lr': config['lr'],
                    'batch_size': config['batch_size'],
                    'f1': res['f1'],
                    'precision': res['precision'],
                    'recall': res['recall'],
                    'tiempo': res['tiempo_ejecucion']
                })
        
        df = pd.DataFrame(datos)
        
      
        
        f.write("CONFIGURACIÓN ÓPTIMA RECOMENDADA:\n")
        mejor_config_idx = df.loc[df['f1'].idxmax()]
        f.write(f"- Neuronas ocultas: {int(mejor_config_idx['neuronas'])}\n")
        f.write(f"- Inicialización: {mejor_config_idx['inicializacion']}\n")
        f.write(f"- Pesado de términos: {mejor_config_idx['pesado']}\n")
        f.write(f"- N-gramas: {mejor_config_idx['ngramas']}\n")
        f.write(f"- Preprocesamiento: {mejor_config_idx['preprocesamiento']}\n")
        f.write(f"- Learning rate: {mejor_config_idx['lr']}\n")
        f.write(f"- Batch size: {int(mejor_config_idx['batch_size'])}\n")
        f.write(f"- F1-score esperado: {mejor_config_idx['f1']:.4f}\n\n")
        