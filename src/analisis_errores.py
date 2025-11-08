import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import cupy as cp

def analizar_errores_detallados(modelo, X_test, y_test, vectorizador, config, textos_originales):
    """Análisis completo de errores de clasificación"""
    try:
        # Convertir a GPU para predicciones
        X_test_gpu = cp.asarray(X_test, dtype=cp.float32)
        preds = modelo.forward(X_test_gpu)
        preds_cpu = cp.asnumpy(preds)
        
        # Obtener predicciones binarias y probabilidades
        y_pred = (preds_cpu > 0.5).astype(int).flatten()
        y_true = np.array(y_test).flatten()
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        
        # Análisis detallado de errores
        falsos_positivos_indices = np.where((y_true == 0) & (y_pred == 1))[0]
        falsos_negativos_indices = np.where((y_true == 1) & (y_pred == 0))[0]
        
        # Ejemplos de errores (limitado a 5 de cada tipo)
        ejemplos_fp = []
        ejemplos_fn = []
        
        for idx in falsos_positivos_indices[:5]:  # Solo primeros 5
            if idx < len(textos_originales):
                ejemplos_fp.append({
                    'texto': textos_originales[idx],
                    'probabilidad': float(preds_cpu[idx][0])
                })
        
        for idx in falsos_negativos_indices[:5]:
            if idx < len(textos_originales):
                ejemplos_fn.append({
                    'texto': textos_originales[idx],
                    'probabilidad': float(preds_cpu[idx][0])
                })
        
        return {
            'matriz_confusion': cm,
            'reporte_clasificacion': classification_report(y_true, y_pred, output_dict=True),
            'falsos_positivos': len(falsos_positivos_indices),
            'falsos_negativos': len(falsos_negativos_indices),
            'ejemplos_fp': ejemplos_fp,
            'ejemplos_fn': ejemplos_fn,
            'y_true': y_true,
            'y_pred': y_pred,
            'predicciones_prob': preds_cpu.flatten()
        }
        
    except Exception as e:
        print(f" Error en análisis de errores detallados: {e}")
        # Retornar análisis básico
        return {
            'matriz_confusion': None,
            'falsos_positivos': 0,
            'falsos_negativos': 0,
            'ejemplos_fp': [],
            'ejemplos_fn': []
        }

def graficar_matrices_confusion(resultados, configuraciones, top_n=3):
    """Grafica matrices de confusión para las mejores configuraciones"""
    try:
        # Filtrar configuraciones que tienen análisis de errores
        resultados_con_errores = [r for r in resultados if 'analisis_errores' in r and r['analisis_errores']['matriz_confusion'] is not None]
        
        if not resultados_con_errores:
            print(" No hay matrices de confusión disponibles para graficar")
            return
        
        # Tomar las mejores configuraciones que tienen análisis de errores
        indices_top = np.argsort([r['f1'] for r in resultados_con_errores])[-top_n:][::-1]
        
        fig, axes = plt.subplots(1, min(top_n, len(indices_top)), figsize=(15, 5))
        if min(top_n, len(indices_top)) == 1:
            axes = [axes]
        
        for i, idx in enumerate(indices_top[:top_n]):
            res = resultados_con_errores[idx]
            cm = res['analisis_errores']['matriz_confusion']
            
            # Crear gráfica manualmente
            im = axes[i].imshow(cm, cmap='Blues', interpolation='nearest')
            axes[i].set_title(f'Config {resultados.index(res)+1}\nF1: {res["f1"]:.3f}')
            axes[i].set_xlabel('Predicho')
            axes[i].set_ylabel('Real')
            axes[i].set_xticks([0, 1])
            axes[i].set_yticks([0, 1])
            axes[i].set_xticklabels(['No Hate', 'Hate'])
            axes[i].set_yticklabels(['No Hate', 'Hate'])
            
            # Añadir valores en las celdas
            for x in range(2):
                for y in range(2):
                    axes[i].text(y, x, str(cm[x, y]), 
                               ha='center', va='center', 
                               color='white' if cm[x, y] > cm.max()/2 else 'black',
                               fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('resultados/graficas/matrices_confusion_top.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(" Gráfica de matrices de confusión generada")
        
    except Exception as e:
        print(f" Error en graficar_matrices_confusion: {e}")

def graficar_distribucion_errores(resultados):
    """Grafica la distribución de errores por configuración"""
    try:
        fp_counts = []
        fn_counts = []
        config_names = []
        f1_scores = []
        
        for i, res in enumerate(resultados):
            if 'analisis_errores' in res:
                fp_counts.append(res['analisis_errores']['falsos_positivos'])
                fn_counts.append(res['analisis_errores']['falsos_negativos'])
                config_names.append(f'Config {i+1}')
                f1_scores.append(res['f1'])
        
        if not fp_counts:
            print(" No hay datos de errores para graficar")
            return
        
        x = np.arange(len(config_names))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Gráfica de barras de errores
        ax1.bar(x - width/2, fp_counts, width, label='Falsos Positivos', alpha=0.7, color='red')
        ax1.bar(x + width/2, fn_counts, width, label='Falsos Negativos', alpha=0.7, color='blue')
        ax1.set_xlabel('Configuraciones')
        ax1.set_ylabel('Cantidad de Errores')
        ax1.set_title('Distribución de Errores por Configuración')
        ax1.legend()
        ax1.set_xticks(x)
        ax1.set_xticklabels(config_names, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Gráfica de F1-score vs Errores
        total_errores = [fp + fn for fp, fn in zip(fp_counts, fn_counts)]
        scatter = ax2.scatter(total_errores, f1_scores, c=range(len(config_names)), 
                            cmap='viridis', s=100, alpha=0.7)
        ax2.set_xlabel('Total de Errores (FP + FN)')
        ax2.set_ylabel('F1-Score')
        ax2.set_title('F1-Score vs Total de Errores')
        ax2.grid(True, alpha=0.3)
        
        # Añadir etiquetas
        for i, (errores, f1) in enumerate(zip(total_errores, f1_scores)):
            ax2.annotate(f'{i+1}', (errores, f1), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('resultados/graficas/distribucion_errores.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(" Gráfica de distribución de errores generada")
        
    except Exception as e:
        print(f" Error en graficar_distribucion_errores: {e}")

def generar_reporte_errores_detallado(resultados, configuraciones):
    """Genera reporte detallado de errores con ejemplos"""
    try:
        with open('resultados/analisis_errores_detallado.txt', 'w', encoding='utf-8') as f:
            f.write("ANÁLISIS DETALLADO DE ERRORES\n")
            f.write("=" * 80 + "\n\n")
            
            for i, (config, res) in enumerate(zip(configuraciones, resultados)):
                if 'analisis_errores' in res and res['analisis_errores']:
                    errores = res['analisis_errores']
                    
                    f.write(f"CONFIGURACIÓN {i+1}\n")
                    f.write(f"F1-score: {res['f1']:.4f}\n")
                    f.write(f"Precision: {res['precision']:.4f}, Recall: {res['recall']:.4f}\n")
                    f.write("-" * 50 + "\n")
                    
                    f.write(f"Falsos Positivos: {errores['falsos_positivos']}\n")
                    f.write(f"Falsos Negativos: {errores['falsos_negativos']}\n")
                    
                    if errores['matriz_confusion'] is not None:
                        f.write(f"Matriz de Confusión:\n")
                        f.write(f"  Real\\Pred | No Hate | Hate\n")
                        f.write(f"  ----------+---------+------\n")
                        f.write(f"  No Hate   |   {errores['matriz_confusion'][0,0]:^5}  | {errores['matriz_confusion'][0,1]:^5}\n")
                        f.write(f"  Hate      |   {errores['matriz_confusion'][1,0]:^5}  | {errores['matriz_confusion'][1,1]:^5}\n")
                    
                    # Ejemplos de errores
                    if errores['ejemplos_fp']:
                        f.write(f"\nEjemplos de Falsos Positivos (No Hate -> Hate):\n")
                        for j, ejemplo in enumerate(errores['ejemplos_fp'][:3]):
                            f.write(f"  {j+1}. Prob: {ejemplo['probabilidad']:.3f} - {ejemplo['texto'][:100]}...\n")
                    
                    if errores['ejemplos_fn']:
                        f.write(f"\nEjemplos de Falsos Negativos (Hate -> No Hate):\n")
                        for j, ejemplo in enumerate(errores['ejemplos_fn'][:3]):
                            f.write(f"  {j+1}. Prob: {ejemplo['probabilidad']:.3f} - {ejemplo['texto'][:100]}...\n")
                    
                    f.write("\n" + "="*80 + "\n\n")
            
        print(" Reporte detallado de errores generado")
        
    except Exception as e:
        print(f" Error en generar_reporte_errores_detallado: {e}")

# Alias para mantener compatibilidad
generar_reporte_errores = generar_reporte_errores_detallado