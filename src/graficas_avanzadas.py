import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def graficar_curvas_aprendizaje_comparativas(resultados, configuraciones):
    """Compara curvas de aprendizaje de todas las configuraciones"""
    try:
        plt.figure(figsize=(15, 8))
        
        # Solo graficar las 5 mejores configuraciones para mayor claridad
        indices_top = np.argsort([r['f1'] for r in resultados])[-5:][::-1]
        
        for i, idx in enumerate(indices_top):
            config = configuraciones[idx]
            res = resultados[idx]
            
            if len(res['train_losses']) > 0:
                epochs = range(len(res['train_losses']))
                plt.plot(epochs, res['train_losses'], 
                        label=f'Config {idx+1} (F1: {res["f1"]:.3f})',
                        alpha=0.7, linewidth=2)
        
        plt.title('Curvas de Pérdida de Entrenamiento - Top 5 Configuraciones')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('resultados/graficas/curvas_aprendizaje_comparativas.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(" Gráfica de curvas de aprendizaje generada")
        
    except Exception as e:
        print(f" Error en graficar_curvas_aprendizaje_comparativas: {e}")

def graficar_evolucion_metricas(resultados, configuraciones):
    """Grafica la evolución de métricas durante el entrenamiento"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Seleccionar las 4 mejores configuraciones
        mejores_indices = np.argsort([r['f1'] for r in resultados])[-4:][::-1]
        
        for i, idx in enumerate(mejores_indices):
            if i >= len(axes):
                break
                
            config = configuraciones[idx]
            res = resultados[idx]
            
            # Graficar pérdidas de entrenamiento y validación
            if len(res['train_losses']) > 0:
                epochs = range(len(res['train_losses']))
                axes[i].plot(epochs, res['train_losses'], label='Train Loss', alpha=0.8, linewidth=2)
                axes[i].plot(epochs, res['test_losses'], label='Test Loss', alpha=0.8, linewidth=2)
            
            axes[i].set_title(f'Config {idx+1} - F1: {res["f1"]:.3f}')
            axes[i].set_xlabel('Épocas')
            axes[i].set_ylabel('Pérdida')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('resultados/graficas/evolucion_metricas_top4.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(" Gráfica de evolución de métricas generada")
        
    except Exception as e:
        print(f" Error en graficar_evolucion_metricas: {e}")

def graficar_boxplot_hiperparametros(resultados, configuraciones):
    """Boxplots del rendimiento por categoría de hiperparámetro"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Organizar datos por hiperparámetro
        hiperparametros = ['neuronas_ocultas', 'inicializacion', 'pesado_terminos', 'lr', 'batch_size']
        
        for i, hp in enumerate(hiperparametros):
            if i >= len(axes):
                break
                
            datos_boxplot = []
            etiquetas = []
            
            # Agrupar por valor del hiperparámetro
            valores_unicos = list(set([str(config[hp]) for config in configuraciones]))
            
            for valor in valores_unicos:
                f1_valores = []
                for config, res in zip(configuraciones, resultados):
                    if str(config[hp]) == valor and res['f1'] > 0:
                        f1_valores.append(res['f1'])
                
                if f1_valores:
                    datos_boxplot.append(f1_valores)
                    # Acortar etiquetas largas
                    if len(valor) > 10:
                        etiquetas.append(valor[:8] + "...")
                    else:
                        etiquetas.append(valor)
        
            if datos_boxplot:
                axes[i].boxplot(datos_boxplot, labels=etiquetas)
                axes[i].set_title(f'F1 por {hp.replace("_", " ").title()}')
                axes[i].set_ylabel('F1-Score')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('resultados/graficas/boxplot_hiperparametros.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(" Gráfica de boxplots generada")
        
    except Exception as e:
        print(f" Error en graficar_boxplot_hiperparametros: {e}")

def graficar_correlacion_hiperparametros(resultados, configuraciones):
    """Grafica correlación entre hiperparámetros y F1-score"""
    try:
        # Crear DataFrame para análisis
        datos = []
        for config, res in zip(configuraciones, resultados):
            if res['f1'] > 0:
                datos.append({
                    'neuronas': config['neuronas_ocultas'],
                    'lr': config['lr'],
                    'batch_size': config['batch_size'],
                    'f1': res['f1'],
                    'precision': res['precision'],
                    'recall': res['recall'],
                    'tiempo': res.get('tiempo_ejecucion', 0)
                })
        
        df = pd.DataFrame(datos)
        
        # Gráfica de dispersión: F1 vs Tiempo
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(df['tiempo'], df['f1'], alpha=0.7, s=80)
        plt.xlabel('Tiempo de Ejecución (s)')
        plt.ylabel('F1-Score')
        plt.title('F1-Score vs Tiempo de Ejecución')
        plt.grid(True, alpha=0.3)
        
        # Añadir etiquetas de configuración
        for i, (tiempo, f1) in enumerate(zip(df['tiempo'], df['f1'])):
            plt.annotate(f'{i+1}', (tiempo, f1), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        # Gráfica de dispersión: F1 vs Neuronas
        plt.subplot(1, 2, 2)
        plt.scatter(df['neuronas'], df['f1'], alpha=0.7, s=80)
        plt.xlabel('Neuronas Ocultas')
        plt.ylabel('F1-Score')
        plt.title('F1-Score vs Neuronas Ocultas')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('resultados/graficas/correlacion_hiperparametros.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(" Gráfica de correlación generada")
        
    except Exception as e:
        print(f" Error en graficar_correlacion_hiperparametros: {e}")