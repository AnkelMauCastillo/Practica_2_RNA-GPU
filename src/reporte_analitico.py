# reportes_completos.py
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def crear_directorios():
    """Crea directorios necesarios"""
    os.makedirs('resultados', exist_ok=True)
    os.makedirs('resultados/graficas', exist_ok=True)
    os.makedirs('resultados/modelos', exist_ok=True)

def generar_reporte_completo_gpu(resultados, configuraciones, X_pru, y_pru):
    """Genera reportes completos en TXT sin mostrar gráficas"""
    
    crear_directorios()
    
    # Reporte principal
    with open('resultados/reporte_completo_gpu.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("REPORTE COMPLETO - PRÁCTICA 2 RNA - OPTIMIZADO PARA GPU\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("INFORMACIÓN DEL SISTEMA\n")
        f.write("-" * 50 + "\n")
        f.write(f"Dispositivo usado: {device}\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"Configuraciones evaluadas: {len(configuraciones)}\n\n")
        
        # Top configuraciones
        mejores_indices = np.argsort([r['f1'] for r in resultados])[-5:][::-1]
        
        f.write("TOP 5 CONFIGURACIONES\n")
        f.write("-" * 50 + "\n")
        for i, idx in enumerate(mejores_indices):
            config = configuraciones[idx]
            res = resultados[idx]
            f.write(f"{i+1}. Config {idx+1} - F1: {res['f1']:.4f}\n")
            f.write(f"   Neuronas: {config['neuronas_ocultas']} | ")
            f.write(f"Init: {config['inicializacion']} | ")
            f.write(f"Pesado: {config['pesado_terminos']} | ")
            f.write(f"Ngramas: {config['ngramas']}\n")
            f.write(f"   Preproc: {config['preprocesamiento']} | ")
            f.write(f"LR: {config['lr']} | ")
            f.write(f"Batch: {config['batch_size']} | ")
            f.write(f"Tiempo: {res['tiempo_entrenamiento']:.1f}s\n\n")
        
        # Tabla completa
        f.write("TABLA COMPLETA DE RESULTADOS\n")
        f.write("=" * 120 + "\n")
        f.write(f"{'Config':^6} {'Neuronas':^8} {'Init':^8} {'Pesado':^8} {'Ngramas':^10} {'LR':^6} {'Batch':^6} {'F1':^8} {'Precision':^10} {'Recall':^8} {'Tiempo':^10}\n")
        f.write("-" * 120 + "\n")
        
        for i, (config, res) in enumerate(zip(configuraciones, resultados)):
            f.write(f"{i+1:^6} {config['neuronas_ocultas']:^8} {config['inicializacion']:^8} "
                   f"{config['pesado_terminos']:^8} {str(config['ngramas']):^10} {config['lr']:^6.3f} "
                   f"{config['batch_size']:^6} {res['f1']:^8.4f} {res['precision']:^10.4f} "
                   f"{res['recall']:^8.4f} {res['tiempo_entrenamiento']:^10.1f}\n")
    
    # Análisis comparativo
    generar_analisis_comparativo_gpu(resultados, configuraciones)
    
    # Gráficas de pérdidas
    generar_graficas_perdidas(resultados, configuraciones)
    
    # Análisis de errores
    generar_analisis_errores(resultados, configuraciones, X_pru, y_pru)
    
    print("✅ Reportes completos generados en carpeta 'resultados/'")

def generar_graficas_perdidas(resultados, configuraciones):
    """Genera gráficas de pérdidas sin mostrarlas en pantalla"""
    
    # Top 3 configuraciones
    mejores_indices = np.argsort([r['f1'] for r in resultados])[-3:][::-1]
    
    plt.figure(figsize=(15, 5))
    
    for i, idx in enumerate(mejores_indices):
        res = resultados[idx]
        config = configuraciones[idx]
        
        plt.subplot(1, 3, i+1)
        plt.plot(res['train_losses'], label='Train Loss', linewidth=2)
        plt.plot(res['test_losses'], label='Val Loss', linewidth=2)
        plt.title(f'Config {idx+1}\nF1: {res["f1"]:.3f}', fontsize=12)
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('resultados/graficas/curvas_perdida_top3.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfica comparativa de todas las configuraciones
    plt.figure(figsize=(12, 8))
    for i, (config, res) in enumerate(zip(configuraciones, resultados)):
        if len(res['test_losses']) > 0:
            final_loss = res['test_losses'][-1]
            plt.plot(res['test_losses'], alpha=0.6, 
                    label=f'Config {i+1} (F1: {res["f1"]:.3f})')
    
    plt.title('Curvas de Pérdida de Validación - Todas las Configuraciones')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('resultados/graficas/curvas_perdida_todas.png', dpi=300, bbox_inches='tight')
    plt.close()

def generar_analisis_comparativo_gpu(resultados, configuraciones):
    """Genera análisis comparativo detallado"""
    
    with open('resultados/analisis_comparativo_gpu.txt', 'w', encoding='utf-8') as f:
        f.write("ANÁLISIS COMPARATIVO DETALLADO\n")
        f.write("=" * 80 + "\n\n")
        
        # Análisis por hiperparámetro
        hiperparametros = ['neuronas_ocultas', 'inicializacion', 'pesado_terminos', 'lr', 'batch_size']
        
        for hp in hiperparametros:
            f.write(f"ANÁLISIS POR {hp.upper()}\n")
            f.write("-" * 40 + "\n")
            
            valores_unicos = list(set([config[hp] for config in configuraciones]))
            analisis = {}
            
            for valor in valores_unicos:
                f1_valores = []
                for config, res in zip(configuraciones, resultados):
                    if config[hp] == valor:
                        f1_valores.append(res['f1'])
                
                if f1_valores:
                    analisis[valor] = {
                        'mean': np.mean(f1_valores),
                        'std': np.std(f1_valores),
                        'min': np.min(f1_valores),
                        'max': np.max(f1_valores)
                    }
                    
                    f.write(f"  {valor}: F1 = {analisis[valor]['mean']:.4f} ± {analisis[valor]['std']:.4f} "
                           f"(min: {analisis[valor]['min']:.4f}, max: {analisis[valor]['max']:.4f})\n")
            
            # Mejor valor
            if analisis:
                mejor_valor = max(analisis, key=lambda x: analisis[x]['mean'])
                f.write(f"  MEJOR: {mejor_valor} (F1: {analisis[mejor_valor]['mean']:.4f})\n\n")