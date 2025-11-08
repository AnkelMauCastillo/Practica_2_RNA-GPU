# configs_optimized.py
CONFIGURACIONES_GPU = [
    # Configuraciones base variando neuronas
    {'neuronas_ocultas': 64, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 32, 'epochs': 100},
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 32, 'epochs': 100},
    {'neuronas_ocultas': 256, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 32, 'epochs': 100},
    {'neuronas_ocultas': 512, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 32, 'epochs': 100},
    
    # Variar inicialización
    {'neuronas_ocultas': 256, 'inicializacion': 'normal', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 32, 'epochs': 100},
    
    # Variar pesado de términos
    {'neuronas_ocultas': 256, 'inicializacion': 'xavier', 'pesado_terminos': 'tfidf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 32, 'epochs': 100},
    
    # Variar n-gramas
    {'neuronas_ocultas': 256, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,2), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 32, 'epochs': 100},
    {'neuronas_ocultas': 256, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (2,2), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 32, 'epochs': 100},
    
    # Variar learning rate
    {'neuronas_ocultas': 256, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.001, 'batch_size': 32, 'epochs': 100},
    {'neuronas_ocultas': 256, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.1, 'batch_size': 32, 'epochs': 100},
    
    # Variar batch size
    {'neuronas_ocultas': 256, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 16, 'epochs': 100},
    {'neuronas_ocultas': 256, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 64, 'epochs': 100},
]