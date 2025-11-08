import cupy as cp
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def entrenar_mlp_gpu(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, lr=0.01):
    """Versión optimizada para GPU"""
    train_losses = []
    test_losses = []

    # Convertir datos a GPU una sola vez
    X_train_gpu = cp.asarray(X_train, dtype=cp.float32)
    y_train_gpu = cp.asarray(y_train, dtype=cp.float32)
    X_test_gpu = cp.asarray(X_test, dtype=cp.float32)
    y_test_gpu = cp.asarray(y_test, dtype=cp.float32)

    for epoch in range(epochs):
        # Mini-batch en GPU
        indices = cp.random.permutation(X_train_gpu.shape[0])
        X_shuffled = X_train_gpu[indices]
        y_shuffled = y_train_gpu[indices]

        for i in range(0, X_train_gpu.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            output = model.forward(X_batch)
            model.backward(X_batch, y_batch, output, lr)

        # Calcular pérdidas
        train_pred = model.forward(X_train_gpu)
        train_loss = model.compute_loss(y_train_gpu, train_pred)
        train_losses.append(float(cp.asnumpy(train_loss)))

        test_pred = model.forward(X_test_gpu)
        test_loss = model.compute_loss(y_test_gpu, test_pred)
        test_losses.append(float(cp.asnumpy(test_loss)))

        if epoch % 50 == 0:
            print(f"Época {epoch}: Train Loss = {train_losses[-1]:.4f}, Test Loss = {test_losses[-1]:.4f}")

    return train_losses, test_losses

def evaluar_modelo_gpu(model, X, y):
    """Evaluación optimizada para GPU"""
    X_gpu = cp.asarray(X, dtype=cp.float32)
    preds = model.forward(X_gpu)
    preds_cpu = cp.asnumpy(preds)
    preds_bin = (preds_cpu > 0.5).astype(int).flatten()
    
    precision = precision_score(y, preds_bin, zero_division=0)
    recall = recall_score(y, preds_bin, zero_division=0)
    f1 = f1_score(y, preds_bin, zero_division=0)
    return precision, recall, f1