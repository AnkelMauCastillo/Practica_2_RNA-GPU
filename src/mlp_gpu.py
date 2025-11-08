import cupy as cp

class MLP_GPU:
    def __init__(self, input_size, hidden_size, output_size, inicializacion='xavier'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicialización en GPU
        if inicializacion == 'xavier':
            self.W1 = cp.random.randn(input_size, hidden_size).astype(cp.float32) * cp.sqrt(1. / input_size)
            self.W2 = cp.random.randn(hidden_size, output_size).astype(cp.float32) * cp.sqrt(1. / hidden_size)
        elif inicializacion == 'normal':
            self.W1 = cp.random.randn(input_size, hidden_size).astype(cp.float32) * 0.01
            self.W2 = cp.random.randn(hidden_size, output_size).astype(cp.float32) * 0.01

        self.b1 = cp.zeros((1, hidden_size), dtype=cp.float32)
        self.b2 = cp.zeros((1, output_size), dtype=cp.float32)

    def sigmoid(self, x):
        return 1 / (1 + cp.exp(-cp.clip(x, -250, 250)))

    def forward(self, X):
        # Convertir a GPU si es necesario
        if not isinstance(X, cp.ndarray):
            X = cp.asarray(X, dtype=cp.float32)
            
        self.z1 = cp.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = cp.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output, lr):
        m = X.shape[0]

        # Gradientes en GPU
        dZ2 = output - y
        dW2 = (1/m) * cp.dot(self.a1.T, dZ2)
        db2 = (1/m) * cp.sum(dZ2, axis=0, keepdims=True)

        dA1 = cp.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.a1 * (1 - self.a1)
        dW1 = (1/m) * cp.dot(X.T, dZ1)
        db1 = (1/m) * cp.sum(dZ1, axis=0, keepdims=True)

        # Actualización en GPU
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def compute_loss(self, y_true, y_pred):
        return cp.mean((y_true - y_pred) ** 2)