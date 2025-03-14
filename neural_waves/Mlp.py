import numpy as np
import matplotlib.pyplot as plt

class Mlp:
    def __init__(self, x_min=-1, x_max=3, error_threshold=0.01, learning_rate=0.001, neurons_hidden_layer=50, sample_size=100):
        self.x_min = x_min
        self.x_max = x_max
        self.error_threshold = error_threshold
        self.learning_rate = learning_rate
        self.neurons_hidden_layer = neurons_hidden_layer
        self.sample_size = sample_size
        
        self.x = np.linspace(self.x_min, self.x_max, self.sample_size)
        self.y = np.sin(self.x / 2) * np.cos(2 * self.x)
        
        self.weights_input_hidden = np.random.rand(self.neurons_hidden_layer, 1)
        self.bias_hidden = np.random.rand(self.neurons_hidden_layer, 1)
        self.weights_hidden_output = np.random.rand(1, self.neurons_hidden_layer)
        self.bias_output = np.random.rand(1, 1)
        
        self.error = 1
        self.iterations = 0
        
        self.error_history = []
        self.iterations_history = []
        
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def train(self):
        while self.error > self.error_threshold:
            self.iterations += 1
            self.error = 0
            
            for i in range(self.sample_size):
                # Forward pass
                x_val = self.x[i]
                y_val = self.y[i]
                
                x_input = np.array([[x_val]])
                y_true = np.array([[y_val]])
                
                z_hidden = np.dot(self.weights_input_hidden, x_input) + self.bias_hidden
                a_hidden = self.tanh(z_hidden)
                
                z_output = np.dot(self.weights_hidden_output, a_hidden) + self.bias_output
                a_output = self.tanh(z_output)
                
                # Cálculo do erro
                self.error += 0.5 * np.power(y_true - a_output, 2)
                
                # Backward pass
                delta_output = (y_true - a_output) * self.tanh_derivative(z_output)
                delta_hidden = np.dot(self.weights_hidden_output.T, delta_output) * self.tanh_derivative(z_hidden)
                
                self.weights_hidden_output += self.learning_rate * np.dot(delta_output, a_hidden.T)
                self.bias_output += self.learning_rate * delta_output
                
                self.weights_input_hidden += self.learning_rate * np.dot(delta_hidden, x_input.T)
                self.bias_hidden += self.learning_rate * delta_hidden
                
            self.error_history.append(self.error.item())
            self.iterations_history.append(self.iterations)
            
            if self.iterations % 25 == 0:
                print(f'Error: {self.error}, Iterations: {self.iterations}')
        
    def predict(self, x):
        x = np.array(x).reshape(1, -1)  # Formata x para shape (1, n)
        
        z_hidden = np.dot(self.weights_input_hidden, x) + self.bias_hidden  # (neurons_hidden_layer, n)
        a_hidden = self.tanh(z_hidden)
        
        z_output = np.dot(self.weights_hidden_output, a_hidden) + self.bias_output  # (1, n)
        a_output = self.tanh(z_output)
        
        return a_output.flatten()  # Retorna um array 1D
        
    def plot(self):
        y_pred = self.predict(self.x)
        
        plt.figure(figsize=(10, 6))
        
        # 1º gráfico: Erro de treinamento vs Iterações (linha 1, coluna 1)
        plt.subplot(2, 2, 1)
        plt.plot(self.iterations_history, self.error_history, label='Erro', color='red')
        plt.xlabel('Iterações')
        plt.ylabel('Erro')
        plt.title('Erro de Treinamento')
        plt.legend()
        
        # 2º gráfico: Função Original (linha 1, coluna 2)
        plt.subplot(2, 2, 2)
        plt.plot(self.x, self.y, label='Função Original', color='blue')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Função Original')
        plt.legend()
        
        # 3º gráfico: Função Aproximada (linha 2, coluna 1)
        plt.subplot(2, 2, 3)
        plt.plot(self.x, y_pred, label='Função Aproximada', color='green')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Função Aproximada')
        plt.legend()
        
        # 4º gráfico: Função Original e Aproximada sobrepostas (linha 2, coluna 2)
        plt.subplot(2, 2, 4)
        plt.plot(self.x, self.y, label='Função Original', color='blue')
        plt.plot(self.x, y_pred, label='Função Aproximada', linestyle='--', color='green')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Original vs Aproximada')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Instancia e treina o MLP
mlp = Mlp()
mlp.train()
mlp.plot()
