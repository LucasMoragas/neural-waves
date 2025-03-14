import numpy as np
import matplotlib.pyplot as plt
from tkinter import *

class Mlp:
    def __init__(self, x_min=-1, x_max=1, error_threshold=0.001, learning_rate=0.001, neurons_hidden_layer=50, sample_size=100):
        """
        Inicializa os parâmetros do modelo:
          - x_min, x_max: Limites do domínio.
          - error_threshold: Erro mínimo para término do treinamento.
          - learning_rate: Taxa de aprendizado.
          - neurons_hidden_layer: Número de neurônios na camada oculta.
          - sample_size: Número de amostras para treinamento.
        """
        self.x_min = x_min
        self.x_max = x_max
        self.error_threshold = error_threshold
        self.learning_rate = learning_rate
        self.neurons_hidden_layer = neurons_hidden_layer
        self.sample_size = sample_size

        # Gera os pontos de entrada e a respectiva função original
        self.x = np.linspace(self.x_min, self.x_max, self.sample_size)
        self.y = np.sin(self.x / 2) * np.cos(2 * self.x)

        # Inicializa os pesos e bias de forma aleatória
        self.weights_input_hidden = np.random.rand(self.neurons_hidden_layer, 1)
        self.bias_hidden = np.random.rand(self.neurons_hidden_layer, 1)
        self.weights_hidden_output = np.random.rand(1, self.neurons_hidden_layer)
        self.bias_output = np.random.rand(1, 1)

        # Variáveis para controlar o erro e o número de iterações
        self.error = 1
        self.iterations = 0

        # Históricos para armazenar a evolução do erro e das iterações
        self.error_history = []
        self.iterations_history = []

        # Flag para sinalizar quando o treinamento estiver concluído
        self.training_complete = False

    def tanh(self, x):
        """Função de ativação tangente hiperbólica."""
        return np.tanh(x)

    def tanh_derivative(self, x):
        """Derivada da tangente hiperbólica."""
        return 1 - np.tanh(x)**2

    def train(self):
        """
        Executa o treinamento do modelo até que o erro seja menor que o limiar especificado.
        Para cada iteração, o modelo realiza o forward pass, calcula o erro e realiza o backpropagation.
        Ao final de cada iteração, os valores de erro e iteração são armazenados nos históricos.
        """
        while self.error > self.error_threshold:
            self.iterations += 1
            self.error = 0

            # Processa cada amostra individualmente
            for i in range(self.sample_size):
                # Seleciona a entrada e a saída desejada para a amostra i
                x_val = self.x[i]
                y_val = self.y[i]

                # Converte os valores em arrays 2D para operações matriciais
                x_arr = np.array([[x_val]])
                y_arr = np.array([[y_val]])

                # Forward pass: calcula a ativação da camada oculta
                z_hidden = np.dot(self.weights_input_hidden, x_arr) + self.bias_hidden
                a_hidden = self.tanh(z_hidden)

                # Forward pass: calcula a saída final
                z_output = np.dot(self.weights_hidden_output, a_hidden) + self.bias_output
                a_output = self.tanh(z_output)

                # Calcula o erro quadrático para a amostra
                self.error += 0.5 * np.power(y_arr - a_output, 2)

                # Backward pass: calcula os deltas para saída e camada oculta
                delta_output = (y_arr - a_output) * self.tanh_derivative(z_output)
                delta_hidden = np.dot(self.weights_hidden_output.T, delta_output) * self.tanh_derivative(z_hidden)

                # Atualiza os pesos e biases com base no gradiente e na taxa de aprendizado
                self.weights_hidden_output += self.learning_rate * np.dot(delta_output, a_hidden.T)
                self.bias_output += self.learning_rate * delta_output
                self.weights_input_hidden += self.learning_rate * np.dot(delta_hidden, x_arr.T)
                self.bias_hidden += self.learning_rate * delta_hidden

            # Armazena os históricos de erro e iterações para visualização
            self.error_history.append(self.error.item())
            self.iterations_history.append(self.iterations)

        # Após o término do treinamento, sinaliza que está completo
        self.training_complete = True

    def predict(self, x):
        """
        Realiza a predição para um valor escalar x utilizando o modelo treinado.
        Retorna um valor escalar como predição.
        """
        x_arr = np.array([[x]])
        z_hidden = np.dot(self.weights_input_hidden, x_arr) + self.bias_hidden
        a_hidden = self.tanh(z_hidden)
        z_output = np.dot(self.weights_hidden_output, a_hidden) + self.bias_output
        a_output = self.tanh(z_output)
        return a_output.item()

    def plot(self):
        """
        Gera três gráficos:
          1. Evolução do erro durante o treinamento.
          2. Função original.
          3. Função aprendida pela MLP.
        """
        plt.figure(figsize=(12, 4))

        # Gráfico 1: Evolução do erro
        plt.subplot(1, 3, 1)
        plt.plot(self.iterations_history, self.error_history, label='Erro')
        plt.xlabel('Iterações')
        plt.ylabel('Erro')
        plt.title('Erro ao longo do treinamento')
        plt.legend()

        # Gráfico 2: Função original
        plt.subplot(1, 3, 2)
        plt.plot(self.x, self.y, label='Função Original', color='blue')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Função Original')
        plt.legend()

        # Gráfico 3: Função aprendida pela MLP
        plt.subplot(1, 3, 3)
        y_pred = np.array([self.predict(val) for val in self.x])
        plt.plot(self.x, self.y, label='Função Original', color='blue')
        plt.plot(self.x, y_pred, label='Função Aprendida', linestyle='dashed', color='red')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Função Aprendida pela MLP')
        plt.legend()

        plt.tight_layout()
        plt.show()