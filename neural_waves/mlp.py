import numpy as np

# Função a ser aproximada
def f(x):
    return np.sin(x / 2) * np.cos(2 * x)

# Normalização dos dados
def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# Classe da MLP
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.learning_rate = learning_rate

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self, x):
        self.hidden_input = np.dot(x, self.weights_input_hidden)
        self.hidden_output = self.tanh(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output)
        return self.output

    def train(self, x, y, max_epochs, tolerance):
        errors = []
        for epoch in range(max_epochs):
            # Forward propagation
            output = self.forward(x)

            # Cálculo do erro
            error = y - output
            current_error = np.mean(error ** 2)
            errors.append(current_error)

            # Critério de parada
            if current_error < tolerance:
                break

            # Backpropagation
            d_output = error * self.tanh_derivative(output)
            error_hidden = d_output.dot(self.weights_hidden_output.T)
            d_hidden = error_hidden * self.tanh_derivative(self.hidden_output)

            # Atualização dos pesos
            self.weights_hidden_output += self.hidden_output.T.dot(d_output) * self.learning_rate
            self.weights_input_hidden += x.T.dot(d_hidden) * self.learning_rate

        return errors, epoch + 1

# Função para gerar dados de treinamento
def generate_data(x_min, x_max, num_samples):
    x = np.linspace(x_min, x_max, num_samples).reshape(-1, 1)
    y = f(x)
    x = normalize(x)  # Normalização dos dados
    return x, y