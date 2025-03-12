import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mlp import MLP, generate_data

# Interface Gráfica
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("MLP Aproximação")

        # Frame principal
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Frame para os controles
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Frame para o gráfico
        self.graph_frame = tk.Frame(self.main_frame)
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Configuração dos gráficos
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(8, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Variáveis de controle
        self.x_min = tk.DoubleVar(value=-1)
        self.x_max = tk.DoubleVar(value=1)
        self.tolerance = tk.DoubleVar(value=0.05)
        self.learning_rate = tk.DoubleVar(value=0.005)
        self.hidden_neurons = tk.IntVar(value=200)
        self.num_samples = tk.IntVar(value=50)
        self.current_error = tk.StringVar(value="0.0")
        self.current_epoch = tk.StringVar(value="0")

        # Criar widgets de controle
        self.create_controls()

    def create_controls(self):
        # Função
        tk.Label(self.control_frame, text="Função: f(x) = sen(x/2) · cos(2x)").grid(row=0, column=0, columnspan=2, pady=5)

        # x mínimo
        tk.Label(self.control_frame, text="x mínimo:").grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(self.control_frame, textvariable=self.x_min, width=10).grid(row=1, column=1, pady=5)

        # x máximo
        tk.Label(self.control_frame, text="x máximo:").grid(row=2, column=0, sticky="w", pady=5)
        tk.Entry(self.control_frame, textvariable=self.x_max, width=10).grid(row=2, column=1, pady=5)

        # Erro Tolerado
        tk.Label(self.control_frame, text="Erro Tolerado:").grid(row=3, column=0, sticky="w", pady=5)
        tk.Entry(self.control_frame, textvariable=self.tolerance, width=10).grid(row=3, column=1, pady=5)

        # Taxa de Aprendizagem
        tk.Label(self.control_frame, text="Taxa de Aprendizagem:").grid(row=4, column=0, sticky="w", pady=5)
        tk.Entry(self.control_frame, textvariable=self.learning_rate, width=10).grid(row=4, column=1, pady=5)

        # Quantidade de Neurônios
        tk.Label(self.control_frame, text="Quantidade de Neurônios:").grid(row=5, column=0, sticky="w", pady=5)
        tk.Entry(self.control_frame, textvariable=self.hidden_neurons, width=10).grid(row=5, column=1, pady=5)

        # Quantidade de Amostras
        tk.Label(self.control_frame, text="Quantidade de Amostras:").grid(row=6, column=0, sticky="w", pady=5)
        tk.Entry(self.control_frame, textvariable=self.num_samples, width=10).grid(row=6, column=1, pady=5)

        # Valor do Erro Atual
        tk.Label(self.control_frame, text="Valor do Erro Atual:").grid(row=7, column=0, sticky="w", pady=5)
        tk.Label(self.control_frame, textvariable=self.current_error).grid(row=7, column=1, pady=5)

        # Número de Ciclos
        tk.Label(self.control_frame, text="Número de Ciclos:").grid(row=8, column=0, sticky="w", pady=5)
        tk.Label(self.control_frame, textvariable=self.current_epoch).grid(row=8, column=1, pady=5)

        # Botão de Treinar
        tk.Button(self.control_frame, text="Aproximar", command=self.train).grid(row=9, column=0, columnspan=2, pady=10)

    def train(self):
        x_min = self.x_min.get()
        x_max = self.x_max.get()
        tolerance = self.tolerance.get()
        learning_rate = self.learning_rate.get()
        hidden_neurons = self.hidden_neurons.get()
        num_samples = self.num_samples.get()

        x, y = generate_data(x_min, x_max, num_samples)

        mlp = MLP(input_size=1, hidden_size=hidden_neurons, output_size=1, learning_rate=learning_rate)

        errors, epochs = mlp.train(x, y, max_epochs=1000, tolerance=tolerance)

        self.current_error.set(f"{errors[-1]:.6f}")
        self.current_epoch.set(f"{epochs}")

        # Gráfico da Função Aproximada
        self.ax1.clear()
        self.ax1.plot(x, mlp.forward(x), label="Função Aproximada", color='blue')
        self.ax1.legend()
        self.ax1.set_title("Função Aproximada")
        self.ax1.set_xlabel("x")
        self.ax1.set_ylabel("f(x)")

        # Gráfico da Aproximação MLP x Real
        self.ax2.clear()
        self.ax2.plot(x, y, label="Função Real", color='green')
        self.ax2.plot(x, mlp.forward(x), label="Aproximação MLP", color='red')
        self.ax2.legend()
        self.ax2.set_title("Aproximação MLP x Real")
        self.ax2.set_xlabel("x")
        self.ax2.set_ylabel("f(x)")

        # Gráfico do Erro por Ciclo
        self.ax3.clear()
        self.ax3.plot(range(epochs), errors, label="Erro", color='purple')
        self.ax3.legend()
        self.ax3.set_title("Erro por Ciclo")
        self.ax3.set_xlabel("Épocas")
        self.ax3.set_ylabel("Erro")

        self.canvas.draw()

# Execução do programa
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")  # Define o tamanho inicial da janela
    app = App(root)
    root.mainloop()