from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
from concurrent.futures import ThreadPoolExecutor
from .Mlp import Mlp
import os

class Interface:
    def __init__(self, root):
        """
        Inicializa a interface:
          - Cria instância da MLP.
          - Configura a janela principal.
          - Cria os frames, inputs, gráficos e labels de status.
          - Prepara o executor para executar o treinamento em background.
        """
        self.mlp = Mlp()
        self.root = root
        self.root.title("Neural Waves")
        self.root.geometry("1200x850")
        self.root.configure(bg="#2E2E2E")

        # Garante que ao fechar a janela, sejam finalizados os processos
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.create_frames()
        self.create_inputs()
        self.set_up_graphs()

        # Executor para rodar o treinamento sem bloquear a interface
        self.executor = ThreadPoolExecutor(max_workers=1)

    def create_frames(self):
        """
        Cria os quatro frames onde serão exibidos os inputs e os gráficos.
        """
        self.frame_top_left = Frame(self.root, bg="#2E2E2E", width=300, height=250)
        self.frame_top_right = Frame(self.root, bg="#2E2E2E", width=300, height=250)
        self.frame_bottom_left = Frame(self.root, bg="#2E2E2E", width=300, height=250)
        self.frame_bottom_right = Frame(self.root, bg="#2E2E2E", width=300, height=250)

        # Organiza os frames utilizando grid
        self.frame_top_left.grid(row=0, column=0, padx=5, pady=5, sticky="nw")
        self.frame_top_right.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.frame_bottom_left.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.frame_bottom_right.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

    def create_inputs(self):
        """
        Cria os inputs (Entry) para configuração dos parâmetros e
        os labels de status que exibem o erro atual e a iteração do treinamento.
        """
        # Título da aplicação
        Label(self.frame_top_left, text="Neural Waves", bg="#2E2E2E", fg="white", font=('Verdana', 16, 'bold'))\
            .grid(row=0, column=1, columnspan=2, pady=10)

        # Frame para inputs
        input_frame = Frame(self.frame_top_left, bg="#2E2E2E")
        input_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=10)

        # Lista de campos: label e o nome da variável associada
        fields = [
            ("X min:", "x_min_entry"),
            ("X max:", "x_max_entry"),
            ("Error Threshold:", "error_threshold_entry"),
            ("Learning Rate:", "learning_rate_entry"),
            ("Neurons:", "neurons_entry"),
            ("Sample Size:", "sample_size_entry"),
        ]

        # Dicionário para armazenar os inputs
        self.entries = {}

        # Cria os labels e entradas dinamicamente
        for i, (label_text, var_name) in enumerate(fields):
            Label(input_frame, text=label_text, bg="#2E2E2E", fg="white", font=('Verdana', 12))\
                .grid(row=i, column=0, sticky="w", padx=5, pady=5)
            entry = Entry(input_frame, bg="#555555", fg="white", insertbackground="white", font=('Verdana', 12))
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries[var_name] = entry

        # Botão para iniciar o treinamento
        self.train_button = Button(self.frame_top_left, text="Train", bg="#555555", fg="white",
                                   font=('Verdana', 12, 'bold'), command=self.on_train)
        self.train_button.grid(row=2, column=1, columnspan=2, pady=10)

        # Frame para os labels de status (erro e iteração)
        status_frame = Frame(self.frame_top_left, bg="#2E2E2E")
        status_frame.grid(row=3, column=0, columnspan=2, padx=20, pady=10)

        # Label para exibir o valor atual do erro
        self.error_label = Label(status_frame, text="Error: N/A", bg="#2E2E2E", fg="white", font=('Verdana', 12))
        self.error_label.grid(row=0, column=0, padx=5, pady=5)

        # Label para exibir a iteração atual
        self.iteration_label = Label(status_frame, text="Iterations: 0", bg="#2E2E2E", fg="white", font=('Verdana', 12))
        self.iteration_label.grid(row=0, column=1, padx=5, pady=5)

    def set_up_graphs(self):
        """
        Configura os gráficos onde serão exibidos:
          - Função original.
          - Aproximação (função aprendida).
          - Evolução do erro.
        Cada gráfico é criado e armazenado em um dicionário.
        """
        self.graphs = {}
        self.graphs["original"] = self.create_graph(self.frame_top_right, "Original Function")
        self.graphs["approximation"] = self.create_graph(self.frame_bottom_left, "Aproximated Function")
        self.graphs["error"] = self.create_graph(self.frame_bottom_right, "Error")

    def create_graph(self, frame, title):
        """
        Cria um gráfico vazio com um título definido e configurações de cores.
        Usa o Matplotlib integrado com Tkinter via FigureCanvasTkAgg.
        """
        x = np.linspace(0, 10, 100)
        y = np.zeros(100)

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#2E2E2E")
        ax.set_facecolor("#2E2E2E")
        ax.plot(x, y, color="cyan", label="Original Function")
        ax.set_title(title, color="white", fontsize=12)
        ax.tick_params(colors='white')
        ax.legend(facecolor="#2E2E2E", edgecolor="white", labelcolor="white")

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        canvas.draw()

        return canvas

    def update_graph(self, graph_key, x, y, x2=None, y2=None, color2="orange", label2="Approximation"):
        """
        Atualiza o gráfico identificado por graph_key.
          - x, y: dados da função original.
          - x2, y2: dados da função aproximada (opcional).
          - color2 e label2 definem o estilo da linha para a aproximação.
        """
        canvas = self.graphs[graph_key]
        fig = canvas.figure
        ax = fig.get_axes()[0]
        ax.clear()
        ax.plot(x, y, color="cyan", label="Original Function")
        if x2 is not None and y2 is not None:
            ax.plot(x2, y2, color=color2, linestyle="dashed", label=label2)
        ax.set_title(graph_key.replace("_", " ").title(), color="white", fontsize=12)
        ax.tick_params(colors='white')
        ax.legend(facecolor="#2E2E2E", edgecolor="white", labelcolor="white")
        canvas.draw()

    # Métodos auxiliares para obter os valores dos inputs
    def get_x_min(self): return float(self.entries["x_min_entry"].get())
    def get_x_max(self): return float(self.entries["x_max_entry"].get())
    def get_error_threshold(self): return float(self.entries["error_threshold_entry"].get())
    def get_learning_rate(self): return float(self.entries["learning_rate_entry"].get())
    def get_neurons(self): return int(self.entries["neurons_entry"].get())
    def get_sample_size(self): return int(self.entries["sample_size_entry"].get())

    def on_train(self):
        """
        Método acionado ao clicar no botão 'Train'.
        Atualiza os parâmetros da MLP com os valores dos inputs,
        reinicia a visualização e dispara o treinamento em uma thread separada.
        Também inicia o loop para atualização dinâmica dos gráficos e status.
        """
        # Atualiza os parâmetros da MLP
        self.mlp.x_min = self.get_x_min()
        self.mlp.x_max = self.get_x_max()
        self.mlp.error_threshold = self.get_error_threshold()
        self.mlp.learning_rate = self.get_learning_rate()
        self.mlp.neurons_hidden_layer = self.get_neurons()  # Atenção: usamos neurons_hidden_layer na classe Mlp
        self.mlp.sample_size = self.get_sample_size()
        self.mlp.x = np.linspace(self.mlp.x_min, self.mlp.x_max, self.mlp.sample_size)
        self.mlp.y = np.sin(self.mlp.x / 2) * np.cos(2 * self.mlp.x)

        # Atualiza o gráfico da função original
        self.update_graph("original", self.mlp.x, self.mlp.y)

        # Reinicia a flag de treinamento
        self.mlp.training_complete = False

        # Inicia o treinamento em segundo plano
        self.executor.submit(self.run_training)

        # Inicia o loop que atualiza os gráficos e os labels de status
        self.update_dynamic_graphs_loop()

    def run_training(self):
        """Executa o método de treinamento da MLP."""
        self.mlp.train()

    def update_dynamic_graphs_loop(self):
        """
        Atualiza dinamicamente os gráficos de aproximação e erro,
        além de atualizar as labels de status (erro e iteração).
        Enquanto o treinamento não estiver completo, reagenda essa função a cada 100 ms.
        """
        # Calcula as predições para todos os pontos de entrada (predict opera em escalares)
        y_pred = np.array([self.mlp.predict(val) for val in self.mlp.x])
        self.update_graph("approximation", self.mlp.x, self.mlp.y, self.mlp.x, y_pred)
        
        # Para o gráfico de erro, usamos o menor tamanho entre os históricos para evitar erro de dimensão
        min_len = min(len(self.mlp.iterations_history), len(self.mlp.error_history))
        self.update_graph("error",
                          np.array(self.mlp.iterations_history[:min_len]),
                          np.array(self.mlp.error_history[:min_len]))
        
        # Atualiza os labels de status se houver registros
        if self.mlp.error_history and self.mlp.iterations_history:
            current_error = self.mlp.error_history[-1]
            current_iteration = self.mlp.iterations_history[-1]
            self.error_label.config(text=f"Error: {current_error:.6f}")
            self.iteration_label.config(text=f"Iterations: {current_iteration}")

        # Se o treinamento ainda não terminou, agenda nova atualização daqui a 100ms
        if not self.mlp.training_complete:
            self.root.after(100, self.update_dynamic_graphs_loop)

    def on_close(self):
        """
        Método chamado ao fechar a janela.
        Garante que o executor seja desligado e que a aplicação seja encerrada.
        Para forçar o encerramento total (inclusive threads em segundo plano), usa os._exit(0).
        """
        self.executor.shutdown(wait=False)
        self.root.quit()
        self.root.destroy()
        os._exit(0)