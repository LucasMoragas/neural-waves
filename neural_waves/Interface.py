from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys 

class Interface:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Waves")
        self.root.geometry("1200x850")
        self.root.configure(bg="#2E2E2E")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.create_frames()
        self.create_inputs()
        self.create_mock_graphs()

    def create_frames(self):
        self.frame_top_left = Frame(self.root, bg="#2E2E2E", width=300, height=250)
        self.frame_top_right = Frame(self.root, bg="#2E2E2E", width=300, height=250)
        self.frame_bottom_left = Frame(self.root, bg="#2E2E2E", width=300, height=250)
        self.frame_bottom_right = Frame(self.root, bg="#2E2E2E", width=300, height=250)

        self.frame_top_left.grid(row=0, column=0, padx=5, pady=5, sticky="nw")
        self.frame_top_right.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.frame_bottom_left.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.frame_bottom_right.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

    def create_inputs(self):
        Label(self.frame_top_left, text="Neural Waves", bg="#2E2E2E", fg="white", font=('Verdana', 16, 'bold')).grid(row=0, column=1, columnspan=2, pady=10)

        input_frame = Frame(self.frame_top_left, bg="#2E2E2E")
        input_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=10)

        fields = [
            ("X min:", "x_min_entry"),
            ("X max:", "x_max_entry"),
            ("Error Threshold:", "error_threshold_entry"),
            ("Learning Rate:", "learning_rate_entry"),
            ("Neurons:", "neurons_entry"),
            ("Sample Size:", "sample_size_entry"),
        ]

        self.entries = {} 

        for i, (label_text, var_name) in enumerate(fields):
            Label(input_frame, text=label_text, bg="#2E2E2E", fg="white", font=('Verdana', 12)).grid(row=i, column=0, sticky="w", padx=5, pady=5)
            entry = Entry(input_frame, bg="#555555", fg="white", insertbackground="white", font=('Verdana', 12))
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries[var_name] = entry  

        self.train_button = Button(self.frame_top_left, text="Train", bg="#555555", fg="white", font=('Verdana', 12, 'bold'))
        self.train_button.grid(row=2, column=1, columnspan=2, pady=10)

    def create_mock_graphs(self):
        self.create_mock_graph(self.frame_top_right, "Graph 1")
        self.create_mock_graph(self.frame_bottom_left, "Graph 2")
        self.create_mock_graph(self.frame_bottom_right, "Graph 3")

    def create_mock_graph(self, frame, title):
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(scale=0.1, size=x.shape)

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#2E2E2E")
        ax.set_facecolor("#2E2E2E")
        ax.plot(x, y, color="cyan")
        ax.set_title(title, color="white", fontsize=12)
        ax.set_xlabel("X Axis", color="white")
        ax.set_ylabel("Y Axis", color="white")
        ax.tick_params(colors='white')

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        canvas.draw()

    def on_close(self):
        self.root.quit()  
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    root = Tk()
    app = Interface(root)
    root.mainloop()
