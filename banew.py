import customtkinter as ctk
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import random

# Konfigurasi Tema
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class QoSRoutingApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("QoS Routing Optimizer - BSM307")
        self.geometry("1200x800")

        # --- DATA STORAGE ---
        self.G = nx.Graph()
        self.node_data = None
        self.edge_data = None
        self.demand_data = None

        # --- UI LAYOUT ---
        self.grid_columnconfigure(0, weight=1) # Sidebar kiri
        self.grid_columnconfigure(1, weight=4) # Visualisasi Tengah
        self.grid_columnconfigure(2, weight=1) # Sidebar kanan
        self.grid_rowconfigure(0, weight=1)

        self.create_sidebar_left()
        self.create_center_view()
        self.create_sidebar_right()

    def create_sidebar_left(self):
        """Panel Kontrol Parameter Jaringan"""
        self.sidebar_left = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(self.sidebar_left, text="QoS Routing Optimizer", font=("Arial", 18, "bold")).pack(pady=10)

        # Bagian Input Data (Node, Edge, Demand)
        ctk.CTkLabel(self.sidebar_left, text="--- Data Management ---").pack(pady=5)
        ctk.CTkButton(self.sidebar_left, text="Load Node Data", command=self.load_nodes).pack(pady=5)
        ctk.CTkButton(self.sidebar_left, text="Load Edge Data", command=self.load_edges).pack(pady=5)
        ctk.CTkButton(self.sidebar_left, text="Load Demand Data", command=self.load_demands).pack(pady=5)

        # Slider Bobot (W)
        ctk.CTkLabel(self.sidebar_left, text="Ağırlıklar (W)").pack(pady=(20, 0))
        self.w_delay = ctk.CTkSlider(self.sidebar_left, from_=0, to=1)
        self.w_delay.pack(pady=5)
        ctk.CTkLabel(self.sidebar_left, text="Gecikme (Delay)").pack()

        self.w_rel = ctk.CTkSlider(self.sidebar_left, from_=0, to=1)
        self.w_rel.pack(pady=5)
        ctk.CTkLabel(self.sidebar_left, text="Güvenilirlik (Reliability)").pack()

    def create_center_view(self):
        """Visualisasi Graf Utama"""
        self.center_frame = ctk.CTkFrame(self, corner_radius=15)
        self.center_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(6, 6), facecolor='#1a1a1a')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.center_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_sidebar_right(self):
        """Panel Hasil dan Statistik"""
        self.sidebar_right = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_right.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(self.sidebar_right, text="Sonuçlar (Results)", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Penampil Metrik
        self.res_delay = ctk.CTkLabel(self.sidebar_right, text="Total Gecikme: -- ms")
        self.res_delay.pack(pady=10)
        
        self.res_rel = ctk.CTkLabel(self.sidebar_right, text="Güvenilirlik: -- %")
        self.res_rel.pack(pady=10)

        ctk.CTkButton(self.sidebar_right, text="Optimize Et", fg_color="purple", command=self.run_optimization).pack(pady=20)

    # --- LOGIKA DATA ---
    def load_nodes(self):
        # Implementasi pandas read_csv di sini
        print("Loading Node Data...")
        # self.node_data = pd.read_csv('nodes.csv')

    def load_edges(self):
        # Membangun graf dari edge data [cite: 62]
        print("Loading Edge Data...")
        # logic build_network yang kita bahas sebelumnya masuk sini

    def load_demands(self):
        # Memuat skenario permintaan (S, D, B) [cite: 103]
        print("Loading Demand Data...")

    def run_optimization(self):
        """Fungsi Pemicu Algoritma (GA atau RL)"""
        # 1. Ambil bobot dari slider 
        # 2. Jalankan Genetic Algorithm solve() [cite: 80]
        # 3. Update grafik dan label hasil 
        self.draw_graph()

    def draw_graph(self):
        self.ax.clear()
        # Gunakan NetworkX untuk menggambar graf 250 node [cite: 24]
        self.ax.set_facecolor('#1a1a1a')
        self.canvas.draw()

if __name__ == "__main__":
    app = QoSRoutingApp()
    app.mainloop()