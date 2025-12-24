import tkinter as tk
from tkinter import ttk, messagebox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import math

# --- KONFIGURASI ---
NODE_COUNT_DEMO = 30  # Jumlah node untuk demo (biar tidak lag). Untuk laporan ubah jadi 250.
CONNECTION_PROB = 0.3 # Peluang koneksi antar node

class QoS_Routing_Interface:
    def _init_(self, root):
        self.root = root
        self.root.title("QoS Odaklı Rotalama - BSM307 Proje (Interactive)")
        self.root.geometry("1300x850")

        # Variabel Data
        self.G = None
        self.pos = None
        self.path_result = None

        # --- Layout Utama ---
        self.left_frame = ttk.Frame(root, padding="10", width=300)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.right_frame = ttk.Frame(root, padding="10")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Inisialisasi Komponen
        self._init_controls()
        self._init_graph_canvas()

    def _init_controls(self):
        """Membuat Panel Kontrol di Kiri"""
        ttk.Label(self.left_frame, text="Kontrol Parameter", font=("Arial", 14, "bold")).pack(pady=10)

        # 1. Input Bobot
        weights_frame = ttk.LabelFrame(self.left_frame, text="Bobot (Total=1.0)")
        weights_frame.pack(fill=tk.X, pady=5)
        
        self.entries = {}
        for idx, (label, default) in enumerate([("W_Delay", "0.4"), ("W_Reliability", "0.4"), ("W_Resource", "0.2")]):
            ttk.Label(weights_frame, text=label).grid(row=idx, column=0, padx=5, pady=2, sticky="w")
            entry = ttk.Entry(weights_frame, width=8)
            entry.insert(0, default)
            entry.grid(row=idx, column=1, padx=5, pady=2)
            self.entries[label] = entry

        # 2. Pilihan Source & Destination
        sd_frame = ttk.LabelFrame(self.left_frame, text="Pilih Node (Klik di Graf)")
        sd_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(sd_frame, text="Source (S):").grid(row=0, column=0, padx=5)
        self.source_var = tk.StringVar()
        self.source_combo = ttk.Combobox(sd_frame, textvariable=self.source_var, width=12, state="readonly")
        self.source_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(sd_frame, text="Dest (D):").grid(row=1, column=0, padx=5)
        self.dest_var = tk.StringVar()
        self.dest_combo = ttk.Combobox(sd_frame, textvariable=self.dest_var, width=12, state="readonly")
        self.dest_combo.grid(row=1, column=1, padx=5, pady=5)

        # 3. Tombol
        btn_frame = ttk.Frame(self.left_frame)
        btn_frame.pack(pady=20, fill=tk.X)
        
        ttk.Button(btn_frame, text="1. Buat Jaringan Baru", command=self.generate_network).pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="2. Hitung & Animasi Jalur", command=self.run_optimization).pack(fill=tk.X, pady=5)

        # 4. Info Hasil
        self.result_frame = ttk.LabelFrame(self.left_frame, text="Hasil Perhitungan")
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.result_label = tk.Label(self.result_frame, text="Belum ada data.", justify=tk.LEFT, anchor="nw", wraplength=250)
        self.result_label.pack(fill=tk.BOTH, padx=5, pady=5)

    def _init_graph_canvas(self):
        """Area Matplotlib untuk menggambar"""
        self.figure, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- INTERAKTIVITAS (Klik & Hover) ---
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)

        # Tooltip box (Awalnya sembunyi)
        self.annot = self.ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="#ffffe0", alpha=0.9),
                            arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)

    def generate_network(self):
        """Membuat Graf Random Erdos-Renyi"""
        # Bersihkan graf lama
        self.ax.clear()
        self.result_label.config(text="Jaringan baru dibuat.")
        
        # Buat Struktur Graf
        self.G = nx.erdos_renyi_graph(n=NODE_COUNT_DEMO, p=CONNECTION_PROB)
        while not nx.is_connected(self.G): # Ulangi jika graf terputus
             self.G = nx.erdos_renyi_graph(n=NODE_COUNT_DEMO, p=CONNECTION_PROB)

        # Isi Atribut Node (Processing Delay, Reliability)
        for node in self.G.nodes():
            self.G.nodes[node]['proc_delay'] = random.uniform(0.5, 2.0)
            self.G.nodes[node]['reliability'] = random.uniform(0.95, 0.999)

        # Isi Atribut Edge (Bandwidth, Delay, Reliability)
        for u, v in self.G.edges():
            self.G.edges[u, v]['bandwidth'] = random.uniform(100, 1000)
            self.G.edges[u, v]['link_delay'] = random.uniform(3, 15)
            self.G.edges[u, v]['reliability'] = random.uniform(0.95, 0.999)

        # Layout Posisi Node (Disimpan agar tidak berubah saat animasi)
        self.pos = nx.spring_layout(self.G, seed=42) 

        # Update Dropdown GUI
        nodes = list(self.G.nodes())
        self.source_combo['values'] = nodes
        self.dest_combo['values'] = nodes
        self.source_combo.set('')
        self.dest_combo.set('')

        # Gambar Awal
        self.draw_graph()
        messagebox.showinfo("Sukses", f"Jaringan {NODE_COUNT_DEMO} Node Terbentuk!")

    def calculate_cost(self, u, v, w_d, w_r, w_bw):
        """Rumus Biaya Gabungan (Weighted Sum)"""
        e_data = self.G.edges[u, v]
        n_data = self.G.nodes[v]

        # 1. Delay (Link + Processing)
        delay = e_data['link_delay'] + n_data['proc_delay']
        # 2. Reliability Cost (-log agar jadi penjumlahan)
        rel_cost = -math.log(e_data['reliability']) + -math.log(n_data['reliability'])
        # 3. Resource (Inverse Bandwidth)
        res_cost = 1000.0 / e_data['bandwidth']

        return (w_d * delay) + (w_r * rel_cost) + (w_bw * res_cost)

    def run_optimization(self):
        """Menjalankan Algoritma & Memulai Animasi"""
        if self.G is None:
            messagebox.showerror("Error", "Buat Jaringan Dulu!")
            return
        
        try:
            # Ambil data dari GUI
            s = int(self.source_combo.get())
            d = int(self.dest_combo.get())
            wd = float(self.entries["W_Delay"].get())
            wr = float(self.entries["W_Reliability"].get())
            wb = float(self.entries["W_Resource"].get())
            
            # Normalisasi Bobot
            total = wd + wr + wb
            wd, wr, wb = wd/total, wr/total, wb/total

            # --- ALGORITMA UTAMA (Disini nanti tempat GA/RL) ---
            # Demo menggunakan Dijkstra dengan Custom Weight
            path = nx.dijkstra_path(
                self.G, source=s, target=d,
                weight=lambda u, v, attr: self.calculate_cost(u, v, wd, wr, wb)
            )

            # Tampilkan Angka
            self.show_metrics(path)
            
            # Jalankan Animasi
            self.animate_path(path)

        except ValueError:
            messagebox.showerror("Error", "Pilih Source/Dest dan pastikan bobot angka valid.")
        except nx.NetworkXNoPath:
            messagebox.showerror("Info", "Tidak ada jalur yang menghubungkan dua node ini.")

    def show_metrics(self, path):
        """Menghitung total metrik untuk laporan"""
        t_delay, t_rel, t_res = 0, 1.0, 0
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            ed = self.G.edges[u, v]
            nd = self.G.nodes[v]
            
            t_delay += ed['link_delay'] + nd['proc_delay']
            t_rel *= (ed['reliability'] * nd['reliability'])
            t_res += (1000.0 / ed['bandwidth'])
            
        # Tambahkan Source Node processing
        t_delay += self.G.nodes[path[0]]['proc_delay']
        t_rel *= self.G.nodes[path[0]]['reliability']

        txt = (f"Jalur: {path}\n\n"
               f"Total Delay: {t_delay:.2f} ms\n"
               f"Total Reliability: {t_rel:.5f}\n"
               f"Resource Cost: {t_res:.2f}")
        self.result_label.config(text=txt)

    # --- BAGIAN VISUAL & INTERAKTIF ---

    def draw_graph(self, highlight_path=None, highlighted_nodes=None):
        """Fungsi menggambar standar"""
        self.ax.clear()
        
        # Gambar Dasar
        nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, node_size=300, node_color='#d3d3d3')
        nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, edge_color='#999999', alpha=0.4)
        nx.draw_networkx_labels(self.G, self.pos, ax=self.ax, font_size=8)

        # Gambar Jalur (Jika ada)
        if highlight_path:
            # Highlight Node
            nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, nodelist=highlighted_nodes, node_color='#ff8c00', node_size=350)
            # Highlight Edge
            path_edges = list(zip(highlight_path, highlight_path[1:]))
            nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, edgelist=path_edges, edge_color='#ff0000', width=2.5)

        self.ax.set_title("Visualisasi Jaringan (Klik Node untuk Memilih)")
        self.ax.axis('off')
        self.canvas.draw()

    def animate_path(self, path):
        """Animasi langkah demi langkah"""
        # Reset dulu
        self.draw_graph()
        
        path_edges = list(zip(path, path[1:]))
        accumulated_nodes = [path[0]]
        accumulated_edges = []

        # Fungsi Loop Animasi
        def step(i):
            if i >= len(path_edges): return
            
            u, v = path_edges[i]
            accumulated_nodes.append(v)
            accumulated_edges.append((u, v))
            
            # Gambar parsial
            nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, nodelist=[u, v], node_color='#ff8c00', node_size=350)
            nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, edgelist=[(u, v)], edge_color='#ff0000', width=2.5)
            self.canvas.draw()
            
            # Jadwalkan langkah berikutnya (kecepatan 400ms)
            self.root.after(400, lambda: step(i+1))

        step(0)

    def on_click(self, event):
        """Menangani Klik Mouse untuk memilih Node"""
        if event.inaxes != self.ax or self.pos is None: return

        # Cari node terdekat
        closest_node = None
        min_dist = float('inf')
        for node, (x, y) in self.pos.items():
            dist = (x - event.xdata)*2 + (y - event.ydata)*2
            if dist < min_dist:
                min_dist = dist
                closest_node = node

        # Jika klik cukup dekat dengan node
        if min_dist < 0.005:
            # Logika: Jika Source kosong -> isi Source. Jika tidak -> isi Dest.
            if not self.source_combo.get():
                self.source_combo.set(closest_node)
            else:
                self.dest_combo.set(closest_node)
            
            # Beri feedback visual (print di console atau update label kecil)
            print(f"Node dipilih: {closest_node}")

    def on_hover(self, event):
        """Menampilkan Tooltip saat mouse bergerak"""
        if event.inaxes != self.ax or self.pos is None: return

        vis = self.annot.get_visible()
        found = False
        
        for node, (x, y) in self.pos.items():
            dist = (x - event.xdata)*2 + (y - event.ydata)*2
            if dist < 0.005: # Mouse diatas node
                node_data = self.G.nodes[node]
                text = (f"Node {node}\n"
                        f"Proc Delay: {node_data['proc_delay']:.2f}ms\n"
                        f"Rel: {node_data['reliability']:.4f}")
                
                self.annot.xy = (x, y)
                self.annot.set_text(text)
                self.annot.set_visible(True)
                self.canvas.draw_idle()
                found = True
                break
        
        if not found and vis:
            self.annot.set_visible(False)
            self.canvas.draw_idle()

if __name__ == "_main_":
    root = tk.Tk()
    app = QoS_Routing_Interface(root)
    root.mainloop()