
from .class_segmentation_otsu import Class_Segmention_Otsu
from .class_asymmetry import ClassAsymmetry
from .class_border import ClassBorder
from .class_color import ClassColor
from .class_for_operations_masks import ClassForOperationsMasks
from .class_diameter import ClassDiameter

from .libraries import *

class GUI(tk.Frame, Class_Segmention_Otsu, ClassAsymmetry, ClassBorder, ClassColor, ClassDiameter):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master = master
        self.master.title("Análisis de Imágenes Médicas")
        self.pack(fill=tk.BOTH, expand=True)
        self.bg_color = "#D6EAF8"
        self.border_color = "#2874A6"
        self.button_color = "#2874A6"
        self.text_color = "#0A0A0A"
        self.text_color_bu = "#FFFFFF"
        self.highlight_color = "#0A0A0A"
        self.create_widgets()

    def create_widgets(self):
        
        style = ttk.Style()
        style.configure('TNotebook', tabposition='n')  
        style.configure('TNotebook.Tab', font=('Helvetica', '12', 'bold'), padding=[20, 10])  

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

        self.tab_segmentation = ttk.Frame(self.notebook)
        self.tab_asymmetry = ttk.Frame(self.notebook)
        self.tab_border = ttk.Frame(self.notebook)
        self.tab_color = ttk.Frame(self.notebook)
        self.tab_diameter = ttk.Frame(self.notebook)

        # Añadir las pestañas al Notebook
        self.notebook.add(self.tab_segmentation, text='Segmentación')
        self.notebook.add(self.tab_asymmetry, text='Asimetría')
        self.notebook.add(self.tab_border, text='Bordes')
        self.notebook.add(self.tab_color, text='Color')
        self.notebook.add(self.tab_diameter, text='Diámetro')

        button_frame = tk.Frame(self)
        button_frame.grid(row=0, column=0, padx=5, pady=5, sticky='ew')

        self.load_button = tk.Button(button_frame, text="Cargar Imagen", command=self.load_image_file, font=("Helvetica", 15, "bold"), bg=self.button_color, fg=self.text_color_bu, relief=tk.GROOVE, bd=2)
        self.load_button.grid(row=0, column=0, padx=100, pady=5, sticky='w')

        self.show_graphs_button = tk.Button(button_frame, text="Mostrar Gráficas", command=self.show_graphs, font=("Helvetica", 15, "bold"), bg=self.button_color, fg=self.text_color_bu, relief=tk.GROOVE, bd=2, state=tk.DISABLED)
        self.show_graphs_button.grid(row=0, column=1, padx=100, pady=5, sticky='w')

    def load_image_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.clear_frames()
            Class_Segmention_Otsu.__init__(self, file_path)
            mask4, self.masked_image_4, resized_mask_4 = self.otzu_edges_4()
            self.show_segmentation_graph()
            ClassForOperationsMasks.__init__(self, mask4)

            self.touch_border = False
            self.height, self.width = self.mask.shape
            contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                for point in contour:
                    x, y = point[0][0], point[0][1]
                    if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
                        self.touch_border = True
                    else:
                        self.touch_border = False

            if self.touch_border:
                mask_c = self.mask
            else:
                mask_c = self.center_mask()

            ClassAsymmetry.__init__(self, mask_c)

            self.mask_right, self.mask_left, self.top_mask, self.down_mask = self.divide_images()
            self.union_left, self.asymmetry_left, self.union_top, self.asymmetry_top, self.percentage_asimetry_left, self.percentage_asimetry_down = self.asymmetry_method()
            self.perimeter_right, self.perimeter_left, self.perimeter_top, self.perimeter_down = self.calculate_perimeters()
            self.a1, self.a2, self.a = self.method_2()

            ClassBorder.__init__(self, mask4)
            self.B_score = self.index_borders_functions()

            ClassColor.__init__(self, file_path, resized_mask_4, False)
            self.Color_Score = self.color_score()

            ClassDiameter.__init__(self, file_path, mask4)

            self.show_graphs_button.config(state=tk.NORMAL)

    def show_graphs(self):
        self.clear_frames()
        self.show_segmentation_graph()
        self.show_asymmetry_graph()
        self.show_asymmetry_graph2()
        self.show_border_graph()
        self.show_color_graph()
        self.show_diameter_graph2()
        self.display_results()
        self.show_graphs_button.config(state=tk.DISABLED)

    def clear_frames(self):
        for tab in [self.tab_segmentation, self.tab_asymmetry, self.tab_border, self.tab_color, self.tab_diameter]:
            for widget in tab.winfo_children():
                widget.destroy()

    def display_results(self):
        self.tsd = round(self.a * 1.3 + self.B_score * 0.1 + self.Color_Score * 0.5 + self.score_diameter() * 0.5, 2)
        self.result = ''
        if self.tsd <= 4.75:
            self.result = 'Puntaje <= 4.75: Bajo riesgo'
        elif self.tsd <= 5.45:
            self.result = 'Puntaje <= 5.75: Riesgo medio'
        else:
            self.result = 'Puntaje > 5.75: Alto riesgo'

        results = [
            ('Puntaje Asimetría:', self.a),
            ('Puntaje Borde:', self.B_score),
            ('Puntaje Color:', self.Color_Score),
            ('Puntaje Diámetro:', self.score_diameter()),
            ('TDS Sumatoria:', self.tsd),
            ('Resultado:', self.result + '\n')  
        ]
        results_frame = tk.Frame(self.tab_diameter, bg=self.bg_color)
        results_frame.grid(row=0,column=1)
        for i, (label, value) in enumerate(results):
            tk.Label(results_frame, text=f'{label} {value}', font=("Helvetica", 15), bg=self.bg_color, fg="black").grid(row=i, column=0, padx=5, pady=5, sticky="w")

    def show_segmentation_graph(self):
        # Primer subplot
        plot_segmentation1 = plt.figure(figsize=(4, 4))
        plt.imshow(self.image_color_rgb)
        plt.title('Imagen', fontsize=10, color=self.text_color)
        plt.axis('off')
        plt.tight_layout()
        plt.close(plot_segmentation1)
        self.segmentation_canvas1 = FigureCanvasTkAgg(plot_segmentation1, master=self.tab_segmentation)
        self.segmentation_canvas1.draw()
        self.segmentation_canvas1.get_tk_widget().grid(row=0, column=0, padx=50, pady=50)

        # Segundo subplot
        plot_segmentation2 = plt.figure(figsize=(4, 4))
        plt.imshow(self.masked_image_4, cmap='gray')
        plt.title('Segmentación', fontsize=10, color=self.text_color)
        plt.axis('off')
        plt.tight_layout()
        plt.close(plot_segmentation2)
        self.segmentation_canvas2 = FigureCanvasTkAgg(plot_segmentation2, master=self.tab_segmentation)
        self.segmentation_canvas2.draw()
     
        self.segmentation_canvas2.get_tk_widget().grid(row=0, column=1, padx=50, pady=50)

    def show_asymmetry_graph(self):
        title1 = f'Diferencia de área horizontal $Ax_{{score}}$: {self.percentage_asimetry_left} %'
        title2 = f'Diferencia de área vertical $Ay_{{score}}$: {self.percentage_asimetry_down} %'

        plot_asymmetry = plt.figure(figsize=(4, 4))
        plt.subplot(2, 1, 1)
        plt.imshow(self.union_left, cmap='gray')
        percentage1 = float(title1.split(': ')[1].strip(' %'))
        color1 = 'red' if percentage1 > 19 else 'blue'
        plt.title(title1, color=color1, fontsize=9)
        plt.axis('off')

        plt.subplot(2, 1, 2)
        plt.imshow(self.union_top, cmap='gray')
        percentage2 = float(title2.split(': ')[1].strip(' %'))
        color2 = 'red' if percentage2 > 19 else 'blue'
        plt.title(title2, color=color2, fontsize=9)
        plt.axis('off')

        plt.tight_layout()
        plt.close(plot_asymmetry)
        canvas = FigureCanvasTkAgg(plot_asymmetry, master=self.tab_asymmetry)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, padx=50, pady=50, sticky="e")

    def show_asymmetry_graph2(self):
        plot_asymmetry2 = plt.figure(figsize=(4, 4))

        ax1 = plt.subplot(2, 1, 1)
        plt.imshow(self.rotated_image, cmap='gray')
        plt.axis('off')

        for start_point, end_point in self.segments_top:
            ax1.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='cyan')

        for start_point, end_point in self.segments_down:
            ax1.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='green')

        perimetro_max_1 = max(self.perimeter_down, self.perimeter_top)
        perimetro_max_2 = max(self.perimeter_right, self.perimeter_left)
        pa1 = abs(self.perimeter_down - self.perimeter_top) * 100 / perimetro_max_1
        pa2 = abs(self.perimeter_right - self.perimeter_left) * 100 / perimetro_max_2

        title_color = 'blue' if pa1 < 9 else 'red'
        title_color2 = 'blue' if pa2 < 9 else 'red'
        ax1.set_title(f'Diferencia perímetro Superior-Inferior: {round(pa1, 2)}', color=title_color, fontsize=9)

        ax2 = plt.subplot(2, 1, 2)
        plt.imshow(self.rotated_image, cmap='gray')
        plt.axis('off')

        for start_point, end_point in self.segments_left:
            ax2.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='red')

        for start_point, end_point in self.segments_right:
            ax2.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='yellow')
        ax2.set_title(f'Diferencia perímetro Derecha-Izquierda: {round(pa2, 2)}', color=title_color2, fontsize=9)

        plt.tight_layout()
        plt.close(plot_asymmetry2)
        canvas = FigureCanvasTkAgg(plot_asymmetry2, master=self.tab_asymmetry)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=1, padx=50, pady=50, sticky="e")

    def show_border_graph(self):
        colors = plt.colormaps.get_cmap('tab10')

        fig = plt.figure(figsize=(4, 4))
        for i in range(len(self.contour)):
            j = (i + 1) % len(self.contour)
            segment_index = int(self.angles[i] // 45)
            plt.plot([self.contour[i][0], self.contour[j][0]], [self.contour[i][1], self.contour[j][1]], color=colors(segment_index))

        plt.scatter(self.cx, self.cy, color='red', s=100, zorder=5, label='Centroide')

        plt.title("División en 8 segmentos", fontsize=10, color=self.text_color)
        plt.legend()
        plt.axis('off')
        plt.axis('equal')
        plt.close(fig)

        canvas = FigureCanvasTkAgg(fig, master=self.tab_border)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, padx=300, pady=50)

    def show_color_graph(self):
        freq_colors_plot = plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        for color, count in self.significant_colors:
            plt.bar(color, count, color=np.array(self.reference_colors[color]) / 255, label=f"{color}: {count / self.total_pixels * 100:.2f}%")
        plt.title("Frecuencia de Colores Clasificados", fontsize=10, color=self.text_color)
        plt.xlabel("Color", color=self.text_color)
        plt.ylabel("Frecuencia", color=self.text_color)
        plt.legend()

        plt.subplot(1, 2, 2)
        result_image = Image.fromarray(self.classified_image)
        plt.imshow(result_image)
        plt.title("Imagen Clasificada", fontsize=10, color=self.text_color)
        plt.axis('off')
        plt.tight_layout()
        plt.close(freq_colors_plot)
        canvas = FigureCanvasTkAgg(freq_colors_plot, master=self.tab_color)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, padx=100, pady=50)

    def show_diameter_graph2(self):
        fig_diameter = plt.figure(figsize=(4, 4))
        plt.imshow(self.mask_centering, cmap='gray')
        plt.title("Diámetro", fontsize=10, color=self.text_color)
        plt.plot([self.pt1[0], self.pt2[0]], [self.pt1[1], self.pt2[1]], 'r-', linewidth=2)
        plt.scatter([self.pt1[0], self.pt2[0]], [self.pt1[1], self.pt2[1]], color='green', s=50)
        plt.axis('off')
        plt.close(fig_diameter)

        canvas = FigureCanvasTkAgg(fig_diameter, master=self.tab_diameter)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, padx=100, pady=50)
