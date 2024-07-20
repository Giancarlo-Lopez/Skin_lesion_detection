
from .libraries import *

class ClassColor():

    def __init__(self, image_path,mask,show_graph):

        self.image_path=image_path
        self.mask=mask
        self.show_graph=show_graph
        self.image_to_pil()
        self.extract_colors_and_positions()
        self.colors_base()
        self.process_and_classify_pixels()
    
    
    def image_to_pil(self):
    
        imagen = Image.open(self.image_path).convert('RGB')
        transformacion = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        imagen_tensor = transformacion(imagen).to(device)
        image_cpu=imagen_tensor.cpu()
        self.image_numpy=image_cpu.numpy()
        self.image_numpy = self.image_numpy.transpose(1, 2, 0)
        return self.image_numpy
    
    def extract_colors_and_positions(self):

        self.red_values = []
        self.green_values = []
        self.blue_values = []
        self.positions = []
        
        for y in range(self.mask.shape[0]): 
            for x in range(self.mask.shape[1]):
                if self.mask[y, x] == 1 or self.mask[y, x] == 255:  
                    r, g, b = self.image_numpy[y, x]  
                    self.red_values.append(r)
                    self.green_values.append(g)
                    self.blue_values.append(b)
                    self.positions.append((x, y))
        
        return self.red_values, self.green_values, self.blue_values, self.positions
    

    def colors_base(self):

        self.reference_colors = { 

        'cafe_claro': np.array([200, 131, 86]),
        'cafe_oscuro': np.array([160, 95, 58]),
        'rojo': np.array([210, 51, 35]),
        'blanco': np.array([255, 255, 255]),
        'negro': np.array([0, 0, 0]),
        'azul': np.array([74, 134, 137])
        }
 
    
    
    def process_and_classify_pixels(self):
    
        color_umbral=5
        image_size = (224, 224)
        pixels = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        self.classified_image = np.zeros_like(pixels)
        color_frequencies = {color: 0 for color in self.reference_colors} 

        for (r, g, b, pos) in zip(self.red_values, self.green_values, self.blue_values, self.positions):
            pixel = np.array([r * 255, g * 255, b * 255], dtype=np.uint8)
            pixels[pos] = pixel
        
            closest_color = min(self.reference_colors, key=lambda c: np.linalg.norm(pixel - self.reference_colors[c]))
            self.classified_image[pos] = self.reference_colors[closest_color]
            color_frequencies[closest_color] += 1

        self.total_pixels = sum(color_frequencies.values())
        self.significant_colors = [(color, count) for color, count in color_frequencies.items() if (count / self.total_pixels) * 100 >= color_umbral]
        
        result_image = None
        

        if(self.show_graph=='true'):
            freq_colors_plot = plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            for color, count in self.significant_colors:
                plt.bar(color, count, color=np.array(self.reference_colors[color]) / 255, edgecolor='black',label=f"{color}: {count/self.total_pixels*100:.2f}%")
            plt.title("Frecuencia de Colores Clasificados",fontsize=16)
            plt.xlabel("Color", fontsize=16)
            plt.ylabel("Frequency", fontsize=16)
            plt.legend(fontsize=18)

            plt.subplot(1, 2, 2)
            result_image = Image.fromarray(self.classified_image)
            plt.imshow(result_image)
            plt.title("Imagen Clasificada",fontsize=16)
            plt.axis('off')  

            plt.tight_layout()

    
    def color_score(self):
        self.color_s=len(self.significant_colors)
        return self.color_s