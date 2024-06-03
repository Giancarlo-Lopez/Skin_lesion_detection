import numpy as np
import cv2
from .class_for_operations_masks import ClassForOperationsMasks
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt

class ClassBorder(ClassForOperationsMasks):

    def __init__(self, mask):
        super().__init__(mask)
        self.mask_area_regions()
        self.cal_areas()
        self.calcule_perimeters()

    def mask_area_regions(self):
        self.mask_centering = self.center_mask()
        center_mass = center_of_mass(self.mask_centering)
        self.center_mass_x = round(center_mass[1])
        self.center_mass_y = round(center_mass[0])
        
        self.sectors = [np.zeros_like(self.mask_centering, dtype=bool) for _ in range(8)]
        for y in range(self.mask_centering.shape[0]):
            for x in range(self.mask_centering.shape[1]):
                if self.mask_centering[y, x]:
                    angulo = np.degrees(np.arctan2(y - self.center_mass_y, x - self.center_mass_x)) % 360
                    indice_sector = int(angulo // 45) % 8
                    self.sectors[indice_sector][y, x] = True
        return self.sectors
    
    def cal_areas(self):
        self.areas = [np.sum(sector) for sector in self.sectors]
        return self.areas
  
    def plot_areas(self):
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        for i, ax in enumerate(axs.flat):
            ax.imshow(self.sectors[i], cmap='gray')
            ax.set_title(f'Sector {i+1}')
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    def calcule_perimeters(self):
        self.mask_centering=self.mask
       
        contours, _ = cv2.findContours(self.mask_centering, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contour = contours[0].squeeze()

        M = cv2.moments(self.contour)

        self.cx = int(M['m10'] / M['m00'])
        self.cy = int(M['m01'] / M['m00'])

        self.angles = np.arctan2(self.contour[:, 1] - self.cy, self.contour[:, 0] - self.cx) * 180 / np.pi
        self.angles[self.angles < 0] += 360  

        self.segment_lengths = np.zeros(8) 

      
        for i in range(len(self.contour)):
            j = (i + 1) % len(self.contour)
            segment_index = int(self.angles[i] // 45)
            distance = np.linalg.norm(self.contour[i] - self.contour[j])
            self.segment_lengths[segment_index] += distance

        self.segment_lengths

        return self.segment_lengths
    
    
    def plot_perimeters(self):

        colors = plt.cm.get_cmap('tab10', 8)

        plt.figure(figsize=(6, 6))
        for i in range(len(self.contour)):
            j = (i + 1) % len(self.contour)
            segment_index = int(self.angles[i] // 45)
            plt.plot([self.contour[i][0], self.contour[j][0]], [self.contour[i][1], self.contour[j][1]], color=colors(segment_index))

        plt.scatter(self.cx, self.cy, color='red', s=100, zorder=5, label='Centroide')

        plt.title("División en 8 segmentos")
        plt.legend()
        plt.axis('off')
        plt.axis('equal')
        plt.show()


    def index_borders_functions(self):
        index_borde=[]
        for i in range(8):
            index=(self.segment_lengths[i]**2)/(2*pi*self.areas[i])
            index_borde.append(round(index,2))
        self.B_score=round(sum(index_borde),2)
        return self.B_score
