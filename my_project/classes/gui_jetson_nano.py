
import threading
import Jetson.GPIO as GPIO
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import center_of_mass
from math import *
from PIL import Image
from torchvision import transforms 
import fitz  
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from collections import Counter
import torch.nn as nn
import seaborn as sns
import os


#################################################################################################################
###########################################Operaciones###########################################################
#################################################################################################################

class ClassForOperationsMasks():
    
    def __init__(self,mask):
        self.mask=mask
        self.centroid_mask()
    
    def mask_n(self):
        if isinstance(self.mask, torch.Tensor):
            if self.mask.device == torch.device('cpu'):
                return self.mask.numpy()
            else:
                return self.mask.cpu().numpy()
        elif isinstance(self.mask, np.ndarray):
            return self.mask

    def centroid_mask(self):

        center_mass = center_of_mass(self.mask)
        self.center_mass_x = round(center_mass[1])
        self.center_mass_y = round(center_mass[0])
        return self.center_mass_x, self.center_mass_y
        
    def center_mask(self):
    
        width_xo = self.mask.shape[1]//2
        width_yo = self.mask.shape[0]//2
        

        displacement_difference_x = round(width_xo - self.center_mass_x)
        displacement_difference_y = round(width_yo - self.center_mass_y)

        if displacement_difference_x < 0:
            displacement_difference_x = abs(displacement_difference_x)
            mask_c = np.roll(self.mask, -displacement_difference_x, axis=1)
        else:
            mask_c = np.roll(self.mask, displacement_difference_x, axis=1)

        if displacement_difference_y > 0:
            displacement_difference_y = abs(displacement_difference_y)
            mask_c = np.roll(mask_c, displacement_difference_y, axis=0)
        else:
            mask_c = np.roll(mask_c, displacement_difference_y, axis=0)

        return mask_c
    
    def perimeter(self):
        self.contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeters = []
        for contour in self.contours:
            
            perimeter1 = cv2.arcLength(contour, True)  
            perimeters.append(perimeter1)

        return perimeters
    
    def filter_mask(self):
        self.mask = self.mask.astype(np.uint8)
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            self.mask = np.zeros_like(self.mask)
            cv2.drawContours(self.mask, [max_contour], -1, 255, thickness=cv2.FILLED)
        else:
            self.mask = np.zeros_like(self.mask)
        return self.mask
#################################################################################################################
###########################################Segmentacion##########################################################
#################################################################################################################
class Class_Segmention_Otsu():

    def __init__(self,image_path):
        self.image_path=image_path
        self.load_image()
        self.otzu_borders0()
        self.otzu_borders()
        self.hair_remove()
        self.otzu_edges()
        self.otzu_edges_2()
        self.otzu_edges_3()
    
    def load_image(self):
        
        self.image_color = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        if self.image_color is None:
            raise FileNotFoundError("Verifique la ruta de la imagen.")
        self.image_color_rgb = cv2.cvtColor(self.image_color, cv2.COLOR_BGR2RGB)
        self.image_gray = cv2.cvtColor(self.image_color_rgb, cv2.COLOR_RGB2GRAY)
        return self.image_color_rgb

    def otzu_borders0(self):
        
        blurred = cv2.GaussianBlur(self.image_gray, (15, 15), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)
        
        self.contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.largest_contour = max(self.contours, key=cv2.contourArea)
        mask0 = np.zeros_like(self.image_gray)
        cv2.drawContours(mask0, [self.largest_contour], -1, 255, thickness=cv2.FILLED)
        masked_image = cv2.bitwise_and(self.image_color_rgb, self.image_color_rgb, mask=thresh)

        return masked_image, mask0

    def otzu_borders(self):

        mask11 = np.zeros_like(self.image_gray)
        cv2.drawContours(mask11, [self.largest_contour], -1, 255, thickness=cv2.FILLED)
        masked_image11 = cv2.bitwise_and(self.image_color_rgb, self.image_color_rgb, mask=mask11)

        return mask11,masked_image11
     

    def otzu_edges(self):

        self.largest_contours_1 = sorted(self.contours, key=cv2.contourArea, reverse=True)[:3]
        mask1=np.zeros_like(self.image_gray)
        cv2.drawContours(mask1, self.largest_contours_1, -1, 255, thickness=cv2.FILLED)
        masked_image_1 = cv2.bitwise_and(self.image_color_rgb, self.image_color_rgb, mask=mask1)
        return mask1, masked_image_1


    def otzu_edges_2(self):

        image_center_x = self.image_color.shape[1] / 2
        closest_contour = None
        min_distance_to_center = float('inf')

        for contour in self.largest_contours_1:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                centroid_x = int(M['m10'] / M['m00'])
                distance = abs(centroid_x - image_center_x)
                if distance < min_distance_to_center:
                    min_distance_to_center = distance
                    closest_contour = contour
        mask2=np.zeros_like(self.image_gray)
        if closest_contour is not None:
            cv2.drawContours(mask2, [closest_contour], -1, 255, thickness=cv2.FILLED)

        masked_image_2 = cv2.bitwise_and(self.image_color_rgb, self.image_color_rgb, mask=mask2)
        
        return mask2, masked_image_2


    def hair_remove(self):
 
        kernel = cv2.getStructuringElement(1,(15,15)) 
        blackhat = cv2.morphologyEx(self.image_gray, cv2.MORPH_BLACKHAT, kernel)
        filter_gaussian= cv2.GaussianBlur(blackhat,(15,15),cv2.BORDER_DEFAULT)
        _,thresholding = cv2.threshold(filter_gaussian,10,255,cv2.THRESH_BINARY)
        self.final_image = cv2.inpaint(self.image_color_rgb,thresholding,1,cv2.INPAINT_TELEA)
        return self.final_image

    def otzu_edges_3(self):

        image_gray_f = cv2.cvtColor(self.final_image, cv2.COLOR_RGB2GRAY)
        
        blurred_f = cv2.GaussianBlur(image_gray_f, (15, 15), 0)
        _, thresh_f = cv2.threshold(blurred_f, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_f = cv2.bitwise_not(thresh_f)

        self.contours_f, _ = cv2.findContours(thresh_f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contours_f = sorted(self.contours_f, key=cv2.contourArea, reverse=True)[:1]

        mask_f = np.zeros_like(self.image_gray)
        cv2.drawContours(mask_f, largest_contours_f, -1, 255, thickness=cv2.FILLED)
        
        masked_image_f = cv2.bitwise_and(self.final_image, self.final_image, mask=mask_f)
        
        return mask_f, masked_image_f
        
    def otzu_edges_4(self):

        image_gray_4 = cv2.cvtColor(self.final_image, cv2.COLOR_BGR2GRAY)
        
        blurred_4 = cv2.GaussianBlur(image_gray_4, (25, 25), 0)
        _, thresh_4 = cv2.threshold(blurred_4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_4 = cv2.bitwise_not(thresh_4)
        contours_4, _ = cv2.findContours(thresh_4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contours_4 = sorted(contours_4, key=cv2.contourArea, reverse=True)[:3]

        image_center_x4 = self.final_image.shape[1] / 2
        image_center_y4 = self.final_image.shape[0] / 2

        closest_contour_4 = None
        min_distance_to_center_4 = float('inf')

        for contour in largest_contours_4:
            M4 = cv2.moments(contour)
            if M4['m00'] != 0:
                centroid_x4 = int(M4['m10'] / M4['m00'])
                centroid_y4 = int(M4['m01'] / M4['m00'])
            
                distance4 = np.sqrt((centroid_x4 - image_center_x4) ** 2 + (centroid_y4 - image_center_y4) ** 2)
                if distance4 < min_distance_to_center_4:
                    min_distance_to_center_4 = distance4
                    closest_contour_4 = contour

        mask4 = np.zeros_like(self.image_gray)
        if closest_contour_4 is not None:
            cv2.drawContours(mask4, [closest_contour_4], -1, 255, thickness=cv2.FILLED)

        masked_image_4 = cv2.bitwise_and(self.final_image, self.final_image, mask=mask4)
    
        resized_mask_4 = cv2.resize(mask4, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        return mask4, masked_image_4,resized_mask_4


#################################################################################################################
###########################################Asimetria#############################################################
#################################################################################################################
class ClassAsymmetry(ClassForOperationsMasks):
    
    def __init__(self, mask):
        super().__init__(mask)
        
        self.mask_centering = self.center_mask() 
        self.is_touching_border()
        self.find_farthest_points()  
        self.angle_g()
        self.rotate_image()
        self.divide_images()  

    def is_touching_border(self):
        self.touch_border=False
        self.height, self.width = self.mask.shape
        self.contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        for contour in self.contours:
            for point in contour:
                x, y = point[0][0], point[0][1]
                if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
                    self.touch_border=True
                else:
                    self.touch_border=False
        return self.contours

    def find_farthest_points(self):
        if self.touch_border:
            self.mask_centering=self.mask
        self.contours, _ = cv2.findContours(self.mask_centering, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.max_dist = 0
        farthest_pair = (0, 0)  

        for contour in self.contours:
            for i in range(len(contour)):
                for j in range(i + 1, len(contour)):
                    pt1 = contour[i][0]
                    pt2 = contour[j][0]
                    dist = np.linalg.norm(pt1 - pt2) 
                    if dist > self.max_dist:
                        self.max_dist = dist
                        farthest_pair = (pt1, pt2) 
        self.pt1=farthest_pair[0]
        self.pt2=farthest_pair[1]

        return self.pt1,self.pt2
    
    def angle_g(self):
        x1=self.pt1[0]
        y1=self.pt1[1]
        x2=self.pt2[0]
        y2=self.pt2[1]

        dx = x1 - x2
        dy = y1 - y2
        m = dy / dx 
        angle = np.arctan(m)
        self.angle_grades = np.degrees(angle)
        return self.angle_grades
    
    def rotate_image(self):
        (height, width) = self.mask_centering.shape
        self.center = (width / 2, height / 2)
        
        matrix_rotation = cv2.getRotationMatrix2D(self.center, self.angle_grades, 1.0)
        self.rotated_image = cv2.warpAffine(self.mask_centering, matrix_rotation, (width, height))
        if self.touch_border:
            self.rotated_image=self.mask   
        return self.rotated_image
    
    def divide_images(self):
        #center_mass = center_of_mass(self.rotated_image)
        self.center_mass_x = self.width//2
        self.center_mass_y = self.height//2

        self.mask_left = self.rotated_image.copy()
        self.mask_left[:, self.center_mass_x:] = 0
        self.mask_right = self.rotated_image.copy()
        self.mask_right[:,:self.center_mass_x] = 0
        self.mask_right  = np.roll(self.mask_right, 1, axis=1)
    
        
        self.top_mask = np.zeros_like(self.rotated_image)  
        self.down_mask = np.zeros_like(self.rotated_image)  
        self.top_mask[:self.center_mass_y, :] = self.rotated_image[:self.center_mass_y, :]
        self.down_mask[self.center_mass_y:, :] = self.rotated_image[self.center_mass_y:, :]
        self.down_mask  = np.roll(self.down_mask, 1, axis=0)
        
        return self.mask_right, self.mask_left, self.top_mask, self.down_mask

    def reflect_mask_x_y(self, mask, mirror_value_x, mirror_value_y):
        
        reflected_mask_x = np.zeros_like(mask)
        reflected_mask_y = np.zeros_like(mask)
        self.height, self.width = mask.shape

        for y in range(self.height):
            for x in range(self.width):
                new_x = mirror_value_x + (mirror_value_x - x)
                new_y = mirror_value_y + (mirror_value_y - y)
                if 0 <= new_x < self.width:
                    reflected_mask_x[y, new_x] = mask[y, x]
                if 0 <= new_y < self.height:
                    reflected_mask_y[new_y, x] = mask[y, x]

        return reflected_mask_x, reflected_mask_y

    def area_union(self,mask1,mask2):
        self.mask_union = mask1 ^ mask2
        self.mask_two = mask1 + mask2
        area_mask = np.sum(mask1 == 255).item()
        area_union=np.sum(self.mask_union == 255).item()
        self.percentage_asimetry=round(area_union*100/area_mask)
        return self.mask_union,self.mask_two,self.percentage_asimetry

    def asymmetry_method(self):

        reflected_right, _ = self.reflect_mask_x_y(self.mask_right, self.center_mass_x, self.center_mass_y)
        _, reflected_top = self.reflect_mask_x_y(self.top_mask, self.center_mass_x, self.center_mass_y)

        union_left, asymmetry_left,self.percentage_asimetry_left= self.area_union(self.mask_left, reflected_right)
        union_top, asymmetry_top,self.percentage_asimetry_down = self.area_union(self.down_mask,reflected_top)

        return union_left, asymmetry_left, union_top, asymmetry_top,self.percentage_asimetry_left,self.percentage_asimetry_down
    

    def calculate_perimeters(self):

        self.segments_top = []
        self.segments_down = []
        self.segments_left = []
        self.segments_right = []

        contours, _ = cv2.findContours(self.rotated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cx, cy = self.width // 2, self.height // 2
        self.perimeter_top = self.perimeter_down = self.perimeter_left = self.perimeter_right = 0

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

        for contour in contours:
            for i in range(len(contour)):
                start_point = contour[i][0]
                end_point = contour[(i + 1) % len(contour)][0]
                angle = np.degrees(np.arctan2(start_point[1] - cy, start_point[0] - cx))% 360
                
                segment_length = np.linalg.norm(start_point - end_point)
                if 0 <= angle < 180:
                    self.perimeter_top += segment_length
                    self.segments_top.append((start_point, end_point))
                elif 180 <= angle < 360:
                    self.perimeter_down += segment_length
                    self.segments_down.append((start_point, end_point))
                if 90 <= angle < 270:
                    self.perimeter_left += segment_length
                    self.segments_left.append((start_point, end_point))
                elif (0 <= angle < 90) or (270 <= angle < 360):
                    self.perimeter_right += segment_length
                    self.segments_right.append((start_point, end_point))

        return self.perimeter_top, self.perimeter_down, self.perimeter_left, self.perimeter_right
    
    def plot_perimeters(self):
        plt.figure(figsize=(10, 8))
        
        ax1 = plt.subplot(1, 2, 1)
        plt.imshow(self.rotated_image, cmap='gray')
        plt.axis('off')

        for start_point, end_point in self.segments_top:
            ax1.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='blue')

        for start_point, end_point in self.segments_down:
            ax1.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='green')
        perimetro_max_1=max(self.perimeter_down,self.perimeter_top)
        perimetro_max_2=max(self.perimeter_right,self.perimeter_left)
        pa1=abs(self.perimeter_down-self.perimeter_top)*100/perimetro_max_1
        pa2=abs(self.perimeter_right-self.perimeter_left)*100/perimetro_max_2
        ax1.set_title(f'Diferencia perímetro Superior-Inferior: {round(pa1,2)}')
        #ax1.set_title(f'Perímetro Superior-Inferior',fontsize=16) 

        ax2 = plt.subplot(1, 2, 2)
        plt.imshow(self.rotated_image, cmap='gray')
        plt.axis('off')

        for start_point, end_point in self.segments_left:
            ax2.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='red')

        for start_point, end_point in self.segments_right:
            ax2.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='yellow')
        ax2.set_title(f'Perímetro Derecha-Izquierda: {round(pa2,2)}')
        #ax2.set_title(f'Perímetro Derecha-Izquierda',fontsize=16)
        plt.tight_layout()
        plt.show()

    
    def method_2(self):
        #min_percentage_asymetry_1=19
        min_percentage_asymetry_1=24
        self.A_score_1=0
        if(self.percentage_asimetry_left<min_percentage_asymetry_1 and self.percentage_asimetry_down<min_percentage_asymetry_1):
            self.A_score_1=0
        elif(self.percentage_asimetry_left<min_percentage_asymetry_1 or self.percentage_asimetry_down<min_percentage_asymetry_1):
            self.A_score_1=1
        else:
            self.A_score_1=2

        
        max_perimeter_right_left=max(self.perimeter_left,self.perimeter_right)
        max_perimeter_top_down=max(self.perimeter_top,self.perimeter_down)
        difference_perimeter_right_left= (abs(self.perimeter_left-self.perimeter_right)*100)/max_perimeter_right_left
        difference_perimeter_top_down= (abs(self.perimeter_top-self.perimeter_down)*100)/max_perimeter_top_down
        
        p1_umbral=5.5
        p2_umbral=7

        self.A_score_2=0
        if (difference_perimeter_right_left < p1_umbral and difference_perimeter_top_down <p1_umbral):
            self.A_score_2=0
        elif(difference_perimeter_right_left < p2_umbral or difference_perimeter_top_down <p2_umbral):
            self.A_score_2=1
        else:
            self.A_score_2=2

        self.A_score=(self.A_score_1+self.A_score_2)/2
        #self.A_score=self.A_score_1
        return self.A_score_1,self.A_score_2,self.A_score


#################################################################################################################
###########################################Borde#################################################################
#################################################################################################################
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

        plt.title("División en 8 segmentos",fontsize=12)
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


#################################################################################################################
###########################################Color#################################################################
#################################################################################################################
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

        
        self.reference_colors_1 = {
        'cafe_claro': np.array([144,76,0]),
        'cafe_oscuro': np.array([40,0,0]),
        'rojo': np.array([176, 54, 27]),
        'blanco': np.array([255, 255, 255]),
        'negro': np.array([0, 0, 0]),
        'azul': np.array([83, 167, 174])
        }

        self.reference_colors_2 = { 
        'cafe_claro': np.array([210, 180, 140]),
        'cafe_oscuro': np.array([101, 67, 33]),
        'rojo': np.array([255, 0, 0]),
        'blanco': np.array([255, 255, 255]),
        'negro': np.array([50, 50, 50]),
        'azul': np.array([0, 0, 255])
        }

        self.reference_colors_3 = { 
        'cafe_claro': np.array([200, 131, 86]),
        'cafe_oscuro': np.array([160, 95, 58]),
        'rojo': np.array([210, 51, 35]),
        'blanco': np.array([255, 255, 255]),
        'negro': np.array([0, 0, 0]),
        'azul': np.array([74, 134, 137])
        }
 
    
    
    def process_and_classify_pixels(self):
        #self.reference_colors=self.reference_colors_1
        #self.reference_colors=self.reference_colors_2
        self.reference_colors=self.reference_colors_3
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
            plt.xlabel("Color")
            plt.ylabel("Frecuencia")
            plt.legend()

            plt.subplot(1, 2, 2)
            result_image = Image.fromarray(self.classified_image)
            plt.imshow(result_image)
            plt.title("Imagen Clasificada",fontsize=16)
            plt.axis('off')  

            plt.tight_layout()

    
    def color_score(self):
        self.color_s=len(self.significant_colors)
        return self.color_s
#################################################################################################################
###########################################Diametro##############################################################
#################################################################################################################
class ClassDiameter():

    def __init__(self,image_path,mask):
        self.image_path=image_path
        self.mask=mask
        self.find_farthest_points()
        #self.plot_diameter()
        self.obtener_dpi()
    

    def find_farthest_points(self):

        y_indices, x_indices = np.where(self.mask == 255)
        points = np.column_stack((x_indices, y_indices))
        hull = cv2.convexHull(points)
        hull_points = hull[:, 0, :] 

        self.max_dist = 0
        farthest_pair = (0, 0)

        for i in range(len(hull_points)):
            for j in range(i + 1, len(hull_points)):
                dist = np.linalg.norm(hull_points[i] - hull_points[j])
                if dist > self.max_dist:
                    self.max_dist = dist
                    farthest_pair = (hull_points[i], hull_points[j])

        self.pt1, self.pt2 = farthest_pair

        return self.pt1, self.pt2, self.max_dist
    
    def plot_diameter(self):
        
        plt.figure(figsize=(5, 5))
        plt.imshow(self.mask, cmap='gray') 
        plt.title('Diametro')
        plt.plot([self.pt1[0], self.pt2[0]], [self.pt1[1], self.pt2[1]], 'r-', linewidth=2)
        plt.scatter([self.pt1[0], self.pt2[0]], [self.pt1[1], self.pt2[1]], color='green', s=50)
        plt.axis('off')
        plt.show()

    def obtener_dpi(self):
        documento = fitz.open(self.image_path)
        pagina = documento[0]
        pix = pagina.get_pixmap()
        self.dpi_horizontal = pix.xres
        self.dpi_vertical = pix.yres
        documento.close()
        return self.dpi_horizontal, self.dpi_vertical
    
    def score_diameter(self):
        M=round((self.max_dist*25.4)/(20*96),2)
        if (M < 2):
            self.D_score = 0.5
        elif M < 3:
            self.D_score = 1
        elif M < 4:
            self.D_score = 1.5
        elif M < 5:
            self.D_score = 2
        elif M < 6:
            self.D_score = 2.5
        elif M < 7:
            self.D_score = 3
        elif M < 8:
            self.D_score = 3.5
        elif M < 9:
            self.D_score = 4
        elif M < 10:
            self.D_score = 4.5
        else:
            self.D_score = 5

        return self.D_score
#################################################################################################################
###########################################GUI####################################################################
#################################################################################################################



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
        self.segmentation_canvas1 = None
        self.segmentation_canvas2 = None

    def create_widgets(self):
        self.frame_top = tk.Frame(self, bd=5, relief=tk.RAISED, bg=self.bg_color, highlightbackground=self.border_color, highlightcolor=self.border_color, highlightthickness=5)
        self.frame_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.frame_left = tk.Frame(self, bd=5, relief=tk.RAISED, bg=self.bg_color, highlightbackground=self.border_color, highlightcolor=self.border_color, highlightthickness=5)
        self.frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.frame_right = tk.Frame(self, bd=5, relief=tk.RAISED, bg=self.bg_color, highlightbackground=self.border_color, highlightcolor=self.border_color, highlightthickness=5)
        self.frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.frame_bottom = tk.Frame(self, bd=5, relief=tk.RAISED, bg=self.bg_color, highlightbackground=self.border_color, highlightcolor=self.border_color, highlightthickness=5)
        self.frame_bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        button_frame = tk.Frame(self.frame_top, bg=self.bg_color)
        button_frame.grid(row=0, column=0, padx=5, pady=5)

        self.load_button = tk.Button(button_frame, text="Cargar Imagen", command=self.load_image_file, font=("Helvetica", 10, "bold"), bg=self.button_color, fg=self.text_color_bu, relief=tk.GROOVE, bd=2)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.show_graphs_button = tk.Button(button_frame, text="Mostrar Gráficas", command=self.show_graphs, font=("Helvetica", 10, "bold"), bg=self.button_color, fg=self.text_color_bu, relief=tk.GROOVE, bd=2, state=tk.DISABLED)
        self.show_graphs_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.open_camera_button = tk.Button(button_frame, text="Abrir Cámara", command=self.open_camera, font=("Helvetica", 10, "bold"), bg=self.button_color, fg=self.text_color_bu, relief=tk.GROOVE, bd=2)
        self.open_camera_button.pack(side=tk.LEFT, padx=5, pady=5)

    def load_image_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.clear_top_frame()
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

    def open_camera(self):
        self.master.destroy()  
        #CameraGUI()
        CameraApp()  

    def show_graphs(self):
        self.clear_graphs()
        self.show_asymmetry_graph()
        self.show_asymmetry_graph2()
        self.show_border_graph()
        self.show_color_graph()
        self.show_diameter_graph2()
        self.display_results()
        self.show_graphs_button.config(state=tk.DISABLED)

    def clear_frames(self):
        for frame in [self.frame_left, self.frame_right, self.frame_bottom]:
            for widget in frame.winfo_children():
                widget.destroy()

    def clear_graphs(self):
        for frame in [self.frame_left, self.frame_right, self.frame_bottom]:
            for widget in frame.winfo_children():
                widget.destroy()

    def clear_top_frame(self):
        if self.segmentation_canvas1:
            self.segmentation_canvas1.get_tk_widget().destroy()
            self.segmentation_canvas1 = None
        if self.segmentation_canvas2:
            self.segmentation_canvas2.get_tk_widget().destroy()
            self.segmentation_canvas2 = None

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
        results_frame = tk.Frame(self.frame_bottom, bg=self.bg_color)
        results_frame.pack(padx=10, pady=5)

        for i, (label, value) in enumerate(results):
            tk.Label(results_frame, text=f'{label} {value}', font=("Helvetica", 9), bg=self.bg_color, fg="black").grid(row=i // 2, column=i % 2, padx=10, pady=5, sticky="w")

    def show_segmentation_graph(self):
        self.clear_top_frame()
        title1 = 'Imagen'
        title2 = 'Segmentación'

        plot_segmentation1 = plt.figure(figsize=(1.5, 1.5), facecolor=self.bg_color)
        plt.imshow(self.image_color_rgb)
        plt.title(title1, fontsize=10, color=self.text_color)
        plt.axis('off')
        plt.tight_layout()
        plt.close(plot_segmentation1)
        self.segmentation_canvas1 = FigureCanvasTkAgg(plot_segmentation1, master=self.frame_top)
        self.segmentation_canvas1.draw()
        self.segmentation_canvas1.get_tk_widget().grid(row=0, column=3, padx=5, pady=5, sticky="e")

       
        plot_segmentation2 = plt.figure(figsize=(1.5, 1.5), facecolor=self.bg_color)
        plt.imshow(self.masked_image_4, cmap='gray')
        plt.title(title2, fontsize=10, color=self.text_color)
        plt.axis('off')
        plt.tight_layout()
        plt.close(plot_segmentation2)
        self.segmentation_canvas2 = FigureCanvasTkAgg(plot_segmentation2, master=self.frame_top)
        self.segmentation_canvas2.draw()
        self.segmentation_canvas2.get_tk_widget().grid(row=0, column=4, padx=5, pady=5, sticky="e")

    def show_asymmetry_graph(self):
        title1 = f'Diferencia de área horizontal $Ax_{{score}}$: {self.percentage_asimetry_left} %'
        title2 = f'Diferencia de área vertical $Ay_{{score}}$: {self.percentage_asimetry_down} %'

        plot_asymmetry = plt.figure(figsize=(3, 2), facecolor=self.bg_color)
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
        canvas = FigureCanvasTkAgg(plot_asymmetry, master=self.frame_left)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=10, pady=10)

    def show_asymmetry_graph2(self):
        plot_asymmetry2 = plt.figure(figsize=(3, 2), facecolor=self.bg_color)

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
        canvas = FigureCanvasTkAgg(plot_asymmetry2, master=self.frame_left)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=10, pady=10)

    def show_border_graph(self):
        colors = plt.colormaps.get_cmap('tab10')

        fig = plt.figure(figsize=(2, 2), facecolor=self.bg_color)
        for i in range(len(self.contour)):
            j = (i + 1) % len(self.contour)
            segment_index = int(self.angles[i] // 45)
            plt.plot([self.contour[i][0], self.contour[j][0]], [self.contour[i][1], self.contour[j][1]], color=colors(segment_index))

        plt.scatter(self.cx, self.cy, color='red', s=100, zorder=5, label='Centroide')

        plt.title("División en 8 segmentos", fontsize=8, color=self.text_color)
        plt.legend()
        plt.axis('off')
        plt.axis('equal')
        plt.close(fig)

        canvas = FigureCanvasTkAgg(fig, master=self.frame_right)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=10, pady=10)

    def show_color_graph(self):
        freq_colors_plot = plt.figure(figsize=(5, 2), facecolor=self.bg_color)
        plt.subplot(1, 2, 1)
        for color, count in self.significant_colors:
            plt.bar(color, count, color=np.array(self.reference_colors[color]) / 255, label=f"{color}: {count / self.total_pixels * 100:.2f}%")
        plt.title("Frecuencia de Colores Clasificados", fontsize=8, color=self.text_color)
        plt.xlabel("Color", color=self.text_color, fontsize=8)
        plt.ylabel("Frecuencia", color=self.text_color, fontsize=8)
        plt.legend()

        plt.subplot(1, 2, 2)
        result_image = Image.fromarray(self.classified_image)
        plt.imshow(result_image)
        plt.title("Imagen Clasificada", fontsize=8, color=self.text_color)
        plt.axis('off')
        plt.tight_layout()
        plt.close(freq_colors_plot)
        canvas = FigureCanvasTkAgg(freq_colors_plot, master=self.frame_bottom)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=10, pady=10)

    def show_diameter_graph2(self):
        fig_diameter = plt.figure(figsize=(2, 2), facecolor=self.bg_color)
        plt.imshow(self.mask_centering, cmap='gray')
        plt.title("Diámetro", fontsize=10, color=self.text_color)
        plt.plot([self.pt1[0], self.pt2[0]], [self.pt1[1], self.pt2[1]], 'r-', linewidth=2)
        plt.scatter([self.pt1[0], self.pt2[0]], [self.pt1[1], self.pt2[1]], color='green', s=50)
        plt.axis('off')
        plt.close(fig_diameter)

        canvas = FigureCanvasTkAgg(fig_diameter, master=self.frame_right)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=10, pady=10)

class CameraApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Camera App")
        self.geometry("1000x600")

        self.create_widgets()

        self.cap = None
        self.setup_camera()

        self.button_pins = [18, 22, 24]
        GPIO.setmode(GPIO.BOARD)
        for pin in self.button_pins:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.add_event_detect(pin, GPIO.FALLING, callback=self.handle_button, bouncetime=300)

        self.running = True
        self.camera_thread = threading.Thread(target=self.show_video)
        self.camera_thread.daemon = True
        self.camera_thread.start()

        self.photo_taken = False
        self.photo_count = 0
        self.last_frame = None

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        self.left_frame = ttk.Frame(self)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.status_label = ttk.Label(self.left_frame, text="Listo para tomar foto", font=('Arial', 14, 'bold'), foreground='blue')
        self.status_label.pack(side=tk.TOP, fill=tk.X, pady=10)

        self.entry_frame = ttk.Frame(self.left_frame)
        self.entry_frame.pack(side=tk.TOP, fill=tk.X, padx=20)

        self.name_label = ttk.Label(self.entry_frame, text="Nombre del archivo:")
        self.name_label.pack(side=tk.LEFT)

        self.name_entry = ttk.Entry(self.entry_frame)
        self.name_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.video_label = ttk.Label(self.left_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.right_frame = ttk.Frame(self, width=400, height=600)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.photo_label = ttk.Label(self.right_frame, anchor='center')
        self.photo_label.pack(fill=tk.BOTH, expand=True)

    def setup_camera(self):
        gstreamer_pipeline = (
            "v4l2src device=/dev/video0 ! "
            "image/jpeg, width=1280, height=720, framerate=30/1 ! "
            "jpegdec ! videoconvert ! "
            "appsink"
        )
        self.cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise Exception("No se pudo abrir la cámara")

    def show_video(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((600, 450), Image.ANTIALIAS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

    def take_photo(self):
        ret, frame = self.cap.read()
        if ret:
            self.photo_taken = True
            self.status_label.config(text="Foto tomada")
            print("Foto tomada")
            self.last_frame = frame
            self.display_photo(frame)

    def display_photo(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.resize((400, 600), Image.ANTIALIAS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.photo_label.imgtk = imgtk
        self.photo_label.configure(image=imgtk)

    def save_photo(self):
        if self.photo_taken:
            self.photo_count += 1
            base_name = self.name_entry.get().strip()
            filename = f'{base_name}_{self.photo_count}.jpg'
            cv2.imwrite(filename, self.last_frame)
            self.last_saved_image_path = filename
            print(f"Foto guardada en {filename}")
            self.status_label.config(text=f"Foto guardada como {filename}")

    def on_close(self):
        print("Cerrando la cámara y la aplicación.")
        self.running = False
        if self.cap is not None:
            self.cap.release()
        GPIO.cleanup()
        self.destroy()
        self.restart_main_gui()

    def restart_main_gui(self):
        root = tk.Tk()
        app = GUI(root)
        root.geometry("1024x600")
        root.resizable(False, False)
        root.mainloop()

    def handle_button(self, channel):
        if channel == self.button_pins[0]:
            self.take_photo()
        elif channel == self.button_pins[1]:
            self.save_photo()
        elif channel == self.button_pins[2]:
            self.on_close()

root = tk.Tk()
app = GUI(root)
root.geometry("1024x600")
root.resizable(False, False)
root.mainloop()
