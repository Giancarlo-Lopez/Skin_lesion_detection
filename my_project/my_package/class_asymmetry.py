import numpy as np
import cv2
from .class_for_operations_masks import ClassForOperationsMasks

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

        ax2 = plt.subplot(1, 2, 2)
        plt.imshow(self.rotated_image, cmap='gray')
        plt.axis('off')

        for start_point, end_point in self.segments_left:
            ax2.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='red')

        for start_point, end_point in self.segments_right:
            ax2.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='yellow')
        ax2.set_title(f'Diferencia perímetro Derecha-Izquierda: {round(pa2,2)}')

        plt.tight_layout()
        plt.show()

    
    def method_2(self):
        min_percentage_asymetry_1=19
        #min_percentage_asymetry_1=24
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
