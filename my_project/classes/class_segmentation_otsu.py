'''
import cv2
import numpy as np
'''
from .libraries import *
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
        
        blurred = cv2.GaussianBlur(self.image_gray, (15, 15), 1)
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
        n_kenel=13
        kernel = cv2.getStructuringElement(1,(n_kenel,n_kenel)) 
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
        count=0
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
        
        height, width=mask4.shape
        pixel_top_left=mask4[0,0]
        pixel_top_right=mask4[0,width-1]
        pixel_down_left=mask4[height-1,0]
        pixel_down_right=mask4[height-1,width-1]

        if (pixel_top_left == 0 and pixel_top_right == 0 and pixel_down_left == 0 and pixel_down_right == 0):
            masked_image_4 = cv2.bitwise_and(self.final_image, self.final_image, mask=mask4)
            resized_mask_4 = cv2.resize(mask4, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        else:
            mask_corner = np.zeros((height, width), dtype=np.uint8)
            center_y, center_x = height // 2, width // 2  
            radius = (width)//2 
            Y, X = np.ogrid[:height, :width]
            distance_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            mask_corner[distance_from_center <= radius] = 255  
            mask4[mask_corner == 0] = 0
            n=23
            kernel = np.ones((n, n), np.uint8)  
            img_erode = cv2.erode(mask4, kernel, iterations=1) 
            mask4 = cv2.dilate(img_erode,kernel, iterations=3)
            masked_image_4 = cv2.bitwise_and(self.final_image, self.final_image, mask=mask4)
            resized_mask_4 = cv2.resize(mask4, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        return mask4, masked_image_4,resized_mask_4
