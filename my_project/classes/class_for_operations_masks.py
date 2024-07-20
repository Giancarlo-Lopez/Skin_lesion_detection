from .libraries import *

class ClassForOperationsMasks():
    
    def __init__(self,mask):
        self.mask=mask
        self.centroid_mask()
    
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