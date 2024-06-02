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