class ClassForPlot():
        
    def plot_two_images(image1, image2, title1, title2):
        plt.figure(figsize=(8, 5))  

        plt.subplot(1, 2, 1)  
        plt.imshow(image1, cmap='gray')
        if title1:
            percentage1 = float(title1.split(': ')[1].strip(' %')) 
            color1 = 'red' if percentage1 > 19 else 'blue'
            plt.title(title1, color=color1)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(image2, cmap='gray')
        if title2:
            percentage2 = float(title2.split(': ')[1].strip(' %')) 
            color2 = 'red' if percentage2 > 19 else 'blue'
            plt.title(title2, color=color2)
        plt.axis('off')
        plt.tight_layout()  
        plt.show()


    def plot_two_image(image1, image2, title1, title2):
        plt.figure(figsize=(10, 6))  

        plt.subplot(1, 2, 1)  
        plt.imshow(image1, cmap='gray')
        plt.title(title1)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(image2, cmap='gray')
        plt.title(title2)
        plt.axis('off')
        plt.show()
 
    def plot_three_images(image1, image2, image3, title1, title2, title3):
        plt.figure(figsize=(15, 5))  
        plt.subplot(1, 3, 1)  
        plt.imshow(image1, cmap='gray')
        plt.title(title1)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(image2, cmap='gray')
        plt.title(title2)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(image3, cmap='gray')
        plt.title(title3)
        plt.axis('off')

        plt.tight_layout()  
        plt.show()

    def plot_four_image(image1, image2, image3, image4, title1, title2, title3, title4):
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1) 
        plt.imshow(image1, cmap='gray')
        plt.title(title1)
        plt.axis('off') 

        plt.subplot(2, 2, 2) 
        plt.imshow(image2, cmap='gray')
        plt.title(title2)
        plt.axis('off')

        plt.subplot(2, 2, 3) 
        plt.imshow(image3, cmap='gray')
        plt.title(title3)
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(image4, cmap='gray')
        plt.title(title4)
        plt.axis('off')

        plt.tight_layout() 
        plt.show()

    def plot_fifteen_images(images, title):
        plt.figure(figsize=(10, 14))  

        for i in range(15):
            ax = plt.subplot(5, 3, i + 1)
            ax.imshow(images[i], cmap='gray') 
            ax.set_title(title[i])
            ax.axis('off')  

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.1) 
        plt.show()

    def plot_six_images(images, titles):

        plt.figure(figsize=(10, 7))  

        for i in range(6):
            plt.subplot(2, 3, i + 1) 
            plt.imshow(images[i], cmap='gray')  
            plt.title(titles[i])  
            plt.axis('off')  

        plt.tight_layout()  
        plt.subplots_adjust(wspace=0.1, hspace=0.1) 
        plt.show()  
