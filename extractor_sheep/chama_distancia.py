import cv2
from extractor_sheep import distancia
#import distancia
import sys
import math
from tkinter import filedialog
import os

class EuclideanDistanceActiveCountor():
     def __init__(self):
            pass
     def run(self, image):

            # Process command line arguments
            #def main():
            #file_to_load = filedialog.askopenfilename() #"uems/teste.png"
            #if len(sys.argv) > 1:
            #    file_to_load = sys.argv[1]

            #pegar nome da ultima pasta
            #valor = file_to_load.split("/")

            #nome_image = valor[len(valor)-1] #ver caminho 

            #peso = nome_image.split(".")[0].split("_")
            #peso = int(peso[2])/10 #ver caminho

            # Loads the desired image
            #img = cv2.imread( file_to_load,0 )
            img=image
            scale_percent = 25 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            #print(dim)
            print(width)
            print(height)
            # resize image
            img2 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            image=img2
            #retval, img3 = cv2.threshold(img2, 10, 255, cv2.THRESH_BINARY)
            #image = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
            #cv2.imshow('',image)
            #cv2.waitKey(3000)

            # Creates the snake
            snake = distancia.Snake(image,  closed = True)

            # Window, window name and trackbars
            snake_window_name = "Snakes"
            controls_window_name = "Controls"
            cv2.namedWindow( snake_window_name )
            cv2.namedWindow( controls_window_name )
            cv2.createTrackbar( "Alpha", controls_window_name, math.floor( snake.alpha * 100 ), 100, snake.set_alpha )
            cv2.createTrackbar( "Beta",  controls_window_name, math.floor( snake.beta * 100 ), 100, snake.set_beta )
            cv2.createTrackbar( "Delta", controls_window_name, math.floor( snake.delta * 100 ), 100, snake.set_delta )
            cv2.createTrackbar( "W Line", controls_window_name, math.floor( snake.w_line * 100 ), 100, snake.set_w_line )
            cv2.createTrackbar( "W Edge", controls_window_name, math.floor( snake.w_edge * 100 ), 100, snake.set_w_edge )
            cv2.createTrackbar( "W Term", controls_window_name, math.floor( snake.w_term * 100 ), 100, snake.set_w_term )
            labels=None
            types=None
            values=None
            # Core loop
            while( True ):

                # Gets an image of the current state of the snake
                snakeImg,labels,types,values = snake.visualize()
                # Shows the image
                cv2.imshow( snake_window_name, snakeImg )
    
                # Processes a snake step
                snake_changed = snake.step()

                # Stops looping when ESC pressed
                k = cv2.waitKey(33)
                if k == 27:
                    break
                    cv2.destroyAllWindows()
                if values!=None:
                    return labels,types,values
                    break
if __name__ == "__main__":
   lista={"/home/diegopc/Documents/inovisao/testes/extractor_sheep/B2_360_352_210_json_label.png",
                "/home/diegopc/Documents/inovisao/testes/extractor_sheep/B2_360_352_210_json_labell.png",
                "/home/diegopc/Documents/inovisao/testes/extractor_sheep/B5_1910_345_30_json_label.png",
                "/home/diegopc/Documents/inovisao/testes/extractor_sheep/B5_1910_345_270_json_label.png",
                "/home/diegopc/Documents/inovisao/testes/extractor_sheep/B5_1910_345_300_json_label.png",
                "/home/diegopc/Documents/inovisao/testes/extractor_sheep/B5_1910_345_330_json_label.png",
                "/home/diegopc/Documents/inovisao/testes/extractor_sheep/B5_1910_345_360_json_label.png",
                "/home/diegopc/Documents/inovisao/testes/extractor_sheep/B5_1910_345_390_json_label.png",
                "/home/diegopc/Documents/inovisao/testes/extractor_sheep/B5_1910_345_420_json_label.png",
                "/home/diegopc/Documents/inovisao/testes/extractor_sheep/B5_1910_345_450_json_label.png",
                "/home/diegopc/Documents/inovisao/testes/extractor_sheep/B5_1910_345_480_json_label.png",
                "/home/diegopc/Documents/inovisao/testes/extractor_sheep/B5_1910_345_510_json_label.png",
                "/home/diegopc/Documents/inovisao/testes/extractor_sheep/B5_1910_345_540_json_label.png",
                "/home/diegopc/Documents/inovisao/testes/extractor_sheep/B5_1910_345_570_json_label.png",
                "/home/diegopc/Documents/inovisao/testes/extractor_sheep/B5_1910_345_660_json_label.png",
                "/home/diegopc/Documents/inovisao/testes/extractor_sheep/B5_1910_345_690_json_label.png",
                "/home/diegopc/Documents/inovisao/testes/extractor_sheep/B5_1910_345_720_json_label.png",
                "/home/diegopc/Documents/inovisao/testes/extractor_sheep/B5_1910_345_810_json_label.png"}
   for caminho in lista:
       alg=EuclideanDistanceActiveCountor() 
       print(caminho)
       img=cv2.imread(caminho)
       alg.run(img)          

