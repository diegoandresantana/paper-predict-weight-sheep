"""
=========================
Measure region properties
=========================

This example shows how to measure properties of labelled image regions.

"""
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.draw import ellipse
from skimage.measure import label, regionprops
from skimage.transform import rotate

class MinorMajorAxis():
     def __init__(self):
            #__init__(self, k, band,  theta):
            pass
     def run(self, image):
            image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            #retval, image = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
            dimensions = image.shape
            # height, width, number of channels in image
            height = image.shape[0]
            width = image.shape[1] 
            
            label_img = label(image)
            regions = regionprops(label_img)

            fig, ax = plt.subplots()
            ax.imshow(image, cmap=plt.cm.gray)
            labels = []
            values = []
            for props in regions:
                y0, x0 = props.centroid
                orientation = props.orientation
                #print("Major: ")
                labels.append("Major_axis")
                #print(props.major_axis_length)
                values.append(props.major_axis_length)
                #print("Minorajor: ")	
                labels.append("Minor_axis")
                values.append(props.minor_axis_length)
                #print(props.minor_axis_length)	
               
                x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
                y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
                x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
                y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length
                
                ax.imshow(image, cmap=plt.cm.gray)
                ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
                ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
                ax.plot(x0, y0, '.g', markersize=15)

                minr, minc, maxr, maxc = props.bbox
                bx = (minc, maxc, maxc, minc, minc)
                by = (minr, minr, maxr, maxr, minr)
                ax.plot(bx, by, '-b', linewidth=2.5)
                break
            types = ['real'] * len(labels)
            ax.axis((0, width, height, 0))
            #plt.show()
            return labels, types, values


class AreaPerimeter():
     def __init__(self):
            #__init__(self, k, band,  theta):
            pass
     def run(self, image):
            areaTotal=0
            perimeterTotal=0
            extend=0
            aspect_ratio=0
            solidity=0
            leftmost=None
            rightmost=None
            topmost=None
            bottommost=None

            labels = []
            values = []
            image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            #retval, image = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
            dimensions = image.shape
            height = image.shape[0]
            width = image.shape[1] 
            cnts, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cXX in cnts:
               #print('Contorno=',c)
               M = cv2.moments(cXX)
               if M["m00"] != 0:
                 cx = int(M["m10"] / M["m00"])
                 cy = int(M["m01"] / M["m00"])  
                 #print('',cx)

               area = cv2.contourArea(cXX)
               leftmost = np.array(tuple(cXX[cXX[:,:,0].argmin()][0]))
               rightmost = np.array(tuple(cXX[cXX[:,:,0].argmax()][0]))
               topmost = np.array(tuple(cXX[cXX[:,:,1].argmin()][0]))
               bottommost = np.array(tuple(cXX[cXX[:,:,1].argmax()][0]))
               
               
               #Aspect Ratio
               x,y,w,h = cv2.boundingRect(cXX)
               aspect_ratio =aspect_ratio+ float(w)/h
               #Extent is the ratio of contour area to bounding rectangle area.
               x,y,w,h = cv2.boundingRect(cXX)
               rect_area = w*h
               extent = float(area)/rect_area
               ###Solidity is the ratio of contour area to its convex hull area.
               hull = cv2.convexHull(cXX)
               hull_area = cv2.contourArea(hull)
               solidity = solidity+ (float(area)/hull_area)
               #print('AREA do contorno=',area)
               areaTotal=areaTotal+area
               perimeter = cv2.arcLength(cXX,True)
               #print('PERIMETRO=',perimeter)
               perimeterTotal=perimeterTotal+perimeter
               break
            #encontra 4 pontos extremos e faz a distancia euclidiana 
            labels.append("Eucl_A_B")
            values.append(cv2.norm(leftmost - rightmost, cv2.NORM_L2))
            labels.append("Eucl_A_C")
            values.append(cv2.norm(leftmost - topmost, cv2.NORM_L2))
            labels.append("Eucl_A_D")
            values.append(cv2.norm(leftmost - bottommost, cv2.NORM_L2))
            labels.append("Eucl_B_C")
            values.append(cv2.norm(rightmost - topmost, cv2.NORM_L2))
            labels.append("Eucl_B_D")
            values.append(cv2.norm(rightmost - bottommost, cv2.NORM_L2))
            labels.append("Eucl_C_D")  
            values.append(cv2.norm(topmost - bottommost, cv2.NORM_L2))

            #Equivalent Diameter is the diameter of the circle whose area is same as the contour area.
            equi_diameter = np.sqrt(4*areaTotal/np.pi)            
            labels.append("EquivalentDiameter")
            values.append(equi_diameter)
            labels.append("AspectRatio")
            values.append(aspect_ratio)
            labels.append("Extent")
            values.append(extent)
            labels.append("Solidity")
            values.append(solidity)
            #print('AREA do contorno=',area)
            labels.append("Area")
            values.append(areaTotal)
            labels.append("Perimeter")
            values.append(perimeterTotal)
            #print('PERIMETRO=',perimeterTotal)
           
            types = ['real'] * len(labels)

            return labels, types, values
if __name__ == "__main__":
    image=cv2.imread("B5_1910_345_30_json_label.png")
    MinorMajorAxis().run(image)
