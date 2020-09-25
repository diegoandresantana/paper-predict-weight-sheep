from extractor_sheep.kcurvature import KCURVATURE
from extractor_sheep.image_moments import HuMoments, RawCentralMoments
from extractor_sheep.plot_regionprops import AreaPerimeter,MinorMajorAxis 
from extractor_sheep.chama_distancia import EuclideanDistanceActiveCountor
import numpy as np
from sys import version_info
import cv2
import itertools
import glob
import os
def save_output(relation, classes, labels, types, data, output_file):
        """Save output file in ARFF format.
        
        Parameters
        ----------
        relation : string
            Name of relation.
        classes : list of string
            List of classes names.
        labels : list of string
            List of attributes names.
        types : list of string
            List of attributes types.
        data : list of list of string
            List of instances.
        output_file : string
            Path to output file.
        """
        if version_info[0] >= 3:
                arff = open(output_file, 'wb')
                arff.write(bytes(str("%s %s\n\n" % ('@relation', relation)), 'utf-8'))

                for label, t in zip(labels, types):
                        arff.write(bytes(str("%s %s %s\n" % ('@attribute', label, t)), 'utf-8'))
                
                #arff.write(bytes(str("%s %s {%s}\n\n" % ('@attribute', 'classe', ', '.join(classes))), 'utf-8'))
                arff.write(bytes(str('@data\n\n'), 'utf-8'))

                for instance in data:
                    #print(instance)
                    instance = map(str, instance)
                    line = ",".join(instance)
                    arff.write(bytes(str(line + "\n"), 'utf-8'))

                arff.close()
        else:
                arff = open(output_file, 'wb')

                arff.write("%s %s\n\n" % ('@relation', relation))

                for label, t in zip(labels, types):
                    arff.write("%s %s %s\n" % ('@attribute', label, t))

                #arff.write("%s %s {%s}\n\n" % ('@attribute', 'classe', ', '.join(classes)))

                arff.write('@data\n\n')

                for instance in data:
                    instance = map(str, instance)
                    line = ",".join(instance)
                    arff.write(line + "\n")

                arff.close()

def process(folder):

  list_file= glob.glob(os.path.join(folder+'*','*.png')) 
  print(list_file)
  list_image=[]
  for f in list_file:
    image=cv2.imread(f)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    retval, image = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
    image=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 
    list_image.append(image)
  
  extractors=[AreaPerimeter,KCURVATURE,HuMoments]
  labels=[] # nome dos artributos
  types=[]  # tipos de cada atributo
  values=[] # valores relativos aos atributos
  classes=[] # numero de identificação da ovelha
 
  for i, image in enumerate(list_image):
    array_name=list_file[i].split('/')
    name_file=array_name[len(array_name)-1]
    weight=int(name_file.split('_')[2])/10   #pega o peso do nome do arquivo e divide por 10
    id=name_file.split('_')[0]+'_'+name_file.split('_')[1] # baia_ numero da ovelha
    
    kernel = np.ones((5,5), np.uint8)
    image=cv2.dilate(image, kernel,6)
    #horizontal_img = cv2.flip(image, 0 )
    #vertical_img = cv2.flip( image, 1 )
    #both_img = cv2.flip( image, -1 )
    #classe=""
    if i<1:
       lab,typ, val = [list(itertools.chain.from_iterable(ret)) for ret in zip(*([extractor().run(image.copy()) for extractor in extractors]))]    
	   #peso
       lab.append("Weight")
       typ.append("real")
       val.append(weight)
       #val.append(id)

       #identificador
       #lab.append("sheep_number")
       #typ.append("string")
       #val.append(id)

       #classes.append(id)
       labels.append(lab)
       types.append(typ)
       values.append(val)


       #lab,typ, val = [list(itertools.chain.from_iterable(ret)) for ret in zip(*([extractor().run(horizontal_img.copy()) for extractor in extractors]))] 
       #val.append(weight)
       #values.append(val)

       #lab,typ, val = [list(itertools.chain.from_iterable(ret)) for ret in zip(*([extractor().run(vertical_img.copy()) for extractor in extractors]))] 
       #val.append(weight)
       #values.append(val)

       #lab,typ, val = [list(itertools.chain.from_iterable(ret)) for ret in zip(*([extractor().run(both_img.copy()) for extractor in extractors]))] 
       #val.append(weight)
       #values.append(val)
    else:
       lab,typ, val = [list(itertools.chain.from_iterable(ret)) for ret in zip(*([extractor().run(image.copy()) for extractor in extractors]))] 
       val.append(weight)
       #val.append(id)
       values.append(val)
       #if id in classes:
       #    classe=""
       #else: 
           #classes.append(id)
       #print(val)
       
       #lab,typ, val = [list(itertools.chain.from_iterable(ret)) for ret in zip(*([extractor().run(horizontal_img.copy()) for extractor in extractors]))] 
       #val.append(weight)
       #values.append(val)

       #lab,typ, val = [list(itertools.chain.from_iterable(ret)) for ret in zip(*([extractor().run(vertical_img.copy()) for extractor in extractors]))] 
       #val.append(weight)
       #values.append(val)

       #lab,typ, val = [list(itertools.chain.from_iterable(ret)) for ret in zip(*([extractor().run(both_img.copy()) for extractor in extractors]))] 
       #val.append(weight)
       #values.append(val)

  #print(labels)
  #print(types)
  #print(values)
  save_output("sheep_weight", classes, labels[0], types[0], values, folder+"sheep.arff")


if __name__ == "__main__":
  for i in range(8):
    process("/home/diegopc/Documents/inovisao/testes/fold/f"+str(i+1)+"/train/")
    process("/home/diegopc/Documents/inovisao/testes/fold/f"+str(i+1)+"/val/")
