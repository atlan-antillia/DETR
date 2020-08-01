# DetectionTransformer.py
#
# Copyright (c) 2020 Antillia.com TOSHIYUKI ARAI. ALL RIGHTS RESERVED.
# 
# This DetectionTransformer class is based on the following web site
#
# https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb
#
# 2020/06/13:
#  Updated detect method to take a filters parameter, which can be used  
# to limit classes of detected objects specified by the filters when the class labels are drawn,
# not in a detection process. Of course, this is a very simple way, but far from efficient.

# 2020/06/16:
#  Added detect_all method to DetectionTransformer class.
#  Added FiltersParser class.

# 2020/08/01:
#  Added a procedure to save detected_objects information to a text file.

import os

import glob
import pathlib
import os.path
import traceback
from pathlib import Path
import sys
from DETRdemo import DETRdemo

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from PIL import Image, ImageDraw,  ImageFont

import requests
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T

#2020/08/01
from FiltersParser import *

class DetectionTransformer():
    # Constructor
    def __init__(self, classes=COCO_CLASSES):
        self.classes = classes
        n_classes    = len(self.classes)
        self.detr = DETRdemo(num_classes=n_classes)
        state_dict = torch.hub.load_state_dict_from_url(
              url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
              map_location='cpu', check_hash=True)
        self.detr.load_state_dict(state_dict)
        self.detr.eval();

    #2020/06/16 
    # Detect each image in input_image_dir, and save detected image to output_dir
    def detect_all(self, input_image_dir, output_image_dir, filters):
        
        image_list  = []

        if os.path.isdir(input_image_dir):
          image_list.extend(glob.glob(os.path.join(input_image_dir, "*.png")) )
          image_list.extend(glob.glob(os.path.join(input_image_dir, "*.jpg")) )

        print("image_list {}".format(image_list) )
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)
            
        for image_filename in image_list:
            #image_filename will take images/foo.png
            image_file_path = os.path.abspath(image_filename)
            
            print("filename {}".format(image_file_path))
            
            self.detect(image_file_path, output_image_dir, filters)


    #2020/06/13 Added filters parameter to select objects, which takes a list something 
    # like this [person] to select persons only.
    # 
    def detect(self, image_filepath, output_image_dir, filters):
        
        im = Image.open(image_filepath)
        # If im were "RGBA", convert to "RGB".
        im = im.convert("RGB")

        transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            
        img = transform(im).unsqueeze(0)
        
        # Call detect_objects method
         
        scores, boxes = self.detect_objects(img, self.detr, transform, im.size)

        w, h = im.size
        fsize = int(w/130)
        if fsize <10:
          fsize = 12
        # Draw labels only on the input image
        
        (detected_img, detected_objects) = self.draw_labels(im, scores, boxes, fsize, filters)

        filename_only = self.get_filename_only(image_filepath)
        output_image_filepath = os.path.join(output_image_dir, filename_only)
        
        print("filters {}".format(filters))
        
        if filters is not None:
           parser = FiltersParser(str(filters), self.classes)
           output_image_filepath = parser.get_ouput_filename(image_filepath, output_image_dir) 
           
        print(output_image_filepath)
        print("saved as {}".format(output_image_filepath))
        
        detected_img.save(output_image_filepath)


        # Save detected_objects data to a detected_objects_path file.
        # [(1, 'car', '0.9'), (2, 'person', '0.8'),... ]  
        detected_objects_path = output_image_filepath + '.txt'

        with open(detected_objects_path, mode='w') as f:
          for item in detected_objects:
             line = str(item) + "\n"
             f.write(line)


    def detect_objects(self, img, model, transform, size):
        # mean-std normalize the input image (batch-size: 1)
        # standard PyTorch mean-std input image normalization

        # propagate through the model
        outputs = model(img)

        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]

        #keep = probas.max(-1).values > 0.7
        keep = probas.max(-1).values > 0.5

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'][0, keep], size)
        return probas[keep], bboxes_scaled


    def draw_labels(self, pil_img, prob, boxes, fsize, filters):
        draw = ImageDraw.Draw(pil_img)
        print("Fontsize {}".format(fsize))
        font = ImageFont.truetype("arial", fsize)
        index = 1
        detected_objects = []
        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            cl = p.argmax()
            
            object_class = self.classes[cl]
            text = f'{index} {object_class}: {p[cl]:0.2f}'
            detected_object = f'{index} {object_class} {p[cl]:0.2f}'
            # If filters specified

            if filters != None and isinstance(filters, list) and object_class in filters:
                draw.text((xmin, ymin), text, font=font, fill=(255, 255, 255))
                print(text)
                detected_objects.append(detected_object)
                index += 1
            elif filters == None or len(filters) == 0:
                draw.text((xmin, ymin), text, font=font, fill=(255, 255, 255))
                print(text)
                detected_objects.append(detected_object)

                index += 1
        return (pil_img, detected_objects)

    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)


    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def get_filename_only(self, input_image_filename):

       rpos  = input_image_filename.rfind("/")
       fname = input_image_filename

       if rpos >0:
           fname = input_image_filename[rpos+1:]
       else:
           rpos = input_image_filename.rfind("\\")
           if rpos >0:
              fname = input_image_filename[rpos+1:]
       return fname



def parse_argv(argv):
    #The following img.png is taken from 
    # 'https://user-images.githubusercontent.com/11736571/77320690-099af300-6d37-11ea-9d86-24f14dc2d540.png'
    input_image_path = "./images/img.png"
    output_image_dir = None
    str_filters      = None
    filters          = None
    if len(argv) >= 2:
      input_image_path = argv[1]
        
    if len(argv) >= 3:
      output_image_dir = argv[2]

    if len(argv) == 4:
      # Specify a string like this [person,motorcycle] or "[person,motorcycle]" ,
      str_filters = argv[3]
      filtersParser = FiltersParser(str_filters, COCO_CLASSES)
      filters = filtersParser.get_filters()

                          
    if not os.path.exists(input_image_path):
        print("Not found {}".format(input_image_path))
        raise Exception("Not found {}".format(input_image_path))
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)                  
        
    return (input_image_path, output_image_dir, filters)
                   


#########################################
#
if __name__ == "__main__":
    
    classes = COCO_CLASSES
    
    try:
        # python DetectionTransformer.py input_image_filepath output_jmage_dir filters
         
        (input_image_filepath, output_image_dir, filters, ) = parse_argv(sys.argv)
        print("input_image_filepath {}".format(input_image_filepath))
        print("output_image_dir     {}".format(output_image_dir))
        print("filters              {}".format(filters))
         
        if os.path.isfile(input_image_filepath):
          detr = DetectionTransformer(classes)
            
          detr.detect(input_image_filepath, output_image_dir, filters)

        #2020/06/16 Added the following lines to support image_folders.
        #python DetectionTransformer.py ./input_image_dir/  ./output_image_dir/ [person,car]
        elif os.path.isdir(input_image_filepath):
          input_image_dir  = input_image_filepath
          detr = DetectionTransformer(classes)

          # This is a batch operation of object detection to all images in an input_image_dir. 
          detr.detect_all(input_image_dir, output_image_dir, filters)
        else:
          raise Exception("Unsupported imput_image {}".format(input_image_filepath))

    except:
        traceback.print_exc()
