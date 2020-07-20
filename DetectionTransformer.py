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

import os

import glob
import pathlib
import os.path
import traceback
from pathlib import Path
import sys
from DETRdemo import DETRdemo

from PIL import Image, ImageDraw,  ImageFont

import requests
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T

## coco classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


class DetectionTransformer():
    # Constructor
    def __init__(self):
        self.detr = DETRdemo(num_classes=91)
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
            
            detected_image = self.detect(image_file_path, filters)
        
            parser = FiltersParser(str(filters), CLASSES)
            output_image_filename = parser.get_ouput_filename(image_file_path, output_image_dir)
            print("output_image_filename {}".format(output_image_filename))
            
            detected_image.save(output_image_filename)


    #2020/06/13 Added filters parameter to select objects, which takes a list something 
    # like this [person] to select persons only.
    # 
    def detect(self, image_file_path, filters):
        
        im = Image.open(image_file_path)
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
        
        pil_img = self.draw_labels(im, scores, boxes, fsize, filters)

        return pil_img


    def detect_objects(self, img, model, transform, size):
        # mean-std normalize the input image (batch-size: 1)
        # standard PyTorch mean-std input image normalization

        # propagate through the model
        outputs = model(img)

        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.7

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'][0, keep], size)
        return probas[keep], bboxes_scaled


    def draw_labels(self, pil_img, prob, boxes, fsize, filters):
        draw = ImageDraw.Draw(pil_img)
        print("Fontsize {}".format(fsize))
        font = ImageFont.truetype("arial", fsize)
        index = 1
        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            cl = p.argmax()
            
            object_class = CLASSES[cl]
            text = f'{index} {object_class}: {p[cl]:0.2f}'
            # If filters specified

            if filters != None and isinstance(filters, list) and object_class in filters:
                draw.text((xmin, ymin), text, font=font, fill=(255, 255, 255))
                print(text)
                index += 1
            elif filters == None or len(filters) == 0:
                draw.text((xmin, ymin), text, font=font, fill=(255, 255, 255))
                print(text)
                index += 1
        return pil_img

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



class FiltersParser:

  # Specify a str_filters string like this "[person,motorcycle]" ,

  def __init__(self, str_filters, classes):
      print("FiltersParser {}".format(str_filters))
      
      self.str_filters  = str_filters
      self.classes      = classes
      self.filters  = []

      
  def get_filters(self):
      self.filters = []
      if self.str_filters != None:
          tmp = self.str_filters.strip('[]').split(',')
          if len(tmp) > 0:
              for e in tmp:
                  e = e.lstrip()
                  e = e.rstrip()
                  if e in self.classes :
                    self.filters.append(e)
                  else:
                    print("Invalid label(class)name {}".format(e))
      return self.filters


  def get_ouput_filename(self, input_image_filename, image_out_dir):
        rpos  = input_image_filename.rfind("/")
        fname = input_image_filename
        if rpos >0:
            fname = input_image_filename[rpos+1:]
        else:
            rpos = input_image_filename.rfind("\\")
            if rpos >0:
                fname = input_image_filename[rpos+1:]
          
        #print("Input filename {}".format(fname))
        
        abs_out  = os.path.abspath(image_out_dir)
        if not os.path.exists(abs_out):
            os.makedirs(abs_out)

        filname = ""
        if self.str_filters != None:
            filname = self.str_filters.strip('[]')
            filname = filname.strip("'")
            filname += "_"

        output_image_filename = os.path.join(abs_out, filname + fname)
        return output_image_filename


def parse_argv(argv):
    #The following img.png is taken from 
    # 'https://user-images.githubusercontent.com/11736571/77320690-099af300-6d37-11ea-9d86-24f14dc2d540.png'
    input_image_filename = "./images/img.png"
    str_filters = None
    filters = None
    if len(sys.argv) >= 2:
        input_image_filename = sys.argv[1]

    if len(sys.argv) == 3:
        # Specify a string like this [person,motorcycle] or "[person,motorcycle]" ,
        str_filters = sys.argv[2]
           
    parser = FiltersParser(str_filters, CLASSES) 
    filters = parser.get_filters()
                          
    if not os.path.exists(input_image_filename):
        print("Not found {}".format(input_image_filename))
        raise Exception("Not found {}".format(input_image_filename))
    else:
        output_image_filename = parser.get_ouput_filename(input_image_filename, "detected")
                  
    return (input_image_filename, filters, output_image_filename)
                   

  
  
#########################################
#
if __name__ == "__main__":

    try:
        # input_image_file filters
        if len(sys.argv) == 2 or len(sys.argv) == 3:
            (input_image_filename, filters, output_image_filename) = parse_argv(sys.argv)
            print("input_image_filename {}".format(input_image_filename))
            print("filters {}".format(filters))
          
            detr = DetectionTransformer()
            
            detected_img= detr.detect(input_image_filename, filters)
            
            print("saved as {}".format(output_image_filename))
            
            detected_img.save(output_image_filename)

        #2020/06/16 Added the following lines to suppoer image_folders.
        #python DetectionTransformer.py ./input_images/  ./out_images/ [person,car]
        elif len(sys.argv) == 4:
            input_image_dir  = sys.argv[1]
            output_image_dir = sys.argv[2]
            
            parser = FiltersParser(sys.argv[3], CLASSES)
            
            filters = parser.get_filters()
            
            detr = DetectionTransformer()
            
            # This is a batch operation of object detection to all images in an input_image_dir. 
            detr.detect_all(input_image_dir, output_image_dir, filters)

    except:
        traceback.print_exc()
