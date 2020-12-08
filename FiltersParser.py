#******************************************************************************
#
#  Copyright (c) 2020-2021 Antillia.com TOSHIYUKI ARAI. ALL RIGHTS RESERVED.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#******************************************************************************

# FiltersParser.py

import os

## coco classes
COCO_CLASSES = [
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


class FiltersParser:

  # Specify a str_filters string like this "[person,motorcycle]" ,

  def __init__(self, str_filters, classes=COCO_CLASSES):
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
            filname = self.str_filters.strip("[]").replace("'", "").replace(", ", "_")
            filname += "_"

        output_image_filename = os.path.join(abs_out, filname + fname)
        return output_image_filename

