
<h1>
DetectionTransformer
</h1>

<h4>
This DetectionTransformer class is based on the following web site
 https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb
</h4>
<br>
See also:<a href='https://github.com/facebookresearch/detr'>DETR: End-to-End Object Detection with Transformers</a>
<br>
<br>
We have installed torch and torchvision in the following way:<br><br>
 
pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html<br>
<br>
On PyTorch installation see :<a href="https://pytorch.org/resources/">Resources | PyTorch</a> 
<br>
<br>
Please run the following script to detect objects in an image file:<br>

<pre>
>python DetectionTransformer.py image_file_or_dir output_image_dir [filters]
</pre>
If image_file_or_dir were a single image file, 
the commnand above will generate a detected_image_file, detected_objects_csv_file, and objects_stats_csv_file in output_image_dir, respectively.
<br>
If image_file_or_dir were a diretory, the simlar process  will be applied to each image file (png, jpg).<br>
<br>

The optional <i>filters</i> parameter is a list of classes to be selected from the detected objects in a post-processing stage
after a detection process.<br>
 To specify the classes to be selected in the post-processing stage, we use the list format like this.
<pre>
  [class1, class2,.. classN]
</pre>

<br>
<b>Example 1:</b><br>

<font size=2>
<pre>
python DetectionTransformer.py images/img.png detected
</pre>
</font>
<br>
<img src="./detected/img.png">
<br>
<br>
detected_objects<br>
<img src="./detected/img.png.csv.png">

<br>
<br>
objects stats_csv<br>
<img src="./detected/img.png_stats.csv.png">

<br><br>
<b>Example 2:</b><br>
<font size=2>
<pre>
python DetectionTransformer.py images/ShinJuku.jpg detected
</pre>
</font>
<br><br>
<img src="./detected/ShinJuku.jpg">


<br><br>
<b>Example 3:</b><br>

<font size=2>
<pre>
python DetectionTransformer.py images/ShinJuku2.jpg detected
</pre>
</font>
<br><br>
<img src="./detected/ShinJuku2.jpg">

<br><br>
<b>Example 4:</b><br>

<font size=2>
<pre>
python DetectionTransformer images/Takashimaya2.jpg detected
</pre>
</font>
<br><br>
<img src="./detected/Takashimaya2.jpg">
<br><br>

<b>Example 5:</b><br>
<b>
 Let's apply filters to draw matched labels specified by the filters on the input image.
<br>
</b>
<font size=2>
<pre>
python DetectionTransformer.py images/img.png detected [person,car]
</pre>
In this case, the objects of <i>person</i> or <i>car</i> will be selected from the detected objects found in <i>images/img.png</i>.
</font>
<br><br>
<img src="./detected/person_car_img.png">
<br><br>

<img src="./detected/person_car_img.png.csv.png">

<br><br>
<img src="./detected/person_car_img.png_stats.csv.png">

<br><br>

<b>Example 6:</b><br>
<b>
 You can specify input_image_dir, output_image_dir in the following way.<br><br>
</b>
<font size=2>
<pre>
python DetectionTransformer.py images detected [person]
</pre>
By using the filter "[person]", you can count the number of persons in each image of the images directory.<br> 

</font>
<br>
<img src="./detected/person_img.png">
<br><br>

<img src="./detected/all_person_img.png.txt.png">

