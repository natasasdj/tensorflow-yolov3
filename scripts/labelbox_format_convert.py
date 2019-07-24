import json
import os
import sys
from numpy.random import permutation
from random import random, seed
from shutil import copyfile


# input arguments: 
camera_type = sys.argv[1] # R, L or T
assert camera_type in ['R', 'L', 'T']
test_sample_prob = float(sys.argv[2])
assert test_sample_prob>=0 and test_sample_prob<=1
val_sample_prob = float(sys.argv[3])
assert val_sample_prob>=0 and val_sample_prob<=1
random_seed = int(sys.argv[4])
seed(a=random_seed)

'''
camera_type = 'L'
test_sample_prob = 0.2
val_sample_prob = 0.2
random_seed = 15
seed(a=random_seed)
'''
#home_dir = '/home/natasa'
# input and out data directories
home_dir = '/home/IRISAD/natasa.sdj'
#data_dir = os.path.join(home_dir, 'tensorflow-yolov3/robot_dataset')
data_dir = os.path.join(home_dir, 'tensorflow-yolov3/robot_dataset_2')
input_dir = os.path.join(data_dir, 'camera_' + camera_type)

train_dir = os.path.join(data_dir, 'train')
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

test_dir = os.path.join(data_dir, 'test')
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

val_dir = os.path.join(data_dir, 'val')
if not os.path.exists(val_dir):
    os.makedirs(val_dir)


# read json file exported from labelbox
filename = os.path.join(data_dir, 'camera_' + camera_type + '.json')
with open(filename) as f:
    data = json.load(f)


# object.names is a file that contains object class names (each line = one class name), 
# if a class name contains a space it must  be replaced with underscore (e.g. 'washing machine' with 'washing_machine')
 
with open(os.path.join(data_dir, 'object.names')) as f:
    object_names = f.read().strip().split('\n')

# make a dictionary of object names, where each name correspond to an integer string
object_dict = {}
for i in range(len(object_names)):
    object_dict[object_names[i]] = str(i)


# shuffle data randomly    
data = permutation(data)

no_test = round(len(data)*test_sample_prob)
no_val = round(len(data)*val_sample_prob)

# with aprobab
for i in range(len(data)):    
    #i = 0
    #print(i)    
    image_name = data[i]['External ID']
    split = image_name.split('.')
    image_name_ = split[0] + '_' + camera_type + '.' + split[1]
    image_path = os.path.join(input_dir, image_name)

    #print(image_path)
    if i < no_test:
        output_dir = test_dir
    elif i < no_test + no_val:
        output_dir = val_dir
    else:
        output_dir = train_dir

    #output_dir = train_dir if  i < no_train else test_dir
    new_image_path = os.path.join(output_dir, image_name_)
    copyfile(image_path, new_image_path)
     
    objects = data[i]['Label']
    classes = objects.keys()
    line = new_image_path 
    for class_ in classes:
        # class_ = classes[0]
        class_int = object_dict[class_.replace(" ", "_")]
        objects_ = objects[class_]
        for object_ in objects_:
            #print(object_)
            x_min = min([object_['geometry'][0]['x'], object_['geometry'][1]['x'], object_['geometry'][2]['x'], object_['geometry'][3]['x']])
            y_min = min([object_['geometry'][0]['y'], object_['geometry'][1]['y'], object_['geometry'][2]['y'], object_['geometry'][3]['y']])
            x_max = max([object_['geometry'][0]['x'], object_['geometry'][1]['x'], object_['geometry'][2]['x'], object_['geometry'][3]['x']])
            y_max = max([object_['geometry'][0]['y'], object_['geometry'][1]['y'], object_['geometry'][2]['y'], object_['geometry'][3]['y']])
            s = str(x_min) + ',' + str(y_min) + ','  + str(x_max) + ',' + str(y_max) + ',' + class_int
            line = line + ' ' + s

    line = line + '\n'
    #print(line)
    annot_file = os.path.join(output_dir, 'annotation.txt')
    with open(annot_file, 'a+') as f:
        f.write(line)






