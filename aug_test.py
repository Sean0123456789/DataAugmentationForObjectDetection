from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

img_dir="/media/user/HDD/Data_processing/lta/data_for_training/model_generated/2020-07-24-Clark_Quay/images/2020-07-24-Clark_Quay_Round1_A_B_Left_Aerolion_000071.jpg"
anno_dir = "/media/user/HDD/Data_processing/lta/data_for_training/model_generated/2020-07-24-Clark_Quay/annotations/2020-07-24-Clark_Quay_Round1_A_B_Left_Aerolion_000071.xml"

def get_category_id():

    return {"Crack": 1.0, "Spalling": 2.0, "Water_seepage": 3.0}

def read_xml(xml):
    boxes = []
    tree = ET.parse(xml)
    root = tree.getroot()
    for object in root.findall("object"):

        class_name = object.find("name").text
        xmin = int(object.find("bndbox").find("xmin").text)
        ymin = int(object.find("bndbox").find("ymin").text)
        xmax = int(object.find("bndbox").find("xmax").text)
        ymax = int(object.find("bndbox").find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax, get_category_id()[class_name]])

    return boxes

# print(read_xml(anno_dir))
bboxes = np.array(read_xml(anno_dir))


img = cv2.imread(img_dir)[:,:,::-1]
# cv2.imshow("show", img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# print(bboxes[:,:4])
print(type(bboxes))
# drawed_image = draw_rect(img, bboxes)
# plt.imshow(drawed_image)
# plt.show()

# transforms = Sequence([RandomHorizontalFlip(1), RandomScale(0.2, diff = True), RandomRotate(10)])
transforms = Sequence([RandomHorizontalFlip(1), RandomTranslate(0.1, diff=True), RandomRotate(10), RandomShear(0.2), RandomScale(0.1, diff = True)])
img, bboxes = transforms(img, bboxes)
print(bboxes)
drawed_image = draw_rect(img, bboxes)
plt.imshow(drawed_image)
plt.show()

