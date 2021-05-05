from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from lxml import etree
import codecs
import copy

def get_classname():

    return {"1.0": "Crack", "2.0":"Spalling", "3.0":"Water_seepage", "4.0":"Ground_water_seepage"}


def get_category_id():

    return {"Crack": 1.0, "Spalling": 2.0, "Water_seepage": 3.0, "Ground_water_seepage":4.0}


def save_anno(anno_save_dir, name_xml, tree_xml):

    # tree_xml.find("filename").text = name_xml[:-4]+".jpg"
    prettifyResult = prettify(tree_xml)

    if not os.path.exists(anno_save_dir):
        os.makedirs(anno_save_dir)

    out_file = codecs.open(os.path.join(anno_save_dir, name_xml), 'w', encoding='utf-8')
    out_file.write(prettifyResult.decode('utf8'))
    out_file.close()

def prettify(elem):
    """
        Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf8')
    root = etree.fromstring(rough_string)

    return etree.tostring(root, pretty_print=True, encoding='utf-8').replace("  ".encode(), "\t".encode())


def generate_xml_template(IMG_SIZE=None, FOLDER_NAME=None, IMG_NAME=None, IMG_PATH="None", IMG_CHANNEL=3):

    # Check conditions
    if IMG_NAME is None or \
            FOLDER_NAME is None or \
            IMG_PATH is None:
        return None
    # print("=================================")
    top = ET.Element('annotation')

    folder = ET.SubElement(top, 'folder')
    folder.text = FOLDER_NAME

    filename = ET.SubElement(top, 'filename')
    filename.text = IMG_NAME

    if IMG_PATH is not None:
        localImgPath = ET.SubElement(top, 'path')
        localImgPath.text = "localImgPath"

    source = ET.SubElement(top, 'source')
    database = ET.SubElement(source, 'database')
    database.text = "databaseSrc"

    size_part = ET.SubElement(top, 'size')
    width = ET.SubElement(size_part, 'width')
    height = ET.SubElement(size_part, 'height')
    depth = ET.SubElement(size_part, 'depth')
    width.text = str(IMG_SIZE[1])
    height.text = str(IMG_SIZE[0])
    depth.text = str(IMG_CHANNEL)

    segmented = ET.SubElement(top, 'segmented')
    segmented.text = '0'
    return top

def appendObjects(tree_xml, bboxes, root_save_dir, name_xml, IMG_SIZE):
    # print(len(bboxes))
    # print(tree_xml)

    for box in bboxes:
        # class_name = get_classname()[str(box[0])]
        # ymin = box[1]
        # xmin = box[2]
        # ymax = box[3]
        # xmax = box[4]
        class_name = get_classname()[str(box[4])]
        ymin = box[1]
        xmin = box[0]
        ymax = box[3]
        xmax = box[2]

        obj = ET.SubElement(tree_xml, 'object')
        obj_name = ET.SubElement(obj, 'name')
        obj_pose =ET.SubElement(obj, 'pose')
        obj_pose.text = 'Unspecified'

        truncated = ET.SubElement(obj, 'truncated')
        if int(float(ymax)) == int(float(IMG_SIZE[0])) or (int(float(ymin)) == 1):
            truncated.text = "1"  # max == height or min
        elif (int(float(xmax)) == int(float(IMG_SIZE[1]))) or (int(float(xmin)) == 1):
            truncated.text = "1"  # max == width or min
        else:
            truncated.text = "0"
        is_difficulty = ET.SubElement(obj, 'difficult')
        is_difficulty.text =str(0)

        obj_bndbox = ET.SubElement(obj, 'bndbox')
        bndbox_xmin = ET.SubElement(obj_bndbox, 'xmin')
        bndbox_ymin = ET.SubElement(obj_bndbox, 'ymin')
        bndbox_xmax = ET.SubElement(obj_bndbox, 'xmax')
        bndbox_ymax = ET.SubElement(obj_bndbox, 'ymax')

        obj_name.text = class_name
        bndbox_xmin.text = str(int(xmin))
        bndbox_ymin.text = str(int(ymin))
        bndbox_xmax.text = str(int(xmax))
        bndbox_ymax.text = str(int(ymax))

    save_anno(root_save_dir, name_xml, tree_xml)

    if 0:
        out_file = None
        prettifyResult =prettify(tree_xml)
        out_file = codecs.open("/media/user/HDD/Data_processing/lta/data_for_training/20191128/test_%d.xml"%i, 'w', encoding='utf-8')
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()

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

def data_augment(image, bboxes, image_name, root_save, aug_times=3):

    transforms = Sequence([RandomHorizontalFlip(1), RandomTranslate(0.1, diff=True), RandomRotate(10), RandomShear(0.1),
                           RandomScale(0.1, diff=True)])

    for i in range(aug_times):
        try:
            image_name1 = image_name[:-4] + '_%d'%i +image_name[-4:]
            # print(image_name1)

            img, bboxes_t = transforms(copy.deepcopy(image), copy.deepcopy(bboxes))
            # drawed_image = draw_rect(img, bboxes_t)
            # plt.imshow(drawed_image)
            # plt.show()
            # plt.close()
            if 1:
                xml_tree = generate_xml_template(IMG_SIZE=img.shape, FOLDER_NAME="JPEGImages", IMG_NAME=image_name1)

                anno_save_dir = os.path.join(root_save, "Annotations")

                appendObjects(copy.deepcopy(xml_tree), bboxes_t, anno_save_dir, name_xml=image_name1[:-4] + ".xml",
                              IMG_SIZE=img.shape)

                img_save_dir = os.path.join(root_save, "JPEGImages", image_name1)
                cv2.imwrite(img_save_dir, img[:, :, ::-1])

        except:
            print(image_name)

if __name__ == '__main__':

    if 0:
        img_dir = "/media/user/HDD/Data_processing/lta/data_for_training/model_generated/2020-07-24-Clark_Quay/images/2020-07-24-Clark_Quay_Round1_A_B_Left_Aerolion_000071.jpg"
        anno_dir = "/media/user/HDD/Data_processing/lta/data_for_training/model_generated/2020-07-24-Clark_Quay/annotations/2020-07-24-Clark_Quay_Round1_A_B_Left_Aerolion_000071.xml"

        bboxes = np.array(read_xml(anno_dir))
        img = cv2.imread(img_dir)[:, :, ::-1]
        image_name = "Left_Aerolion_000071.jpg"
        root_save = "/media/user/HDD/others_temp/test"
        data_augment(img, bboxes, image_name, root_save, aug_times=5)

    if 1:
        # root_dir = "/media/user/HDD/Data_processing/lta/data_for_training/model_generated"
        # root_dir = "/home/user/Training_data/4classes"
        # dataset_list = os.listdir(root_dir)

        # for dataset in dataset_list:
        #     print(dataset)

        # img_folder_dir = os.path.join(root_dir, dataset, "val", "data", "JPEGImages")
        # anno_folder_dir = os.path.join(root_dir, dataset, "train", "data", "Annotations")

        img_folder_dir = "/home/user/Training_data/3classes/training/2021-04-24-Little_India/train/data/JPEGImages"
        anno_folder_dir = "/home/user/Training_data/3classes/training/2021-04-24-Little_India/train/data/Annotations"

        # save_root_dir = os.path.join(root_dir, dataset, "train", "data_aug")
        save_root_dir = "/home/user/Training_data/3classes/training/2021-04-24-Little_India/train/data_aug"
        img_save_root = os.path.join(save_root_dir, "JPEGImages")
        anno_save_root = os.path.join(save_root_dir, "Annotations")

        if not os.path.isdir(img_save_root):
            os.makedirs(img_save_root)

        if not os.path.isdir(anno_save_root):
            os.makedirs(anno_save_root)

        img_list = os.listdir(img_folder_dir)

        for img_name in img_list:
            img_dir = os.path.join(img_folder_dir, img_name)
            anno_dir = os.path.join(anno_folder_dir, img_name[:-4]+ ".xml")

            bboxes = np.array(read_xml(anno_dir))
            img = cv2.imread(img_dir)[:, :, ::-1]
            data_augment(img, bboxes, img_name, save_root_dir, aug_times=3)


    if 0:
        root_dir = "/home/user/Training_data/LTA/Train/data_supplement"
        dataset_list = os.listdir(root_dir)

        for dataset in dataset_list:
            print(dataset)

            img_folder_dir = os.path.join(root_dir, dataset, "JPEGImages")
            anno_folder_dir = os.path.join(root_dir, dataset, "Annotations")

            save_root_dir = os.path.join(root_dir, dataset, "aug")
            img_save_root = os.path.join(save_root_dir, "JPEGImages")
            anno_save_root = os.path.join(save_root_dir, "Annotations")

            if not os.path.isdir(img_save_root):
                os.makedirs(img_save_root)

            if not os.path.isdir(anno_save_root):
                os.makedirs(anno_save_root)

            img_list = os.listdir(img_folder_dir)

            for img_name in img_list:
                img_dir = os.path.join(img_folder_dir, img_name)
                anno_dir = os.path.join(anno_folder_dir, img_name[:-4]+ ".xml")

                bboxes = np.array(read_xml(anno_dir))
                img = cv2.imread(img_dir)[:, :, ::-1]
                data_augment(img, bboxes, img_name, save_root_dir, aug_times=3)










