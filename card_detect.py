import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

import albumentations as A

from pycocotools.coco import COCO #.coco.COCO

class CardDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        
        self.img_dim = preproc.img_dim
        self.rgb_means = preproc.rgb_means
        
        self.coco = COCO(txt_path)
        self.list_annotations = self.coco.getAnnIds()
        
        
        
        
        #f = open(txt_path,'r')
        #lines = f.readlines()
        #isFirst = True
        #labels = []
        #for line in lines:
            #line = line.rstrip()
            #if line.startswith('#'):
                #if isFirst is True:
                    #isFirst = False
                #else:
                    #labels_copy = labels.copy()
                    #self.words.append(labels_copy)
                    #labels.clear()
                #path = line[2:]
                #path = txt_path.replace('label.txt','images/') + path
                #self.imgs_path.append(path)
            #else:
                #line = line.split(' ')
                #label = [float(x) for x in line]
                #labels.append(label)

        #self.words.append(labels)

    def __len__(self):
        return len(self.list_annotations)

    def __getitem__(self, index):
        
        
        #print(index, len(self.list_annotations))
        
        ann = self.list_annotations[index]
        num_ann = self.coco.loadAnns([ann])
        
        first_ann = num_ann[0]
        keypoints = first_ann['keypoints']
    
    
        point1_x = int(keypoints[0])
        point1_y = int(keypoints[1])
    
        point2_x = int(keypoints[3])
        point2_y = int(keypoints[4])
    
        point3_x = int(keypoints[6])
        point3_y = int(keypoints[7])
    
        point4_x = int(keypoints[9])
        point4_y = int(keypoints[10])
    
    
        xy_keypoints = [
              (point1_x, point1_y),
              (point2_x, point2_y),
              (point3_x, point3_y),
              (point4_x, point4_y),
          ]      
    
    
        class_sides = [
              'left_up',
              'right_up',
              'right_down',
              'left_down',
          ]      
    
        bbox = first_ann['bbox']
        image_id = first_ann['image_id']
    
        #min_bbox_image = int(max(bbox[2],bbox[3]))


        class_labels = ['card']
    
        num_image = self.coco.loadImgs([image_id])
        first_image = num_image[0]
        #image_file_name = first_image['file_name']
        image_path = first_image['path']
    
    
        if image_path.find('test') > -1:
            image_path = image_path.replace('/datasets','data/coco-card-dataset/test')
        else:
            image_path = image_path.replace('/datasets','data/coco-card-dataset/train')
    
        list_bbox = []
        list_bbox.append(bbox)

        #Визуализация до
        img = cv2.imread(image_path)
        
        image_size = self.img_dim

        real_h,real_w,_ = img.shape
        #img_ratio = real_w/real_h
    
        max_h = min(real_h,real_w)
    
        min_h = min(real_w + 10,max_h)
    
        transform = A.Compose([
              A.ShiftScaleRotate(always_apply=False, p=1.0, shift_limit=(-0.17, 0.17), scale_limit=(-0.60, 0.10), rotate_limit=(-22, 22), interpolation=cv2.INTER_AREA, border_mode=1, value=(0, 0, 0), mask_value=None),
              A.RandomSizedCrop(always_apply=False, p=1.0, min_max_height=(min_h, max_h), height=image_size, width=image_size, interpolation=cv2.INTER_AREA),
              ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'])
                                , keypoint_params=A.KeypointParams(format='xy', remove_invisible = False, label_fields=['class_sides']))      
    
        image, res_boxes, res_keypoints = self.image_transform_save_bbox(transform, image_size, img, class_labels, list_bbox, xy_keypoints, class_sides)

        
        #img = cv2.imread(self.imgs_path[index])
        height, width, _ = image.shape

        #labels = self.words[index]
        annotations = np.zeros((0, 13))
        
        #print(15)
        
        if len(res_boxes) == 0:
            return annotations
        
        if len(res_keypoints) == 0:
            return annotations
        
        #for idx, label in enumerate(labels):

        boxes = res_boxes[0]
        
        annotation = np.zeros((1, 13))
        # bbox
        annotation[0, 0] = int(boxes[0])  # x1
        annotation[0, 1] = int(boxes[1])  # y1
        annotation[0, 2] = int(boxes[0]) + int(boxes[2])  # x2
        annotation[0, 3] = int(boxes[1]) + int(boxes[3])  # y2


        boxes_t = np.zeros((1, 4))
        # bbox
        boxes_t[0, 0] = int(boxes[0])  # x1
        boxes_t[0, 1] = int(boxes[1])  # y1
        boxes_t[0, 2] = int(boxes[0]) + int(boxes[2])  # x2
        boxes_t[0, 3] = int(boxes[1]) + int(boxes[3])  # y2


        # landmarks
        
        x1, y1 = res_keypoints[0]
        x1 = int(x1)
        y1 = int(y1)

        x2, y2 = res_keypoints[1]
        x2 = int(x2)
        y2 = int(y2)

        x3, y3 = res_keypoints[2]
        x3 = int(x3)
        y3 = int(y3)

        x4, y4 = res_keypoints[3]
        x4 = int(x4)
        y4 = int(y4)

        
        annotation[0, 4] = x1    # l0_x
        annotation[0, 5] = y1    # l0_y
        annotation[0, 6] = x2    # l1_x
        annotation[0, 7] = y2    # l1_y
        annotation[0, 8] = x3   # l2_x
        annotation[0, 9] = y3   # l2_y
        annotation[0, 10] = x4  # l3_x
        annotation[0, 11] = y4  # l3_y
        
        keypoints_t = np.zeros((1, 8))
        keypoints_t[0, 0] = x1    # l0_x
        keypoints_t[0, 1] = y1    # l0_y
        keypoints_t[0, 2] = x2    # l1_x
        keypoints_t[0, 3] = y2    # l1_y
        keypoints_t[0, 4] = x3   # l2_x
        keypoints_t[0, 5] = y3   # l2_y
        keypoints_t[0, 6] = x4  # l3_x
        keypoints_t[0, 7] = y4  # l3_y
        
        
        
        #if (annotation[0, 4]<0):
            #annotation[0, 12] = -1
        #else:
 
        annotation[0, 12] = 1
        
        labels_t = np.ones((1, 1))
        #labels_t = np.expand_dims(labels_t, 1)
        
        height, width, _ = image.shape
        
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height

        keypoints_t[:, 0::2] /= width
        keypoints_t[:, 1::2] /= height   
        
        targets_t = np.hstack((boxes_t, keypoints_t, labels_t))

        annotations = np.append(annotations, annotation, axis=0)
            
        target = np.array(annotations)
        
        if self.preproc is not None:
            
            #image, target = self.preproc(image, target)
            
            #my preproc
            #height, width, _ = image.shape
            
            image = image.astype(np.float32)
            image -= self.rgb_means
            image = image.transpose(2, 0, 1)
            
            #target[:, 0::2] /= width
            #target[:, 1::2] /= height
            
            a = 1

        #target = target.reshape(13)

        return torch.from_numpy(image), targets_t


    def image_transform_save_bbox(self, transform, image_size, image, class_labels, bbox_list, keypoints_list, class_sides, loop = True):
    
        save_boxes = True
    
        while True:
    
            # Augment an image
            transformed = transform(image=image, bboxes = bbox_list, keypoints=keypoints_list, class_labels=class_labels, class_sides = class_sides)
            transformed_image = transformed["image"]    
            transformed_bboxes = transformed["bboxes"] 
            transformed_keypoints = transformed['keypoints']
            #transformed_class_labels = transformed['class_labels']
    
            max_y, max_x, _ = transformed_image.shape
    
            #Проверим, что количество боксов сохранилось после
            if len(bbox_list) != len(transformed_bboxes):
                if loop:
                    continue
                else:
                    save_boxes = False
            else:
                to_continue = False
                for bbox in transformed_bboxes:
                    new_x = bbox[0]
                    new_y = bbox[1]
                    if round(new_x) == 0 or round(new_y) == 0:
                        to_continue = True
                        break
                    elif round(new_x + bbox[2]) >= max_x - 1 or round(new_y + bbox[3]) >= max_y - 1:
                        to_continue = True
                        break
                if to_continue == False:
                    break
                else:
                    if loop:
                        continue
                    else:
                        save_boxes = False
    
            #break
    
        if save_boxes:
            pass
    
        return transformed_image, transformed_bboxes, transformed_keypoints
