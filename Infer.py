import keras
import tensorflow as tf
from object_detector_retinanet.keras_retinanet import models
from object_detector_retinanet.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from object_detector_retinanet.keras_retinanet.utils.visualization import draw_box, draw_caption
from object_detector_retinanet.keras_retinanet.utils.colors import label_color
from object_detector_retinanet.keras_retinanet.utils import EmMerger
from object_detector_retinanet.utils import create_folder, root_dir


# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
from tqdm import tqdm_notebook
import argparse
import shutil
from sys import platform
import warnings
warnings.filterwarnings('ignore')

def infer(weights, image_dir, labels_to_name, output_file, threshold, hard_score_rate):
    model = models.load_model(weights, backbone_name='resnet50', convert = 1, nms=False)
    labels_to_names = labels_to_name
    csv_data_lst = []
    csv_data_lst.append(['image_id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'hard_score'])
    threshold = threshold
    hard_score_rate = hard_score_rate
    max_detections = 9999
    image_dir = image_dir
    images = os.listdir(image_dir)
    if os.path.exists(str(output_file) + '/' + 'images'):
        shutil.rmtree(str(output_file) + '/' + 'images')
    os.makedirs(str(output_file) + '/' + 'images')
    
    # Run inference
    t0 = time.time()
    for img in images:
        image_path = image_dir + '/' + img
        t = time.time()
        image = read_image_bgr(image_path)
    #     patches = image.unfold(0, 128, 128).unfold(1, 128, 128).unfold(2, 3, 3)
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        image, scale = resize_image(image)
        boxes, hard_scores, labels, soft_scores = model.predict_on_batch(np.expand_dims(image, axis=0))
        soft_scores = np.squeeze(soft_scores, axis=-1)
        soft_scores = hard_score_rate * hard_scores + (1 - hard_score_rate) * soft_scores
        boxes /= scale
        indices = np.where(hard_scores[0, :] > threshold)[0]
        scores = soft_scores[0][indices]
        hard_scores = hard_scores[0][indices]
        scores_sort = np.argsort(-scores)[:max_detections]
        image_boxes = boxes[0, indices[scores_sort], :]
        image_scores = scores[scores_sort]
        image_hard_scores = hard_scores[scores_sort]
        image_labels = labels[0, indices[scores_sort]]
        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
        results = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_hard_scores, axis=1),
             np.expand_dims(image_labels, axis=1)], axis=1)
        filtered_data = EmMerger.merge_detections(image_path, results)
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        for ind, detection in filtered_data.iterrows():
            box = np.asarray([detection['x1'], detection['y1'], detection['x2'], detection['y2']])
            filtered_boxes.append(box)
            filtered_scores.append(detection['confidence'])
            filtered_labels.append('{0:.2f}'.format(detection['hard_score']))
            row = [image_path, detection['x1'], detection['y1'], detection['x2'], detection['y2'],
                   detection['confidence'], detection['hard_score']]
            csv_data_lst.append(row)

        for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
            # scores are sorted so we can break
            if score < threshold:
                break

            color = [31, 0, 255]  #change the length of color array here based on the object classes, here I have hardcoded!

            b = box.astype(int)
            draw_box(draw, b, color=color)

            caption = "{} {:.3f}".format(labels_to_names[0], score) #hardcoded the index in the dictionary
            draw_caption(draw, b, caption)

        plt.figure(figsize=(20, 20))
        plt.axis('off')
        plt.imshow(draw)
        plt.savefig(str(output_file) + '/' + 'images' + '/' + str(img.split('.')[0]) + '.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/root/Documents/SKU110K/iou_resnet50_csv_06.h5', help='path to weights file')
    parser.add_argument('--images', type=str, default='samples', help='path to images')
    parser.add_argument('--labels_map', type=dict, default={0: 'object'}, help='dictionary with keys as index and values as objects')
    parser.add_argument('--output_file', type=str, default='output')
    parser.add_argument('--threshold', type=float, default=0.05, help='value of threshold')
    parser.add_argument('--hard_score_rate', type=float, default=0.5, help='ratio of hard_score:soft_score')
    opt = parser.parse_args()
    print(opt)
    
    # set tf backend to allow memory to grow, instead of claiming everything
    def get_session():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(get_session())
    
    infer(opt.weights,
          opt.images,
          opt.labels_map,
          opt.output_file,
          opt.threshold,
          opt.hard_score_rate)
                        