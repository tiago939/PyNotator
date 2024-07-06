import cv2
import numpy as np
import sys
import os
from os import listdir
from os.path import isfile, join
import torch
import json
import yaml
from ultralytics import YOLO
import argparse

from load_model import YOLOModelLoader
from train_model import train_yolo_model

dir_path = './data/images/'
#video_name = 'video_240622_3'
#dir_path = './dataset/videos/' + video_name + '.mp4'

if not os.path.exists('./data/dataset/images'):
    os.makedirs('./data/dataset/images')
if not os.path.exists('./data/dataset/labels'):
    os.makedirs('./data/dataset/labels')

# Create an instance of YOLOModelLoader
loader = YOLOModelLoader(model_path='yolov10n.pt')

# Load the default pre-trained model
model = loader.load_model(use_custom=False)
print("Default model loaded successfully.")

# Load model classes
model_classes = model.names

# Load YAML classes
with open("./data/data.yaml", 'r') as file:
    dataset_config = yaml.safe_load(file)
yaml_classes = dataset_config['names']

# Create a mapping from model class IDs to YAML class IDs
model_to_yaml_class_id = {model_id: yaml_classes.index(name) for model_id, name in model_classes.items() if name in yaml_classes}

names_color = [(255,0,0) for _ in range(len(yaml_classes))]

mouse_coordinates = (0,0)
pointA = (0,0)
pointB = (1,1)
on_click = False
drawing = False
last_ptA = None
last_ptB = None
scale = 0.8
mouse_move = False
buttons = []

def mouse_drawing(event, x, y, flags, params):
    global mouse_coordinates, on_click, pointA, pointB, drawing, mouse_move
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_coordinates = (x,y)
        if drawing is True:
           pointB = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pointA = (x, y)
        on_click = True
        mouse_move = True
    else:
        on_click = False

    if event == cv2.EVENT_LBUTTONUP:
        mouse_move = False
        drawing = False
        pointB = (x, y)

def select_bb(bb_list, point, scale):
    norm_list = np.zeros(len(bb_list))
    for k in range(len(bb_list)):
        bb = bb_list[k]
        x1,y1,x2,y2 = int(scale*bb['xmin']), int(scale*bb['ymin']), int(scale*bb['xmax']), int(scale*bb['ymax'])
        cx = (x2+x1)/2
        cy = (y2+y1)/2
        norm_list[k] = (cx-point[0])**2.0 + (cy-point[1])**2.0
    least_index = np.argmin(norm_list)
    bb = bb_list[least_index]
    return least_index, bb

def move_bb(bb, point, scale):
     point = (point[0]/scale, point[1]/scale)
     w = bb['xmax'] - bb['xmin']
     h = bb['ymax'] - bb['ymin']
     bb_new = [(int((point[0]-w/2)), int((point[1]-h/2))), (int((point[0]+w/2)), int((point[1]+h/2)))]
     return bb_new

def predict(img):
    # Check if model classes match YAML classes
    if not set(yaml_classes).issubset(set(model_classes.values())):
        raise ValueError("Model classes do not include all YAML classes.")

    # Make predictions
    results = model(img)
    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                model_class_id = int(box.cls)
                confidence = box.conf[0].item()

                # Filter detections to include only classes from YAML file and map to YAML class IDs
                if model_class_id in model_to_yaml_class_id:
                    yaml_class_id = model_to_yaml_class_id[model_class_id]
                    detections.append([x1, y1, x2, y2, yaml_class_id, confidence])

    return detections

def train(model):
    print("Converting labels to txt format...")
    os.system("python json2txt.py 1")
    print("Preparing dataset...")
    os.system("python prepare_dataset.py 1")
    train_yolo_model(model)

class inputHandler():
    def __init__(self, path):
        self.counter = 0
        self.sourceType = path[-3:]
        self.im = []
        if self.sourceType == 'mp4':
            self.mode = 'video'
            self.source = cv2.VideoCapture(path)
            self.length = int(self.source.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self.mode = 'image'
            images_path = [f for f in listdir(path) if isfile(join(path, f))]
            #images_path = sorted( images_path, key=lambda a: int(a.split("_")[1][:-4]) )
            images_path = [path + img for img in images_path]
            self.source = images_path
            self.length = len(self.source)

    def get(self):
        if self.mode == 'video':
            if len(self.im) == 0:
                status, self.im = self.source.read()
            else:
                self.im = self.im

        if self.mode == 'image':
            self.im = cv2.imread(self.source[self.counter])

        return self.im

    def increase(self):
        if self.mode == 'video':
            status, self.im = self.source.read()

        self.counter += 1
        if self.counter < 0:
            self.counter = self.length - 1
        if self.counter > self.length - 1:
            self.counter = 0

    def decrease(self):
        self.counter -= 1
        if self.counter < 0:
            self.counter = self.length - 1
        if self.counter > self.length - 1:
            self.counter = 0

class Button():

    def __init__(self, x, y, w, h, text, click_color=(0,255,0)):
        buttons.append(self)
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.color = (0,255,0)
        self.click = False
        self.last_click = False
        self.persistent_click = False
        self.click_color = click_color

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        textSize = cv2.getTextSize(self.text, self.font, 0.75, 2)[0]
        self.textX = self.x + int(abs(self.w - textSize[0])/2)
        self.textY = self.y + int(1.333*abs(self.h - textSize[1])/2)

    def draw(self, image, mc):
        mc_x = mc[0]
        mc_y = mc[1]
        if self.x < mc_x < self.x+self.w and self.y < mc_y < self.y+self.h:
            if on_click == False:
                c = (200, 0, 0)
                self.click = False
                self.last_click = False
            else:
                c = self.click_color
                self.click = True
                if self.last_click == False:
                    self.last_click = True
                    self.persistent_click = not self.persistent_click
                elif self.last_click == True:
                    self.click = False
        else:
            c = (255,0,0)
            self.click = False
            self.last_click = False

        cv2.rectangle(image,(self.x,self.y),(self.x+self.w,self.y+self.h),c,-1)
        cv2.rectangle(image,(self.x,self.y),(self.x+self.w,self.y+self.h),(0,0,0),2)
        cv2.putText(image, '%s' % self.text, (self.textX, self.textY), self.font, 0.75, (0,0,0), 2, cv2.LINE_AA)

class Dropdown():

    def __init__(self, x, y, w, h, text):
        buttons.append(self)
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.color = (0,255,0)
        self.click = False
        self.last_click = False
        self.drop_click = False
        self.labels = yaml_classes
        self.labels_click = [False for _ in range(len(self.labels))]
        self.labels_last_click = [False for _ in range(len(self.labels))]

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        textSize = cv2.getTextSize(self.text, self.font, 0.75, 2)[0]
        self.textX = self.x + int(abs(self.w - textSize[0])/2)
        self.textY = self.y + int(1.75*abs(self.h - textSize[1])/2)

    def draw(self, image, mc):
        mc_x = mc[0]
        mc_y = mc[1]
        if self.x < mc_x < self.x+self.w and self.y < mc_y < self.y+self.h:
            if on_click == False:
                c = (200,0,0)
                self.click = False
                self.last_click = False
            else:
                c = (0,255,0)
                self.click = True
                if self.last_click == False:
                    self.last_click = True
                    self.drop_click = not self.drop_click
                elif self.last_click == True:
                    self.click = False

        else:
            c = (255,0,0)
            self.click = False
            self.last_click = False

        if self.drop_click == True:
            labels_color = [(255, 0, 0) for _ in range(len(self.labels))]
            for i in range(len(self.labels)):
                l = 2+i
                x1 = self.x
                y1 = self.y+(l-1)*self.h+5
                x2 = self.x+self.w
                y2 = self.y+l*self.h+5

                if x1 < mc_x < x2 and y1 < mc_y < y2:
                    if on_click == True:
                        labels_color[i] = (0, 255, 0)
                        self.labels_click[i] = True
                        if self.labels_last_click[i] == False:
                            self.labels_last_click[i] = True
                        elif self.labels_last_click[i] == True:
                            self.labels_click[i] = False
                    else:
                        labels_color[i] = (200, 0, 0)
                        self.labels_click[i] = False
                        self.labels_last_click[i] = False
                else:
                    labels_color[i] = (255, 0, 0)
                    self.labels_click[i] = False
                    self.labels_last_click[i] = False

                cv2.rectangle(image,(x1,y1),(x2,y2),labels_color[i],-1)
                cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,0),2)
                label = self.labels[i]
                labelSize = cv2.getTextSize(label, self.font, 0.75, 2)[0]
                labelX = x1 + int(abs(self.w - labelSize[0])/2)
                labelY = y1 + int(1.75*abs(self.h - labelSize[1])/2)
                cv2.putText(image, '%s' % label, (labelX, labelY), self.font, 0.75, (255,255,255), 2, cv2.LINE_AA)

        cv2.rectangle(image,(self.x,self.y),(self.x+self.w,self.y+self.h),c,-1)
        cv2.putText(image, '%s' % self.text, (self.textX, self.textY), self.font, 0.75, (255,255,255), 2, cv2.LINE_AA)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_drawing)

W = 1900
H = 950
background = np.zeros((H,W,3)).astype(np.uint8)
frame = 1*background

img_counter = 0
source = inputHandler(dir_path)
image = source.get()

image_h, image_w = image.shape[0], image.shape[1]
image_h_scaled = 864#int(scale*image_h)
image_w_scaled = 1536#int(scale*image_w)
image = cv2.resize(image, (image_w_scaled, image_h_scaled))

button_quit = Button(1800, 0, 100, 100, 'Quit')
button_export = Button(1800, 150, 100, 100, 'Export')
button_ENP = Button(1800, 250, 100, 100, 'E+N+P')

button_draw = Button(1650, 0, 100, 50, 'Draw')
button_save = Button(1650, 50, 100, 50, 'Save')
button_select = Button(1650, 100, 100, 50, 'Select')
button_move = Button(1650, 150, 100, 50, 'Move')
button_delete = Button(1650, 200, 100, 50, 'Delete')
button_scaleMinus = Button(1650, 250, 100, 50, 'Scale -')
button_scalePlus = Button(1650, 300, 100, 50, 'Scale +')

button_next = Button(1800, 900, 100, 50, 'Next')
button_previous = Button(1700, 900, 100, 50, 'Previous')
button_zoom_in = Button(1650, 900, 50, 50, '+')
button_zoom_out = Button(1600, 900, 50, 50, '-')

button_train = Button(1800, 750, 100, 50, 'Train', click_color=(0,0,255))
button_nextpredict = Button(1800, 850, 100, 50, 'N + P')
button_predict = Button(1800, 800, 100, 50, 'Predict')

dropdown_classes = Dropdown(1450, 0, 150, 50, 'Labels')

def draw_buttons(frame, mouse_coordinates):
    for button in buttons:
        button.draw(frame, mouse_coordinates)
    
bb_list = []
frame_counter = 1
new_frame = True
while True:
    scale = 1
    cv2.moveWindow('Frame', 0, 0)
    key = cv2.waitKey(1)

    frame = 1*background
    frame[0:image_h_scaled, 0:image_w_scaled] = image

    if new_frame == True:
        for bb in bb_list:
            bb['xmin'] = bb['xmin'] - 15
            bb['xmax'] = bb['xmax'] + 15
            bb['ymin'] = bb['ymin'] - 15
            bb['ymax'] = bb['ymax'] + 15
        new_frame = False
    
    if len(bb_list) > 0:
        for bb in bb_list:
            if 'label' in bb.keys() and 'confidence' in bb.keys():
                name = yaml_classes[bb['label']]
                color = names_color[bb['label']]
                conf = bb['confidence']
                cv2.rectangle(frame, (int(bb['xmin']*scale), int(bb['ymin']*scale)), (int(bb['xmax']*scale), int(bb['ymax']*scale)), color,2)
                cv2.rectangle(frame, (int(bb['xmin']*scale), int(bb['ymax']*scale)), (int(bb['xmax']*scale), 25+int(bb['ymax']*scale)), color, -1)
                cv2.putText(frame, '%s %.2f' % (name, conf), (int(bb['xmin']*scale), 20 + int(bb['ymax']*scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2, cv2.LINE_AA)
            elif 'label' in bb.keys() and 'confidence' not in bb.keys():
                name = yaml_classes[bb['label']]
                color = names_color[bb['label']]
                cv2.rectangle(frame, (int(bb['xmin']*scale), int(bb['ymin']*scale)), (int(bb['xmax']*scale), int(bb['ymax']*scale)), color,2)
                cv2.rectangle(frame, (int(bb['xmin']*scale), int(bb['ymax']*scale)), (int(bb['xmax']*scale), 25+int(bb['ymax']*scale)), color, -1)
                cv2.putText(frame, '%s' % name, (int(bb['xmin']*scale), 20 + int(bb['ymax']*scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (int(bb['xmin']*scale), int(bb['ymin']*scale)), (int(bb['xmax']*scale), int(bb['ymax']*scale)), (255,0,0),2)

    draw_buttons(frame, mouse_coordinates)

    if button_draw.persistent_click == True and drawing == True:
        button_select.persistent_click = False
        button_move.persistent_click = False
        if (0 <= mouse_coordinates[0] <= 1450):
            cv2.rectangle(frame,pointA, pointB, (0,0,255), 2)
            last_ptB = 1*pointB
            last_ptA = 1*pointA
    if button_draw.persistent_click == True and drawing == False:
        if last_ptA is not None and last_ptB is not None:
            cv2.rectangle(frame,last_ptA, last_ptB, (0,0,255), 2)

    if button_save.click == True:
        if last_ptA is not None and last_ptB is not None:
            button_draw.persistent_click = False
            button_select.persistent_click = False
            button_move.persistent_click = False
            ptA = (last_ptA[0]/scale, last_ptA[1]/scale)
            ptB = (last_ptB[0]/scale, last_ptB[1]/scale)
            bb_list.append( {'xmin': ptA[0], 'ymin' : ptA[1], 'xmax' : ptB[0], 'ymax' : ptB[1]}  )
            last_ptB = (0,0)
            last_ptA = (0,0)

    if button_select.persistent_click == True:
        if len(bb_list) != 0:
            button_draw.persistent_click = False
            button_move.persistent_click = False
            bb_id, bbs = select_bb(bb_list, pointB, scale)
            cv2.rectangle(frame, (int(bbs['xmin']*scale), int(bbs['ymin']*scale)), (int(bbs['xmax']*scale), int(bbs['ymax']*scale)), (255,255,255),2)

            if True in dropdown_classes.labels_click:
                label = dropdown_classes.labels_click.index(True)
                bb_list[bb_id]['label'] = label

            if button_delete.click == True:
                bb_list.pop(bb_id)

    if button_scalePlus.click == True:
        button_select.persistent_click = False
        bb_list[bb_id]['xmin'] = bb_list[bb_id]['xmin'] - 5
        bb_list[bb_id]['xmax'] = bb_list[bb_id]['xmax'] + 5
        bb_list[bb_id]['ymin'] = bb_list[bb_id]['ymin'] - 5
        bb_list[bb_id]['ymax'] = bb_list[bb_id]['ymax'] + 5
    
    if button_scaleMinus.click == True:
        button_select.persistent_click = False
        bb_list[bb_id]['xmin'] = bb_list[bb_id]['xmin'] + 5
        bb_list[bb_id]['xmax'] = bb_list[bb_id]['xmax'] - 5
        bb_list[bb_id]['ymin'] = bb_list[bb_id]['ymin'] + 5
        bb_list[bb_id]['ymax'] = bb_list[bb_id]['ymax'] - 5


    if button_move.click == True:
        button_select.persistent_click = False
        button_move.persistent_click = True

    if button_move.persistent_click == True:
        button_draw.persistent_click = False
        bb = bb_list[bb_id]
        cv2.rectangle(frame, (int(bb['xmin']*scale), int(bb['ymin']*scale)), (int(bb['xmax']*scale), int(bb['ymax']*scale)), (0,0,0),2)
        if mouse_move == True:
            if (bb['xmin'] <= mouse_coordinates[0]/scale <= bb['xmax']):
                bb_new = move_bb(bb, mouse_coordinates, scale)
                bb_list[bb_id]['xmin'] = bb_new[0][0]
                bb_list[bb_id]['ymin'] = bb_new[0][1]
                bb_list[bb_id]['xmax'] = bb_new[1][0]
                bb_list[bb_id]['ymax'] = bb_new[1][1]

    if key == ord('z'):
        button_next.click = True
    if button_next.click == True:
        frame_counter += 1
        scale = 1.0
        bb_list = []
        source.increase()
        image = source.get()
        image_h, image_w = image.shape[0], image.shape[1]
        image_h_scaled = 864#int(scale*image_h)
        image_w_scaled = 1536#int(scale*image_w)
        image = cv2.resize(image, (image_w_scaled, image_h_scaled))

    if button_previous.click == True:
        frame_counter -= 1
        scale = 0.8
        bb_list = []
        source.decrease()
        image = source.get()
        image_h, image_w = image.shape[0], image.shape[1]
        image_h_scaled = 864#int(scale*image_h)
        image_w_scaled = 1536#int(scale*image_w)
        image = cv2.resize(image, (image_w_scaled, image_h_scaled))

    if button_zoom_in.click == True:
        scale += 0.01
        image = source.get()
        image_h, image_w = image.shape[0], image.shape[1]
        image_h_scaled = 864#int(scale*image_h)
        image_w_scaled = 1536#int(scale*image_w)
        image = cv2.resize(image, (image_w_scaled, image_h_scaled))
    if button_zoom_out.click == True:
        scale -= 0.01
        image = source.get()
        image_h, image_w = image.shape[0], image.shape[1]
        image_h_scaled = 864#int(scale*image_h)
        image_w_scaled = 1536#int(scale*image_w)
        image = cv2.resize(image, (image_w_scaled, image_h_scaled))

    if button_predict.click == True:
        bb_list = []
        im = cv2.resize(image, (512,512))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ratio_x = image.shape[1]/im.shape[1]
        ratio_y = image.shape[0]/im.shape[0]
        res = predict(im)
        for rect in res:
            x1 = int(ratio_x*rect[0]/scale)
            y1 = int(ratio_y*rect[1]/scale)
            x2 = int(ratio_x*rect[2]/scale)
            y2 = int(ratio_y*rect[3]/scale)
            bb_list.append( {'xmin': x1, 'ymin' : y1, 'xmax' : x2, 'ymax' : y2, 'label': int(rect[4]), 'confidence':float(rect[5])}  )

    if key == ord('n'):
        button_nextpredict.click = True

    if button_nextpredict.click == True:
        bb_list = []
        source.increase()
        image = source.get()
        image_h, image_w = image.shape[0], image.shape[1]
        image_h_scaled = 864#int(scale*image_h)
        image_w_scaled = 1536#int(scale*image_w)
        image = cv2.resize(image, (image_w_scaled, image_h_scaled))
        im = cv2.resize(image, (512,512))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ratio_x = image.shape[1]/im.shape[1]
        ratio_y = image.shape[0]/im.shape[0]
        res = predict(im)
        for rect in res:
            x1 = int(ratio_x*rect[0]/scale)
            y1 = int(ratio_y*rect[1]/scale)
            x2 = int(ratio_x*rect[2]/scale)
            y2 = int(ratio_y*rect[3]/scale)
            bb_list.append( {'xmin': x1, 'ymin' : y1, 'xmax' : x2, 'ymax' : y2, 'label': int(rect[5]), 'confidence':float(rect[4])}  )

    if button_train.click == True:
        draw_buttons(frame, mouse_coordinates)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
        train(model)

    if button_export.click == True:
        labels_dict = []
        for bb in bb_list:
            labels_dict.append(bb)
        with open('./data/dataset/labels/%s_%i.json' % ('image_',frame_counter), 'w') as labels:
            json.dump(labels_dict, labels)
        #im = cv2.resize(image, (1536, 864))
        cv2.imwrite('./data/dataset/images/%s_%i.jpg' % ('image_', frame_counter), image)
        frame_counter += 1

    if key == ord('n'):
        button_ENP.click = True

    if button_ENP.click == True:
        new_frame = True
        labels_dict = []
        for bb in bb_list:
            labels_dict.append(bb)
        with open('./dataset/raw/labels/%s_%i.json' % (video_name,frame_counter), 'w') as labels:
            json.dump(labels_dict, labels)
        im = cv2.resize(image, (1536, 864))
        cv2.imwrite('./dataset/raw/images/%s_%i.jpg' % (video_name, frame_counter), im)
        frame_counter += 1

        bb_list = []
        source.increase()
        image = source.get()
        image_h, image_w = image.shape[0], image.shape[1]
        image_h_scaled = 864#int(scale*image_h)
        image_w_scaled = 1536#int(scale*image_w)
        image = cv2.resize(image, (image_w_scaled, image_h_scaled))
        im = cv2.resize(image, (512,512))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ratio_x = image.shape[1]/im.shape[1]
        ratio_y = image.shape[0]/im.shape[0]
        res = predict(im)
        for rect in res:
            x1 = int(ratio_x*rect[0]/scale)
            y1 = int(ratio_y*rect[1]/scale)
            x2 = int(ratio_x*rect[2]/scale)
            y2 = int(ratio_y*rect[3]/scale)
            bb_list.append( {'xmin': x1, 'ymin' : y1, 'xmax' : x2, 'ymax' : y2, 'label': int(rect[5]), 'confidence':float(rect[4])}  )


    if button_quit.click == True:
        break
    
    cv2.putText(frame, 'Frame %i/%i' % (frame_counter, source.length), (0, 900), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    cv2.imshow("Frame", frame)

cv2.destroyAllWindows()
