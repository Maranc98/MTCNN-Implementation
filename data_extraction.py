
import cv2
import random
import numpy as np
import math
import time
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from models.mtcnn import PNet, RNet, ONet
import torchvision

class ImageManager():
    def __init__(self, min_crop_side=48, biased_crop_sigma=0.2):
        self._min_crop_side = min_crop_side
        # biased_crop_sigma is 0.2 by default because it covers a
        #   good range in [0,1]
        self._biased_crop_sigma = biased_crop_sigma

    def random_crop(self, shape):
        # We assume faces to be somewhat square shaped
        # We assume the pictures to be at least self._min_crop_side sized
        min_side = min(shape[0], shape[1])
        crop_side = max(int(random.random() * min_side), self._min_crop_side)
        x = int(random.random() * (shape[0] - crop_side))
        y = int(random.random() * (shape[1] - crop_side))

        return x, y, crop_side

    def biased_random_crop(self, shape):
        # We assume faces to be somewhat square shaped
        # We assume the pictures to be at least self._min_crop_side sized
        # We assume that good face crops will be closer to the _min_crop_side
        #   than to the minimum side of the picture
        min_side = min(shape[0], shape[1])
        crop_side = max(int(min(abs(random.gauss(0,0.1) * min_side), min_side)), self._min_crop_side)
        x = int(random.random() * (shape[0] - crop_side))
        y = int(random.random() * (shape[1] - crop_side))

        return x, y, crop_side

    def iou(self, box1, box2):
        # These are the coordinates of top-left, btm-right corners of intersection
        x0 = max(box1[0], box2[0])
        y0 = max(box1[1], box2[1])
        x1 = min(box1[2], box2[2])
        y1 = min(box1[3], box2[3])

        intersection = max(0, x1 - x0) * max(0, y1 - y0)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union

    def best_fit(self, box, candidates):
        iou_scores = [self.iou(box, cand) for cand in candidates]
        best = max(iou_scores)
        idx = iou_scores.index(best)

        return best, candidates[idx]

image_manager = ImageManager()

data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

def show_img(image, *boxes, corner_mode=False, transpose=False, scores=None, title=None, offsets=None, landmarks=[]):
    fig, ax = plt.subplots()
    ax.imshow(image)

    colors = ['r', 'g', 'b', 'w']

    for i, box in enumerate(boxes):
        if transpose:
            box = [box[1], box[0], box[3], box[2]]

        if not corner_mode:
            rect = patches.Rectangle(box[:2], box[2], box[3], linewidth=1, edgecolor=colors[i%len(colors)], facecolor='none')
        else:
            rect = patches.Rectangle(box[:2], box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor=colors[i%len(colors)], facecolor='none')

        ax.add_patch(rect)

        if scores is not None:
            if scores[i] is not None:
                plt.text(box[0], box[1], f"{scores[i]:.2f}")

        if offsets is not None:
            if offsets[i] is not None:
                o = offsets[i]

                p0 = [(box[0]+box[2])/2, (box[1]+box[3])/2]
                p1 = [p0[0]+o[0], p0[1]+o[1]]

                plt.plot([p0[0], p1[0]],[p0[1], p1[1]])

                plt.plot([p1[0] - o[2]/2, p1[0] + o[2]/2], [p1[1], p1[1]], 'k:')
                plt.plot([p1[0], p1[0]], [p1[1] - o[3]/2, p1[1] + o[3]/2], 'k:')

    for l in landmarks:
        for i in range(0,len(l), 2):
            plt.plot(l[i], l[i+1], 'o')

    if title is not None:
        plt.title(str(title))

    plt.show()

def show_img_grid(imgs, scores=None, landmarks=None):

    x = math.sqrt(len(imgs)/12)
    w = max(2, math.ceil(4*x))
    h = max(2, math.ceil(3*x))

    f, axarr = plt.subplots(w,h)

    i = 0
    for x in range(w):
        for y in range(h):
            if i < len(imgs):
                axarr[x,y].imshow(imgs[i])
                if scores is not None:
                    axarr[x,y].title.set_text(f"{scores[i]:.2f}")

                i+=1
            axarr[x,y].axis('off')

    f.show()

def get_proximal_crop(bbox, std=0.25):

    x,y,w,h = bbox

    if w > h:
        y = int(max(y-(w-h)/2, 0))
        h = w
    else:
        x = int(max(x-(h-w)/2, 0))
        w = h

    square_bbox = [x,y,w,h]
    random_offset = np.multiply(np.random.normal(0,std,4) * w, [1,1,0,0]).astype(int)

    crop = np.add(square_bbox, random_offset)

    return crop

def get_relative_offsets(gt_box, crop_box):

    pw = crop_box[2] - crop_box[0]
    ph = crop_box[3] - crop_box[1]

    gt_center = ((gt_box[2] + gt_box[0])/2,(gt_box[3] + gt_box[1])/2)
    crop_center = ((crop_box[2] + crop_box[0])/2, (crop_box[3] + crop_box[1])/2)

    tx = (gt_center[0] - crop_center[0]) / pw
    ty = (gt_center[1] - crop_center[1]) / ph
    tw = math.log((gt_box[2] - gt_box[0])/pw)
    th = math.log((gt_box[3] - gt_box[1])/ph)

    return [tx, ty, tw, th]

def relative_center_to_box(box, filter_shape, crop_coordinates):

    l = filter_shape[0]

    box[:,2:] = torch.exp(box[:,2:])
    box = box * l

    box[:,0] = box[:,0] + crop_coordinates[:,1] - box[:,2]/2     # x = crop_center_to_box_center + crop_x_to_crop_center + crop_x - box_width/2
    box[:,0] = torch.add(box[:,0], l/2)
    box[:,1] = box[:,1] + crop_coordinates[:,0] - box[:,3]/2
    box[:,1] = torch.add(box[:,1], l/2)

    box[:,2] = box[:,0] + box[:,2]
    box[:,3] = box[:,1] + box[:,3]

    return box

def get_random_crop(img_shape):

    side = np.random.randint(0,min(img_shape)-1)

    x0 = np.random.randint(0,img_shape[0]-side-1)
    y0 = np.random.randint(0,img_shape[1]-side-1)

    return [x0,y0,side,side]

def get_net_crops(img, net, filter_shape):

    levels = 3
    max_pyramid_eval = 100
    stride = int(filter_shape[0]/2)

    image_pyramid, scale_factor = get_image_pyramid(img, filter_shape=filter_shape, levels=levels, max_image_evaluations=max_pyramid_eval, stride=stride)
    scores, candidates, _ = get_candidates(image_pyramid, scale_factor, net, filter_shape=filter_shape, stride=stride)

    #take = np.where(scores > 0.5)
    #candidates = candidates[take]
    #scores = scores[take]

    candidates, take = non_maximum_suppression(candidates, scores=scores, overlapThresh=0.5)
    scores = scores[take]

    return candidates

def extract_random_crops_MTCNN(save=True, show=False, TOTAL_FACES_TO_SAVE = 10000, FACES_PER_SAVE = 100, IMAGE_RES = (48,48), SAVE_FOLDER = "datasets/WIDER/face_crops/pnet/proximity", CROPS_PER_IMAGE = 5, net=None, net_input_shape=None):
    """
    Extracts faces aroung ground truth faces and computes iou and offset.
    """

    categories = ["positive", "negative", "partfaces"]
    category_indices = {c:0 for c in categories}

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    for c in categories:
        if not os.path.exists(SAVE_FOLDER + "/" + c):
            os.makedirs(SAVE_FOLDER + "/" + c)

    lines = []
    with open("datasets/WIDER/wider_face_split/wider_face_train_bbx_gt.txt") as file:
        lines = file.readlines()

    images_counter = 0
    savefile_index = 0
    lines_iter = iter(lines)
    for line in lines_iter:
        # Reading dataset values
        image_path = line.strip()
        num_faces = int(next(lines_iter))
        # Files with 0 faces still have one line full of zeros we have to read
        values = []
        for i in range(max(1,num_faces)):
            values += [list(map(int, next(lines_iter).strip().split(' ')))]

        boxes = np.array(values)[:,0:4]
        boxes[:,2] = np.sum(boxes[:,[0,2]],axis=1)
        boxes[:,3] = np.sum(boxes[:,[1,3]],axis=1)

        values = np.asarray(values)
        take = np.where((values[:,4] == 0) & (values[:,5] == 0) & (values[:,6] == 0) & (values[:,7] == 0) & (values[:,8] == 0) & (values[:,9] == 0))
        # blur:clear, expression:typical, illumination:normal, occlusion:no, pose:typical, valid image

        values = values[take]
        boxes = boxes[take]

        take = np.where((boxes[:,2] - boxes[:,0] > IMAGE_RES[0]) & (boxes[:,3] - boxes[:,1] > IMAGE_RES[1]))
        values = values[take]
        boxes = boxes[take]

        if len(values) == 0:
            continue

        img = cv2.imread("datasets/WIDER/WIDER_train/images/" + image_path)

        if net is None:
            crops = []

            for repeat in range(CROPS_PER_IMAGE):
                # Get a crop, depending on what kind of sample we are lacking
                if repeat > CROPS_PER_IMAGE/2 or category_indices["negative"] >= TOTAL_FACES_TO_SAVE/3 or np.random.random() > 0.5:
                    ind = int(np.random.random()*len(boxes))
                    bbox = list(values[ind][:4])

                    x,y,l,_ = get_proximal_crop(bbox)
                else:
                    x,y,l,_ = get_random_crop([img.shape[1], img.shape[0]])

                crops += [[x,y,l]]

        else:
            crops = []

            try:
                net_crops = get_net_crops(img, net, net_input_shape)
                net_crops = [get_square_crop(c) for c in net_crops]
                crops += [[c[0], c[1], c[2]-c[0]] for c in net_crops]
            except:
                print("Netcrop failure")

            for _ in range(len(crops)):
                ind = int(np.random.random()*len(boxes))
                bbox = list(values[ind][:4])

                x,y,l,_ = get_proximal_crop(bbox)

                crops += [[x,y,l]]



        for x,y,l in crops:

            # Skip small crops
            if l < IMAGE_RES[0]:
                continue

            # Skip crops that are outside the img
            try:
                img_crop = cv2.resize(img[y:y+l,x:x+l], IMAGE_RES)
            except:
                continue

            # Build target output
            crop_coordinates = [x, y, x+l, y+l]
            score, box = image_manager.best_fit(crop_coordinates, boxes)

            if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
                continue

            relative_offsets = get_relative_offsets(box, crop_coordinates)

            #show_offsets = [r*l if i < 2 else math.exp(r)*l for i, r in enumerate(relative_offsets)]
            #show_img(img, crop_coordinates, box, title=f"{x}, {y}, {l}", corner_mode=True, scores=[score,1], offsets=[show_offsets, None])
            #return

            input = np.asarray(img_crop)
            output = np.concatenate([[score], relative_offsets]).astype(float)

            #show_img(img, [x,y,x+l,y+l], box, title=f"{x}, {y}, {l}", corner_mode=True)
            #show_img(img[y:y+l,x:x+l],title=score)
            #show_img(img_crop,title=score)

            # Stopping condition
            images_counter, category_indices = save_sample(input, output, SAVE_FOLDER, images_counter, category_indices, TOTAL_FACES_TO_SAVE/len(category_indices.values()))
            print(category_indices)

            if all(value > TOTAL_FACES_TO_SAVE/len(category_indices.values()) for value in category_indices.values()):
                print(category_indices)
                return

def get_ckpt_from_name(name):

    txt = "./good_runs/" + name + "/"
    txt += os.listdir(txt)[0]

    return txt

def load_net(name="", type=""):

    if type == "pnet":
        constructor =  PNet.load_from_checkpoint
    elif type == "rnet":
        constructor = RNet.load_from_checkpoint
    elif type == "onet":
        constructor = ONet.load_from_checkpoint

    net = constructor(get_ckpt_from_name(name))
    return net

def get_image_pyramid(img, levels=3, max_image_evaluations=50, filter_shape=(12,12), output_shape=(24,24), stride=None):
    filter_area = filter_shape[0] * filter_shape[1]
    img_shape = img.shape[:2]
    img_area = img.shape[0]*img_shape[1]
    if stride is None:
        scale_factor = np.sqrt(filter_area * max_image_evaluations / img_area)
    else:
        scale_factor = np.sqrt(max_image_evaluations * stride * stride / (img_area)) # filter side approssimato a zero
    scale_factor = min(1,scale_factor, filter_shape[0]/output_shape[0])

    base_shape = [int(s*scale_factor) for s in reversed(img_shape)]
    img = cv2.resize(img, base_shape)

    pyramid = [img]
    for i in range(levels-1):
        pyramid += [cv2.pyrDown(pyramid[i])]

    #[show_img(i) for i in pyramid]

    return pyramid, scale_factor

def get_candidates(pyramid, scale_factor, net, stride=12, filter_shape=(12,12), base_image_=None, device="cpu"):

    scores = []
    boxes = []
    landmarks = []

    for level, img in enumerate(pyramid):
        img_shape = img.shape
        x_steps = int((img.shape[0] - filter_shape[0])/stride) + 1
        y_steps = int((img.shape[1] - filter_shape[1])/stride) + 1

        if(x_steps < 1 or y_steps < 1):
            continue

        img_crops = []
        crop_coordinates = []

        for i in range(x_steps):
            for j in range(y_steps):
                x0 = stride*i
                y0 = stride*j
                xf = x0 + filter_shape[0]
                yf = y0 + filter_shape[1]
                img_crop = img[x0:xf, y0:yf]

                img_crops.append(data_transforms(img_crop).tolist())
                crop_coordinates.append([x0,y0,xf,yf])

        img_batch = torch.Tensor(img_crops).to(device)

        out = net(img_batch)

        if len(out) > 2:
            score, box, landmark = out
        else:
            score, box = out

        if len(box.size()) > 2:
            box = box[:,:,0,0]
            score = score[:,:,0,0]

        crop_coordinates = torch.Tensor(crop_coordinates).view(box.size()).to(device)

        box = relative_center_to_box(box, filter_shape, crop_coordinates)
        box = box / scale_factor * (2**level)

        if len(out) > 2:
            landmark = get_absolute_landmarks(landmark, filter_shape, crop_coordinates)
            landmark = landmark / scale_factor * (2**level)
            landmarks += landmark.tolist()

        scores += score[:,1].view((len(score))).tolist()
        boxes += box.view((len(box), 4)).tolist()

    scores = np.asarray(scores)
    boxes = np.asarray(boxes).astype(int)
    landmarks = np.asarray(landmarks).astype(int)

    return scores, boxes, landmarks

def non_maximum_suppression(candidates, scores=None, overlapThresh=0.5):

    if len(candidates) == 0:
        return [], []

    pick = []

    x1 = candidates[:,0]
    y1 = candidates[:,1]
    x2 = candidates[:,2]
    y2 = candidates[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    if scores is None:
        idxs = np.argsort(area)
    else:
        idxs = np.argsort(scores)

    while len(idxs) > 0:

        i = idxs[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:-1]])
        yy1 = np.maximum(y1[i], y1[idxs[:-1]])
        xx2 = np.minimum(x2[i], x2[idxs[:-1]])
        yy2 = np.minimum(y2[i], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:-1]]

        idxs = np.delete(idxs, np.concatenate(([-1], np.where(overlap > overlapThresh)[0])))

    return candidates[pick].astype("int"), pick

def save_sample(x, y, save_folder, images_counter, category_indices, target_number):

    score = y[0]
    category = "negative" if score < 0.4 else "positive" if score > 0.65 else "partfaces"

    if category_indices[category] <= target_number:

        print(f"Saving data... ({images_counter+1})", x.shape, y.shape)

        np.save(save_folder + f"/{category}/input{category_indices[category]}.npy", x)
        np.save(save_folder + f"/{category}/output{category_indices[category]}.npy", y)

        category_indices[category] += 1
        images_counter += 1

    return images_counter, category_indices

def extract_net_faces(TOTAL_FACES_TO_SAVE = 60000, SAVE_FOLDER = "datasets/WIDER/MTCNN_face_crops/rnet/proximity", IMAGE_RES = (24,24), net=None, net_input_shape=(12,12)):
    """
    Extracts faces aroung ground truth faces and computes iou and offset.
    """

    categories = ["positive", "negative", "partfaces"]
    category_indices = {c:0 for c in categories}

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    for c in categories:
        if not os.path.exists(SAVE_FOLDER + "/" + c):
            os.makedirs(SAVE_FOLDER + "/" + c)

    lines = []
    with open("datasets/WIDER/wider_face_split/wider_face_train_bbx_gt.txt") as file:
        lines = file.readlines()


    images_counter = 0
    lines_iter = iter(lines)
    for line in lines_iter:
        # Reading dataset values
        image_path = line.strip()
        num_faces = int(next(lines_iter))
        # Files with 0 faces still have one line full of zeros we have to read
        values = []
        for i in range(max(1,num_faces)):
            values += [list(map(int, next(lines_iter).strip().split(' ')))]

            boxes = np.array(values)[:,0:4]
            boxes[:,2] = np.sum(boxes[:,[0,2]],axis=1)
            boxes[:,3] = np.sum(boxes[:,[1,3]],axis=1)

        img = cv2.imread("datasets/WIDER/WIDER_train/images/" + image_path)

        # for each image
        #   apply pnet to image pyramid
        #   merge similar crops
        #   compute ground truth IOU for each crop
        #   build positive, negative, partfaces datasets (20k each?)

        image_pyramid, scale_factor = get_image_pyramid(img, filter_shape=net_input_shape, output_shape=IMAGE_RES)

        #show_img(img)
        #for i in image_pyramid:
        #    show_img(i)

        scores, candidates = get_candidates(image_pyramid, scale_factor, net, filter_shape=net_input_shape, stride=net_input_shape[0], base_image_=img)# Test
        #show_img(img, *candidates, corner_mode=True, transpose=True, scores=scores)

        candidates, pick = non_maximum_suppression(candidates)
        #show_img(img, *candidates, corner_mode=True, transpose=True)

        for candidate in candidates:

            # Transpose candidate
            candidate = [
                min(candidate[1], candidate[3]),
                min(candidate[2], candidate[0]),
                max(candidate[1], candidate[3]),
                max(candidate[2], candidate[0])
            ]

            score, box = image_manager.best_fit(candidate, boxes)

            offsets = [box[i] - candidate[i] for i in range(len(box))]

            [x0, y0, x1, y1] = candidate

            try:
                img_crop = cv2.resize(img[x0:x1,y0:y1], IMAGE_RES)
            except:
                #print(x0,x1,y0,y1)
                continue

            #print(score)
            #show_img(img, candidate, box, corner_mode=True)

            x = np.asarray(img_crop)
            y = np.concatenate([[score], offsets]).astype(float)

            #show_img(x)

            images_counter, category_indices = save_sample(x, y, SAVE_FOLDER, images_counter, category_indices, TOTAL_FACES_TO_SAVE/len(category_indices.values()))
            print(category_indices)

            if all(value > TOTAL_FACES_TO_SAVE/len(category_indices.values()) for value in category_indices.values()):
                print(category_indices)
                return

def celeba_square_crop(img_shape, landmarks):

    h,w = img_shape[:2]

    if w > h: # Wont ever happen tho
        d = int((w-h)/2)
        crop = [d, 0, d + h, h]
        landmarks = [l - d if i % 2 == 0 else l for i,l in enumerate(landmarks)]
    else:
        d = int((h-w)/2)
        crop = [0, d, w, d+w]
        landmarks = [l if i % 2 == 0 else l - d for i,l in enumerate(landmarks)]

    return crop, landmarks

def get_relative_landmarks(img_size, landmarks):

    img_c = [i/2 for i in img_size]

    for i in range(0, len(landmarks), 2):

        lx, ly = landmarks[i:i+2]

        landmarks[i] = (lx - img_c[0]) / img_size[0]
        landmarks[i+1] = (ly - img_c[1]) / img_size[1]

    return landmarks

def get_absolute_landmarks(landmarks, filter_shape, crop_coordinates):

    assert filter_shape[0] == filter_shape[1]

    landmarks = landmarks * filter_shape[0]

    for i in range(0,10,2):
        landmarks[:,i] = filter_shape[0]/2 + crop_coordinates[:,1] + landmarks[:,i]
        landmarks[:,i+1] = filter_shape[1]/2 + crop_coordinates[:,0] + landmarks[:,i+1]

    return landmarks

def celeba_rnet_crop(face, rnet):

    levels = 2
    filter_shape = (24,24)
    max_pyramid_eval = 100
    stride = int(filter_shape[0]/2)

    image_pyramid, scale_factor = get_image_pyramid(face, filter_shape=filter_shape, levels=levels, max_image_evaluations=max_pyramid_eval, stride=stride)
    scores, candidates, _ = get_candidates(image_pyramid, scale_factor, rnet, filter_shape=filter_shape, stride=stride)

    take = np.where(scores > 0.95)
    candidates = candidates[take]
    scores = scores[take]

    candidates, take = non_maximum_suppression(candidates, scores, overlapThresh=0.2)
    scores = scores[take]

    return candidates, scores

def get_square_crop(crop):

    w = crop[2]-crop[0]
    h = crop[3]-crop[1]

    if w > h:
        crop[0] += int((w-h)/2)
        crop[2] -= int((w-h)/2)
    else:
        crop[1] += int((h-w)/2)
        crop[3] -= int((h-w)/2)

    return crop

def get_landmarks_in_crop(landmarks, crop):

    l = [landmarks[i] - crop[i%2] for i in range(0,10)]

    is_outside_crop = False
    for i in range(10):
        if landmarks[i] < crop[i%2] or landmarks[i] > crop[i%2 + 2]:
            is_outside_crop = True
            break

    return l, is_outside_crop

def extract_landmarks_celebA(IMAGE_RES = (48,48), SAVE_FOLDER = "datasets/CelebA/landmarks", TOTAL_FACES_TO_SAVE = 200000, rnet=None):

    assert TOTAL_FACES_TO_SAVE <= 202599

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    annotations = open("datasets/CelebA/list_landmarks_align_celeba.txt")
    next(annotations)
    next(annotations)

    save_idx = 1
    for face_idx in range(1,202599+1):

        face = cv2.imread(f"datasets/CelebA/img_align_celeba/{face_idx:06d}.jpg")

        landmarks = next(annotations).split()[1:]
        landmarks = list(map(int, landmarks))

        #show_img(face, landmarks=[landmarks])

        crops, scores = celeba_rnet_crop(face, rnet)

        for crop_idx, crop in enumerate(crops):

            crop = get_square_crop(crop)
            x0,y0,xf,yf = crop

            crop_landmarks, is_outside_crop = get_landmarks_in_crop(landmarks, crop)

            if is_outside_crop:
                continue

            cropped_face = face[y0:yf, x0:xf]

            #show_img(cropped_face, landmarks=[crop_landmarks], title=f"{scores[crop_idx]}")

            # Rescale to filter size
            cropped_face = cv2.resize(cropped_face, IMAGE_RES)
            scale = (xf-x0)/IMAGE_RES[0]
            crop_landmarks = [l/scale for l in crop_landmarks]

            crop_landmarks = get_relative_landmarks(IMAGE_RES, crop_landmarks)

            # Check inverse operation
            #original_landmarks = get_absolute_landmarks(np.asarray([crop_landmarks]), IMAGE_RES, np.asarray([crop])[:,[1,0]]/scale)
            #original_landmarks = original_landmarks * scale
            #original_landmarks = original_landmarks.astype(int)[0]
            #show_img(face, landmarks=[original_landmarks])

            print(f"Saving face {save_idx}")
            np.save(SAVE_FOLDER + f"/input{save_idx-1}.npy", cropped_face)
            np.save(SAVE_FOLDER + f"/output{save_idx-1}.npy", [0]*5 + crop_landmarks)
            save_idx += 1

            if save_idx > TOTAL_FACES_TO_SAVE:
                return

if __name__ == "__main__":

    type = ["pnet", "rnet", "onet"]
    side = [12,24,48]
    net = [
        None,
        load_net(name="PNET_default", type="pnet"),
        load_net(name="RNET_default", type="rnet"),
    ]
    filter_side = [None, 12, 24]

    for i in [0,1,2]:
        start = time.process_time()
        extract_random_crops_MTCNN(
            save=True,
            show=False,
            IMAGE_RES = (side[i], side[i]),
            CROPS_PER_IMAGE = 500,
            FACES_PER_SAVE = 1,
            SAVE_FOLDER = f"datasets/WIDER/MTCNN_face_crops/{type[i]}/proximity",
            TOTAL_FACES_TO_SAVE = 600000,
            net = net[i],
            net_input_shape = (filter_side[i], filter_side[i])
        )
        print(time.process_time() - start)

    for i in [1,2]:
        start = time.process_time()
        extract_landmarks_celebA(
            IMAGE_RES = (side[i], side[i]),
            SAVE_FOLDER = f"datasets/CelebA/{type[i]}/landmarks",
            TOTAL_FACES_TO_SAVE = 200000,
            rnet = net[i],
        )
        print(time.process_time() - start)
