import data_extraction as de
import cv2
import numpy as np
import torch


class MTCNN():

    def __init__(self, pnet_name="PNET_default", rnet_name="RNET_default", onet_name="ONET_default", config=None):

        self.pnet_name = pnet_name
        self.rnet_name = rnet_name
        self.onet_name = onet_name

        if config is not None:
            self.config = config
        else:
            self.config = {
                "pnet_threshold": 0.5,
                "pnet_NMS_threshold": 0.5,
                "rnet_threshold": 0.5,
                "rnet_NMS_threshold": 0.5,
                "onet_threshold": 0.5,
                "onet_NMS_threshold": 0.5,
                "max_pyramid_eval": 300,
                "pyramid_levels": 3,
                "use_gpu": True,
            }

        self.device = "cuda:0" if self.config["use_gpu"] else "cpu"

        self.pnet = de.load_net(self.pnet_name, "pnet").to(self.device)
        self.rnet = de.load_net(self.rnet_name, "rnet").to(self.device)
        self.onet = de.load_net(self.onet_name, "onet").to(self.device)


    def __call__(self, image):

        filter_shape = (12,12)
        stride = int(filter_shape[0]/3)

        image_pyramid, scale_factor = de.get_image_pyramid(image, filter_shape=filter_shape, levels=self.config["pyramid_levels"], max_image_evaluations=self.config["max_pyramid_eval"], stride=stride)
        scores, candidates, _ = de.get_candidates(image_pyramid, scale_factor, self.pnet, filter_shape=filter_shape, stride=stride, device=self.device)

        # Take only the better boxes
        take = np.where(scores > self.config["pnet_threshold"])
        candidates = candidates[take]
        scores = scores[take]

        # NMS with score priority to remove redundant boxes
        candidates, take = de.non_maximum_suppression(candidates, scores=scores, overlapThresh=self.config["pnet_NMS_threshold"])
        scores = scores[take]

        # Filter with the bigger nets
        scores, candidates, _ = self.cascade_step(image, candidates, self.rnet, (24,24), threshold=self.config["rnet_threshold"], nms_threshold=self.config["rnet_NMS_threshold"])
        scores, candidates, landmarks = self.cascade_step(image, candidates, self.onet, (48,48), threshold=self.config["onet_threshold"], nms_threshold=self.config["onet_NMS_threshold"])

        return scores, candidates, landmarks

    def filter_crops(self, images, net, threshold):

        images = torch.Tensor(images).to(self.device)

        out = net(images)

        if len(out) == 3:
            scores, boxes, landmarks = out
        else:
            scores, boxes = out
            landmarks = None

        if len(boxes.size()) > 2:
            boxes = boxes[:,:,0,0]
            scores = scores[:,:,0,0]

        scores = scores.detach().cpu().numpy()[:,1]
        boxes = boxes.detach().cpu().numpy()

        take = np.where(scores > threshold)

        scores = scores[take]
        boxes = boxes[take]
        landmarks = landmarks[take] if landmarks is not None else None

        return scores, boxes, landmarks, take

    def rect_to_square(self, rect):

        # Rect di formato x0,y0,xf,yf
        x0,y0,xf,yf = rect

        dx = xf-x0
        dy = yf-y0

        if dx > dy:
            d = int((dx-dy)/2)
            y0 -= d
            yf += d
        else:
            d = int((dy-dx)/2)
            x0 -= d
            xf += d

        return [x0,y0,xf,yf]

    def cascade_step(self, img, candidates, net, filter_shape, threshold=0.5, nms_threshold=0.5):

        ### Extract the square crops computed by pnet

        if len(candidates) == 0:
            return [],[],[]

        crops_coordinates = []
        showable_imgs = []#
        img_crops = []
        scales = []
        for bbox in candidates:
            y0,x0,yf,xf = self.rect_to_square(bbox)
            try:
                cropped_img = de.data_transforms(cv2.resize(img[x0:xf,y0:yf], filter_shape)).tolist()
                img_crops += [cropped_img]
                showable_imgs += [img[x0:xf,y0:yf]]#
                scales += [(xf-x0)/filter_shape[0]]
                crops_coordinates += [[x0,y0,xf,yf]]
            except:
                pass

        crops_coordinates = torch.Tensor(crops_coordinates).to(self.device)
        scales = torch.Tensor(scales).to(self.device)

        #if False:
        #    grid = [cv2.resize(img, filter_shape) for i, img in enumerate(showable_imgs)]
        #    print(len(grid))
        #    de.show_img_grid(grid)

        # Check the quality of each crop

        if len(img_crops) == 0:
            return [],[],[]

        scores, candidates, landmarks, take = self.filter_crops(img_crops, net, threshold)
        crops_coordinates = crops_coordinates[take]
        scales = scales[take]

        # Compute bounding boxes and apply NMS
        scaled_coodinates = crops_coordinates / scales[:,None]

        #if False:
        #    grid = [cv2.resize(img, filter_shape) for i, img in enumerate(showable_imgs) if i in take[0]]
        #    print(len(grid))
        #
        #    if False and landmarks is not None:
        #        landmarks = de.get_absolute_landmarks(landmarks, filter_shape, crops_coordinates*0)
        #        landmarks = landmarks * scales[:,None]
        #
        #    de.show_img_grid(grid, scores=scores, landmarks=landmarks)

        candidates = de.relative_center_to_box(torch.Tensor(candidates).to(self.device), filter_shape, scaled_coodinates)
        candidates = candidates * scales[:,None]
        candidates = candidates.detach().cpu().numpy()

        candidates[:,[0,1,2,3]] = candidates[:,[0,1,2,3]]
        candidates = candidates.astype(int)
        candidates, take = de.non_maximum_suppression(candidates, scores=scores, overlapThresh=nms_threshold)

        # Compute landmarks
        if landmarks is not None and len(landmarks) > 0:
            landmarks = de.get_absolute_landmarks(landmarks, filter_shape, scaled_coodinates)
            landmarks = landmarks * scales[:,None]
            landmarks = landmarks[take]

        return scores, candidates, landmarks
