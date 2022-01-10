import argparse

class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end

parser = argparse.ArgumentParser()
parser.add_argument("-gpu", "--use_gpu", action="store_true")
parser.add_argument("-p", "--pnet_threshold", default=0.5, type=float, choices=[Range(0.0, 1.0)])
parser.add_argument("-pn", "--pnet_NMS_threshold", default=0.5, type=float, choices=[Range(0.0, 1.0)])
parser.add_argument("-r", "--rnet_threshold", default=0.5, type=float, choices=[Range(0.0, 1.0)])
parser.add_argument("-rn", "--rnet_NMS_threshold", default=0.5, type=float, choices=[Range(0.0, 1.0)])
parser.add_argument("-o", "--onet_threshold", default=0.5, type=float, choices=[Range(0.0, 1.0)])
parser.add_argument("-on", "--onet_NMS_threshold", default=0.5, type=float, choices=[Range(0.0, 1.0)])
parser.add_argument("-py_n", "--max_pyramid_eval", default=300, type=int)
parser.add_argument("-py_l", "--pyramid_levels", default=3, type=int)
parser.add_argument("-f", "--file", default="", type=str)
