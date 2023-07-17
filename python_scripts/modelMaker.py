from __future__ import annotations

import math
import os
from os import listdir
from os.path import join

from tensorflow_lite_support.python.task.vision.core.tensor_image import TensorImage
import tflite_model_maker as mm
from tflite_support.task import vision
from tflite_model_maker import searcher

data_loader = searcher.ImageDataLoader.create("/home/gab/PycharmProjects/tf_similarity/model_efficientNet.tflite")


train_folder = "/home/gab/PycharmProjects/tf_similarity/index_v4 (senza 2e,1e,50c)"
print("Class in index:")
tot_db = 0
for folder in sorted(listdir(train_folder)):
    num_examples = len(listdir(train_folder+"/"+folder))
    tot_db = tot_db + num_examples
    print(folder+" (#examples:{})".format(num_examples))
    abs_folder = join(train_folder, folder)
    data_loader.load_from_folder(abs_folder, folder)
print("Totale esempi: {}".format(tot_db))


scann_options = searcher.ScaNNOptions(
      distance_measure="squared_l2",
      tree=searcher.Tree(num_leaves=math.ceil(math.sqrt(tot_db)), num_leaves_to_search=15),
      score_ah=searcher.ScoreAH(dimensions_per_block=1, anisotropic_quantization_threshold=0.2))


model = searcher.Searcher.create_from_data(data_loader, scann_options)
model.export(
      export_filename="searcher.tflite",
      userinfo="",
      export_format=searcher.ExportFormat.TFLITE)

searcher = vision.ImageSearcher.create_from_file("searcher.tflite")


