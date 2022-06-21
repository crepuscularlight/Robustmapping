import os
import numpy as np
import math

def load_paires(file,graph_pairs_dir):
  paires=[]
  with open(file) as f:
    while True:
      line=f.readline()
      if not line:
        break
      line=line.strip().split(" ")
      paires.append([os.path.join(graph_pairs_dir,line[0]),os.path.join(graph_pairs_dir,line[1])])
  return paires

def process_pair(path):
  data1 = np.load(path[0], allow_pickle=True)
  data2 = np.load(path[1], allow_pickle=True)

  data = {}

  pose1 = data1["pose"]
  pose2 = data2["pose"]

  data["nodes_1"] = data1["nodes"]
  data["nodes_2"] = data2["nodes"]


  dis = math.sqrt((pose1[3] - pose2[3]) ** 2 + (pose1[11] - pose2[11]) ** 2)

  data["pcn_features_1"] = data1["pcn_feature"]
  data["pcn_features_2"] = data2["pcn_feature"]


  data["centers_1"] = data1["centers"]
  data["centers_2"] = data2["centers"]

  data["bbox_1"]=data1["bbox"]
  data["bbox_2"]=data1["bbox"]

  data["distance"] = dis


  return data
