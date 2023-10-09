import mindspore.dataset as ds
import numpy as np
import pickle
import random 
from utils.utils import pickle_load, lung_crop, resample

class WLDATASET(ds.Dataset):
    def __init__(self, 
        phase: str,
        task,
        pkl_file,
        transforms=None,
        bbox_path=None,
        bbox_size=None,
        lung_crop=None
    ):
        super(WLDATASET, self).__init__()
        
        print('BBOX DATASET')

        self.phase = phase 
        self.task = task 
        self.sample_list = pickle_load(pkl_file)

        self.bbox_path = bbox_path
        self.bbox_size = bbox_size
        self.bbox_nums = 0

        self.transforms = transforms
        self.lung_crop = lung_crop
        
        self.get_bbox_dict(self.bbox_path)
        #print(self.data_list[:2])

    def __getitem__(self, idx):
        input_data, label = self.get_data(idx)
        
        if self.transforms is not None:
            if isinstance(self.transforms, list):
                for trans in self.transforms:
                    input_data = trans(input_data)
            else:
                input_data = self.transforms(input_data)

        return input_data, label 
    
    def __len__(self):
        
        return len(self.data_list)

    def get_data(self, idx): 

        record = self.data_list[idx] 
        case_id = record["pid"]
        label = record["label"]
        data_path = record["npy_path"]
        bbox = record["bbox"]
        ct_npy = np.load(data_path)
        
        c_z,c_y,c_x,r_z,r_y,r_x= bbox
        z_s = max(0, c_z-r_z)
        z_e = min(ct_npy.shape[0], c_z+r_z+1)
        y_s = max(0, c_y-r_y)
        y_e = min(ct_npy.shape[1], c_y+r_y+1)
        x_s = max(0, c_x-r_x)
        x_e = min(ct_npy.shape[2], c_x+r_x+1)
                
        ct_bbox = ct_npy[z_s:z_e, y_s:y_e, x_s:x_e]
        ct_bbox = resample(ct_bbox, self.bbox_size)
        ct_npy = ct_bbox 

        return ct_npy, label 
    
    def get_bbox_dict(self, path):
        from collections import defaultdict     
        file_path  = path
        self.bbox_dict = defaultdict(list)

        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                pid, bbox_path, bbox_str = line.split(':')
                bbox = [int(val) for val in bbox_str.strip('[]\n').split(',')]
                radius_z = bbox[3]

                if not self.bbox_dict[pid] or radius_z > self.bbox_dict[pid][0]['bbox'][3]:
                    self.bbox_dict[pid] = [{'pid': pid, 'path': bbox_path, 'bbox': bbox}]
        
        data_list = []
        for record in self.sample_list:
            #print(record)
            pid = record['pid']
            if pid in self.bbox_dict and record['npy_path']==self.bbox_dict[pid][0]['path']:
                label = self.get_label(record['label'])
                data_list.append({
                    'pid': pid, 
                    'label': label,
                    'npy_path': self.bbox_dict[pid][0]['path'], 
                    'bbox': self.bbox_dict[pid][0]['bbox']
                    })
        
        assert self.task != 'EGFR_subtype_3_classification', 'error classes!=2'
        
        if self.phase == 'train':
            self.data_list = self.get_interleaved_data_list(data_list)
            #self.data_list = data_list
        elif self.phase == 'eval':
            self.data_list = data_list
        elif self.phase == 'test': 
            self.data_list = data_list
            


    def get_label(self, label_rec):
        if self.task == "EGFR_subtype_3_classification":
            if label_rec in [0,1,2]:
                label = label_rec
            else:
                raise NotImplementedError("label_rec: "+label_rec)
        
        elif self.task == "EGFR_2_classification":
            if label_rec in [1,2]:
                label = 1
            elif label_rec == 0:
                label = 0
            else:
                raise NotImplementedError("label_rec: "+label_rec)
        
        else:
            raise NotImplementedError("task: "+self.task)
        
        return label

    def get_interleaved_data_list(self, data_list):

        data_dict = {}
        for data in data_list:
            label = data['label']
            if label not in data_dict:
                data_dict[label] = []
            data_dict[label].append(data)
        
        for label in data_dict:
            random.shuffle(data_dict[label])
        
        inter_data_list = []
        max_len = max([len(data_dict[label]) for label in data_dict])
        for i in range(max_len):
            for label in data_dict:
                if i < len(data_dict[label]):
                    inter_data_list.append(data_dict[label][i])
        
        return inter_data_list