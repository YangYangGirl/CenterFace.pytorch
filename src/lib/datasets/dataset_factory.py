from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset
from .sample.whole_body import WholeBodyDataset
from .sample.crop_body import CropBodyDataset
from .sample.egohands import EgohandsDataset


from .dataset.coco import COCO
from .dataset.wholebody import WHOLEBODY
from .dataset.ctdet_wholebody import CTDetWHOLEBODY
from .dataset.cropbody import CROPBODY
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from .dataset.pig import PIG
from .dataset.pig2 import PIG2
from .dataset.centerface import FACE
from .dataset.centerface_hp import FACEHP
from .dataset.ctdet_egohands import CTDetEGOHANDS


dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  'pig': PIG,
  'pig2': PIG2,
  'face': FACE,
  'facehp': FACEHP,
  'wholebody': WHOLEBODY,
  'cropbody': CROPBODY,
  'egohands': CTDetEGOHANDS
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': EgohandsDataset, #WholeBodyDataset,#CTDetDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset,
  'multi_pose_whole': WholeBodyDataset,
  'multi_pose_crop': CropBodyDataset
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):           # 双重继承
    pass
  return Dataset
  
