#!/usr/bin/env python3

import torch.cuda

from pycore.jg import JG
from pycore.pytorch_util import *
import os
import os.path
from gen.image_set_info import *
from gen.train_set import *
from gen.data_type import *
from gen.compile_images_config import *
from gen.train_param import *


class JsInference:

  def __init__(self, model_specific_script_file):
    self.verbose = False
    self.device = None
    self.model = None

    script_path = os.path.realpath(model_specific_script_file)
    self.cached_proj_path = os.path.dirname(script_path)

    t = self.proj_path("train_info")
    todo("do we need JG.train_param?")
    JG.train_param = read_object(TrainParam.default_instance, os.path.join(t, "train_param.json"))
    self.network:NeuralNetwork = read_object(NeuralNetwork.default_instance, os.path.join(t,"network.json"))

    t = self.network.layers[0].input_volume
    self.img_width = t.width
    self.img_height = t.height
    self.img_channels = t.depth

    self.checkpoint_dir = self.proj_path(JG.train_param.target_dir_checkpoint)

    # This information is not available until we have a training set to examine:
    #
    self.image_set_info = None
    self.inf_dir = None

  def log(self, *args):
    if self.verbose or warning("verbosity is on"):
      pr("(verbose:)", *args)


  def prepare_image_info(self):
    if self.image_set_info is not None:
      return
    self.inf_dir = self.proj_path("inference")
    self.image_set_info = read_object(ImageSetInfo.default_instance, os.path.join(self.inf_dir, "image_set_info.json"))


  def proj_path(self, rel_path : str):
    if self.cached_proj_path is None:
      return rel_path
    return os.path.join(self.cached_proj_path, rel_path)


  def prepare_pytorch(self):
    self.log("prepare_pytorch()")

    # Get cpu or gpu device for training.
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.log("PyTorch is using device:",self.device)

    JG.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
      import platform
      sys = platform.system()
      if sys != "Darwin":
        die("CUDA is not available!  System:", sys)

    self.model = self.define_model()
    # Now that model has been constructed, prepare it for use
    self.model.prepare()
    self.model.to(JG.device)



  def define_model(self):
    die("Supply a define_model method")
    return None




  def read_images(self, images_path):
    img_count = self.image_set_info.image_count
    dt = self.network.image_data_type
    images = None
    if dt == DataType.FLOAT32:
      floats_per_image = self.image_set_info.image_length_bytes // BYTES_PER_FLOAT
      images = read_floats(images_path, 0, floats_per_image, img_count)
    elif dt == DataType.UNSIGNED_BYTE:
      bytes_per_image = self.image_set_info.image_length_bytes
      images = read_unsigned_bytes(images_path, 0, bytes_per_image, img_count)
      images = convert_unsigned_bytes_to_floats(images)
    else:
      die("Unsupported image data type:", dt)

    # Convert the numpy array to a pytorch tensor

    # The model wants images with shape (channel, height, width), but Java standard images have
    # shape (height, width, channel), so reshape accordingly
    images = images.reshape((img_count, self.img_height, self.img_width,  self.img_channels))
    images = np.ascontiguousarray(images.transpose(0, 3, 1, 2))
    images = torch.from_numpy(images)
    return images


  def run_inference(self):
    self.restore_checkpoint()
    self.prepare_image_info()
    images_path = os.path.join(self.inf_dir, "images.bin")
    tensor_images = self.read_images(images_path)
    tensor_images = tensor_images.to(self.device)

    self.model.eval()
    with torch.no_grad():
      out_data = self.model(tensor_images)

    show_shape(out_data)
    self.write_results(out_data)


  def write_results(self, tensor):
    results_path = os.path.join(self.inf_dir, "results.bin")
    dt = self.network.label_data_type
    if dt == DataType.FLOAT32:
      labels = tensor.detach().cpu().numpy() # https://stackoverflow.com/questions/49768306/
      #pr("labels:",labels)
      #pr("dtype:",labels.dtype)
      labels.tofile(results_path)
      self.log("results written to",results_path)
    else:
      die("Unsupported label data type:", dt)






  def most_recent_checkpoint(self):
    checkpoint_count = 0
    highest_epoch, best_path = -1, None
    for f in os.listdir(self.checkpoint_dir):
      if f.endswith(".pt"):
        x = chomp(f,".pt")
        checkpoint_count += 1
        epoch = int(x)
        if epoch > highest_epoch:
          highest_epoch, best_path = epoch, os.path.join(self.checkpoint_dir, f)
    return checkpoint_count, best_path


  def construct_checkpoint_path_for_epoch(self, epoch_number):
    return os.path.join(self.checkpoint_dir, f"{epoch_number:06d}.pt")


  def restore_checkpoint(self):
    _, path = self.most_recent_checkpoint()
    if path:
      checkpoint = torch.load(path)
      self.model.load_state_dict(checkpoint['model_state_dict'])
      self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      self.epoch_number = checkpoint['epoch']
      report(f"Restored checkpoint at epoch: {self.epoch_number}")




# Determine if train_set is not None and not the default instance
#
def train_set_defined(train_set:TrainSet):
  return train_set is not None and train_set.directory != ""

