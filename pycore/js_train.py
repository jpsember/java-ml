#!/usr/bin/env python3

import torch.cuda

from gen.cmd_item import CmdItem
from pycore.jg import JG
from pycore.pytorch_util import *
import os
import os.path
from gen.image_set_info import *
from gen.train_set import *
from gen.data_type import *
from gen.compile_images_config import *
from gen.train_param import *
from gen.log_item import *
from gen.special_handling import *
from gen.special_option import SpecialOption
from pycore.tensor_logger import TensorLogger


class JsTrain:

  def __init__(self, model_specific_script_file):
    self.signature = None
    self.verbose = False
    self.device = None
    self.model = None
    self.loss_fn = None
    self.optimizer = None
    self.train_loss = None
    self.done_msg:str = None

    self.epoch_number = 0
    self.snapshot_epoch_interval: float = None
    self.snapshot_next_epoch: int = None

    script_path = os.path.realpath(model_specific_script_file)
    self.cached_proj_path = os.path.dirname(script_path)

    t = self.proj_path("train_info")
    JG.train_param = read_object(TrainParam.default_instance, os.path.join(t, "train_param.json"))
    self.network:NeuralNetwork = read_object(NeuralNetwork.default_instance, os.path.join(t,"network.json"))

    t = self.network.layers[0].input_volume
    self.img_width = t.width
    self.img_height = t.height
    self.img_channels = t.depth

    self.train_data_path = self.proj_path(JG.train_param.target_dir_train)

    # This information is not available until we have a training set to examine:
    #
    self.train_info = None
    self.train_images = None
    self.batch_size = None
    self.batch_total = None
    self.checkpoint_dir = self.proj_path(JG.train_param.target_dir_checkpoint)

    train_set_count = JG.train_param.max_train_sets - 1  # Service tries to provide one more than needed
    check_state(train_set_count > 1)
    self.train_set_list: [TrainSetBuilder] = [None] * train_set_count

    self.last_set_processed = TrainSet.default_instance
    self.last_id_generated = 100 # Set to something nonzero, as the differences are what's important
    self.prev_train_set_dir = None  # directory to be used for testing model

    self.start_time = time_ms()
    self.prev_batch_time = None
    self.recent_image_array = None
    self.recent_model_output = None
    self.image_index = 0


  def prepare_train_info(self, train_dir):
    # If train info has not been read yet, read it from the supplied training set directory
    if self.train_info is not None:
      return
    self.train_info = read_object(ImageSetInfo.default_instance, os.path.join(train_dir, "image_set_info.json"))
    self.train_images = self.train_info.image_count
    self.batch_size = min(JG.train_param.batch_size, self.train_images)
    if self.train_images % self.batch_size != 0:
      warning("training image count", self.train_images,"is not a multiple of the batch size:", self.batch_size)
    self.batch_total = self.train_images // self.batch_size
    pr("batch size:",self.batch_size)


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
    self.loss_fn = self.define_loss_function()
    todo("add ability to calculate and report the accuracy")
    # see https://discuss.pytorch.org/t/how-to-calculate-accuracy-in-pytorch/80476/2
    # e.g. train_acc = torch.sum(y_pred == target)
    #      accuracy = train_acc/number_of_datapoints

    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum = 0.9)


  def define_loss_function(self):
    die("Supply a loss function")
    return None


  def define_model(self):
    die("Supply a define_model method")
    return None


  def discard_stale_train_sets(self):
    # Discard any sets that have already been used the max number of times
    for idx, x in enumerate(self.train_set_list):
      if x is None:
        continue
      recycle_factor = JG.train_param.recycle
      check_state(x.used <= recycle_factor)
      if x.used == recycle_factor:
        delete_directory(x.directory, "set_")
        self.train_set_list[idx] = None


  def signature_changed(self):
    current_content = txt_read(self.signature_path(),"")
    if current_content == "":
      return True
    if not self.signature:
      self.signature = current_content
    return self.signature != current_content


  def signature_path(self):
    return os.path.join(self.train_data_path,"sig.txt")


  # Look for a directory that we haven't processed yet; return TrainSetBuilder representing it, or None
  #
  def find_unclaimed_obj(self):
    for f in os.listdir(self.train_data_path):
      if f.startswith("set_"):
        key = os.path.join(self.train_data_path, f)
        claimed = False
        for x in self.train_set_list:
          if x and x.directory == key:
            claimed = True
            break
        if not claimed:
          return TrainSet.new_builder().set_directory(key)
    return None


  # Choose a data set to use for the next epoch of training.  Wait for one to appear if necessary
  #
  def select_data_set(self):

    self.discard_stale_train_sets()

    total_wait_time = 0
    while True:
      if self.signature_changed():
        self.set_done_msg("signature file has changed or is missing")
        return None

      # Find set with furthest distance from its last usage,
      # filling in empty slots where possible
      #
      max_dist = -1
      best_cursor = -1

      for i, cursor_object in enumerate(self.train_set_list):
        if not train_set_defined(cursor_object):
          obj = self.find_unclaimed_obj()
          if train_set_defined(obj):
            self.train_set_list[i] = obj
            cursor_object = obj
            self.log("found unclaimed directory:", obj)

        if not train_set_defined(cursor_object):
          continue

        # This can't be the one we last used
        #
        if cursor_object.id == self.last_id_generated:
          continue

        dist = len(self.train_set_list)
        if cursor_object.id != 0:
          dist = min(dist, self.last_id_generated - cursor_object.id)
        if dist > max_dist:
          max_dist = dist
          best_cursor = i

      if best_cursor >= 0:
        train_set = self.train_set_list[best_cursor]
        break

      if total_wait_time > 120:
        self.set_done_msg("long delay waiting for train data from streaming service")
        return None

      self.log("...sleeping")
      sleep_time = 0.2
      time.sleep(sleep_time)
      total_wait_time += sleep_time

    self.last_id_generated += 1
    train_set.id = self.last_id_generated
    train_set.used = train_set.used + 1
    self.log("found train set:", base_name(train_set.directory), "used:", train_set.used)
    return train_set.directory


  # Helper function to read images and labels, for either training or testing
  #
  def read_images(self, images_path, labels_path, img_index, img_count):
    dt = self.network.image_data_type
    images = None
    if dt == DataType.FLOAT32:
      floats_per_image = self.train_info.image_length_bytes // BYTES_PER_FLOAT
      images = read_floats(images_path, floats_per_image * img_index, floats_per_image, img_count)
      self.recent_image_array = images
    elif dt == DataType.UNSIGNED_BYTE:
      bytes_per_image = self.train_info.image_length_bytes
      images = read_unsigned_bytes(images_path, bytes_per_image * img_index, bytes_per_image, img_count)
      self.recent_image_array = images
      images = convert_unsigned_bytes_to_floats(images)
    else:
      die("Unsupported image data type:", dt)

    # Convert the numpy array to a pytorch tensor

    # The model wants images with shape (channel, height, width), but Java standard images have
    # shape (height, width, channel), so reshape accordingly
    images = images.reshape((img_count, self.img_height, self.img_width,  self.img_channels))
    images = np.ascontiguousarray(images.transpose(0, 3, 1, 2))

    images = torch.from_numpy(images)
    if self.network.special_option == SpecialOption.PIXEL_ALIGNMENT:
      pr("Performing special option: PIXEL_ALIGNMENT")
      prob_count = 0
      for y in range(self.img_height):
        for x in range(self.img_width):
          for c in range(self.img_channels):
            pv = int(images[0, c, y, x] * 255.0)
            expected = (y * 7 + x * 13 + (c+1)) & 0xff
            if pv != expected:
              prob_count += 1
              pr("Problem with pixel c=",c,"x=",x,"y=",y,"value is",pv,", expected",expected)
              if prob_count == 20:
                halt("lots of problems")
      halt("Stopping, done special option")

    dt = self.network.label_data_type
    if dt == DataType.UNSIGNED_BYTE:
      record_size = self.train_info.label_length_bytes
      labels = read_unsigned_bytes(labels_path, img_index * record_size, record_size, img_count)
      labels = labels.reshape(img_count)
      labels = torch.from_numpy(labels)
      labels = labels.float()
      warning("converting datatype to float32 hopefully")
    elif dt == DataType.FLOAT32:
      record_size_floats = self.train_info.label_length_bytes // BYTES_PER_FLOAT
      labels = read_floats(path=labels_path, offset_in_floats=img_index * record_size_floats,
                  record_size_in_floats=record_size_floats, record_count=img_count)
      labels = torch.from_numpy(labels)
    else:
      die("Unsupported label data type:", dt)
    return images, labels


  def train(self):
    train_set_dir = self.select_data_set()
    if not train_set_dir:       # Are we still waiting for the stream service, or quit flag set?
      return

    self.prepare_train_info(train_set_dir)
    self.prev_train_set_dir = train_set_dir

    train_images_path = os.path.join(train_set_dir, "images.bin")
    train_labels_path = os.path.join(train_set_dir, "labels.bin")

    self.model.train()

    for batch in range(self.batch_total):
      img_index = batch * self.batch_size
      self.log("batch:", batch, "image offset:", img_index)
      JG.batch_number = batch
      tensor_images, tensor_labels = self.read_images(train_images_path, train_labels_path, img_index, self.batch_size)
      tensor_images, tensor_labels = tensor_images.to(self.device), tensor_labels.to(self.device)
      self.optimizer.zero_grad()
      # Compute prediction error
      pred = self.model(tensor_images)

      # Save this model output in case we want to take a snapshot later
      self.recent_model_output = pred

      # See: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
      todo("convenience methods to dump tensors, other pytorch structures")
      show("pred",pred)
      show("tensor_labels",tensor_labels)
      todo("tensor_labels should be converted to float32?")
      loss = self.loss_fn(pred, tensor_labels)
      self.train_loss = loss.item()

      # Backpropagation
      loss.backward()

      if JG.train_param.with_gradient_norm:
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
      self.optimizer.step()



  def done_flag_set(self):
    return self.done_msg is not None


  def set_done_msg(self, msg: str):
    if not self.done_flag_set():
      self.done_msg = msg


  def run_training_session(self):
    self.restore_checkpoint()

    while not self.done_flag_set():
      self.train()
      if self.done_flag_set():
        break

      # Try our new logging
      stats_map = {
        "epoch":self.epoch_number,
        "loss" : self.train_loss,
      }

      # If anyone stored additional stats in JG.aux_stats, include them and clear it
      #
      if JG.aux_stats is not None:
        stats_map.update(JG.aux_stats)
        JG.aux_stats = None
      TensorLogger.default_instance.add_stats(stats_map)

      self.epoch_number += 1
      self.update_snapshots()

      self.process_java_commands()

    self.save_checkpoint()


  def update_snapshots(self):
    if not JG.train_param.generate_snapshots:
      return
    if self.snapshot_next_epoch is None:
      self.snapshot_epoch_interval = 2.0
      self.snapshot_next_epoch = 2
    next_snapshot_epoch = int(self.snapshot_next_epoch)
    if self.epoch_number >= next_snapshot_epoch:
      self.snapshot_epoch_interval *= 1.2
      self.snapshot_next_epoch = self.epoch_number + self.snapshot_epoch_interval
      report("Saving model inference snapshot")
      self.send_inference_result()


  # Send image input, labelled output to streaming service
  #
  def send_inference_result(self):
    # Use the epoch number as the image index
    img_index = self.epoch_number
    # If we already sent this epoch's result, do nothing
    if img_index <= self.image_index:
      return
    self.image_index = img_index

    t = LogItem.new_builder()
    t.special_handling = SpecialHandling.SNAPSHOT
    t.family_id = img_index
    t.family_size = 2
    t.family_slot = 0

    t.name = "image"
    tens = torch.from_numpy(self.recent_image_array)
    TensorLogger.default_instance.add(tens, t)

    t.name = "labels"
    t.family_slot = 1
    tens = self.recent_model_output
    TensorLogger.default_instance.add(tens, t)


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


  def save_checkpoint(self):
    if self.epoch_number <= 0:
      warning("ignoring attempt to save epoch:",self.epoch_number)
      return
    
    path = self.construct_checkpoint_path_for_epoch(self.epoch_number)
    if os.path.exists(path):
      return
    pr("Saving checkpoint:",path)
    # Save to a temporary file and rename afterward, to avoid leaving partially written files around in
    # case user quits program or something
    path_tmp = temp_version(path)
    torch.save({
                'epoch': self.epoch_number,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
                }, path_tmp)
    os.rename(path_tmp, path)


  def log(self, *args):
    if self.verbose:
      pr("(verbose:)", *args)


  def process_java_commands(self):
    for f in os.listdir(self.train_data_path):
      if f.endswith(".pcmd"):
        path = os.path.join(self.train_data_path, f)
        cmd:CmdItem = read_object(CmdItem.default_instance, path)

        a = cmd.args[0]
        if a == "checkpoint":
          self.save_checkpoint()
        elif a == "stop":
          self.set_done_msg("stop command received")
        else:
          die("Unrecognized command:", cmd)

        os.remove(path)




# Determine if train_set is not None and not the default instance
#
def train_set_defined(train_set:TrainSet):
  return train_set is not None and train_set.directory != ""

