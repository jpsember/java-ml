#!/usr/bin/env python3
import time

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
from gen.special_option import SpecialOption
from pycore.stats import Stats
from pycore.tensor_logger import TensorLogger


class JsTrain:

  def __init__(self, train_script_file):
    self.logger = TensorLogger()
    self.signature = None
    self.verbose = False
    self.device = None
    self.model = None
    self.loss_fn = None
    self.optimizer = None
    self.abort_flag = False
    self.stat_train_loss = Stats("Train Loss")
    self.stat_test_loss = Stats("Test Loss")

    self.epoch_number = 0
    self.snapshot_epoch_interval = 2.0
    self.snapshot_next_epoch = 2

    script_path = os.path.realpath(train_script_file)
    self._proj_path = os.path.dirname(script_path)

    t = self.proj_path("train_info")
    self.train_config = read_object(TrainParam.default_instance, os.path.join(t,"train_param.json"))
    self.network:NeuralNetwork = read_object(NeuralNetwork.default_instance, os.path.join(t,"network.json"))
    self.dump_test_labels_counter = self.train_config.dump_test_labels_count

    t = self.network.layers[0].input_volume
    self.img_width = t.width
    self.img_height = t.height
    self.img_channels = t.depth

    self.train_data_path = self.proj_path("train_data")

    # This information is not available until we have a training set to examine:
    #
    self.train_info = None
    self.train_images = None
    self.batch_size = None
    self.batch_total = None
    self.checkpoint_dir = "checkpoints"

    train_set_count = self.train_config.max_train_sets - 1  # Service tries to provide one more than needed
    check_state(train_set_count > 1)
    self.train_set_list: [TrainSetBuilder] = [None] * train_set_count

    self.last_set_processed = TrainSet.default_instance
    self.last_id_generated = 100 # Set to something nonzero, as the differences are what's important
    self.prev_train_set_dir = None  # directory to be used for testing model

    self.last_checkpoint_epoch = None   # epoch last saved as checkpoint
    self.checkpoint_interval_ms = None  # interval between checkpoints; increases nonlinearly up to a max value
    self.checkpoint_last_time_ms = None # time last checkpoint was written

    self.timeout_length = None
    self.start_time = time_ms()
    self.prev_batch_time = None
    self.recent_image_array = None
    self.recent_model_output = None
    self.image_index = 0
    JG.singleton = self


  def add_timeout(self, max_seconds=60):
    self.timeout_length = max_seconds


  def show_test_labels(self):
    result = (self.dump_test_labels_counter > 0)
    if result:
      self.dump_test_labels_counter -= 1
    return result


  def prepare_train_info(self, train_dir):
    # If train info has not been read yet, read it from the supplied training set directory
    if self.train_info is not None:
      return
    self.train_info = read_object(ImageSetInfo.default_instance, os.path.join(train_dir, "image_set_info.json"))
    self.train_images = self.train_info.image_count
    self.batch_size = min(self.train_config.batch_size, self.train_images)
    if self.train_images % self.batch_size != 0:
      warning("training image count", self.train_images,"is not a multiple of the batch size:", self.batch_size)
    self.batch_total = self.train_images // self.batch_size


  def proj_path(self, rel_path : str):
    if self._proj_path is None:
      return rel_path
    return os.path.join(self._proj_path, rel_path)


  def prepare_pytorch(self):
    self.log("prepare_pytorch()")

    # Get cpu or gpu device for training.
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.log("PyTorch is using device:",self.device)

    self.model = self.define_model()
    self.loss_fn = self.define_loss_function()
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
      recycle_factor = self.train_config.recycle
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


  def stop_signal_received(self):
    x = os.path.join(self.train_data_path,"stop.txt")
    return os.path.isfile(x)


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
        self.quit_session("signature file has changed or is missing")

      # Find set with furthest distance from its last usage,
      # filling in empty slots where possible
      #
      max_dist = -1
      best_cursor = -1

      for i, cursor_object in enumerate(self.train_set_list):
        if train_set_undefined(cursor_object):
          obj = self.find_unclaimed_obj()
          if train_set_defined(obj):
            self.train_set_list[i] = obj
            cursor_object = obj
            self.log("found unclaimed directory:", obj)

        if train_set_undefined(cursor_object):
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

      if total_wait_time > 30:
        self.quit_session("long delay waiting for train data from streaming service")
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

    images = images.reshape((img_count, self.img_channels, self.img_height, self.img_width))
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
      labels = labels.long()
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
    if not train_set_dir:       # Are we still waiting for the stream service?
      return

    self.prepare_train_info(train_set_dir)
    self.prev_train_set_dir = train_set_dir

    train_images_path = os.path.join(train_set_dir, "images.bin")
    train_labels_path = os.path.join(train_set_dir, "labels.bin")

    self.model.train()

    for batch in range(self.batch_total):
      img_index = batch * self.batch_size
      self.log("batch:", batch, "image offset:", img_index)

      tensor_images, tensor_labels = self.read_images(train_images_path, train_labels_path, img_index, self.batch_size)
      tensor_images, tensor_labels = tensor_images.to(self.device), tensor_labels.to(self.device)
      self.optimizer.zero_grad()

      # Compute prediction error
      pred = self.model(tensor_images)


      # Save this model output in case we want to take a snapshot later
      self.recent_model_output = pred

      # See: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
      loss = self.loss_fn(pred, tensor_labels)

      # NOTE: this assumes the loss function returned is independent of the batch size
      # Does reading the loss value mess things up?
      self.stat_train_loss.set_value(loss.item())

      # Backpropagation
      self.model.verify_weights("before loss.backward")
      loss.backward()
      self.model.verify_weights("after loss.backward")

      self.optimizer.step()
      self.model.verify_weights("after optimizer step")


  # Perform optional calculations for the test operation; default does nothing
  #
  def update_test(self, pred, tensor_labels, test_image_count:int):
    pass


  def test(self):
    self.model.eval()

    d = self.prev_train_set_dir
    test_images_path = os.path.join(d, "images.bin")
    test_labels_path = os.path.join(d, "labels.bin")

    test_image_count = min(self.train_info.image_count, self.train_config.test_size)

    with torch.no_grad():
      tensor_images, tensor_labels = self.read_images(test_images_path, test_labels_path, 0, test_image_count)
      tensor_images, tensor_labels = tensor_images.to(self.device), tensor_labels.to(self.device)
      pred = self.model(tensor_images)
      loss = self.loss_fn(pred, tensor_labels).item()
      self.stat_test_loss.set_value(loss)
      self.update_test(pred, tensor_labels, test_image_count)


  def quit_session(self, reason):
    if not self.abort_flag:
      pr("...quitting training session, reason:", reason)
      self.abort_flag = True
      # If signature file exists and its content equals the value we read when training began,
      # delete it to signal that this training session has ended
      p = self.signature_path()
      current_content = txt_read(p, "")
      if current_content == self.signature:
        remove_if_exists(p)


  def with_test(self):
    return self.train_config.test_size > 0


  # Generate a report for the test results.  Default implementation includes the stat_test_loss information
  #
  def test_report(self) -> str:
    return f"  {self.stat_test_loss.info()}"


  # Based upon the test results, determine if a training target has been reached.
  # Default implementation returns False
  #
  def test_target_reached(self) -> bool:
    return False


  def run_training_session(self):
    self.restore_checkpoint()
    self.last_checkpoint_epoch = self.epoch_number
    done_msg = None

    while not self.abort_flag:
      self.train()
      if self.abort_flag:
        break
      self.perform_delay()
      s = f"Epoch {self.epoch_number:4}   {self.stat_train_loss.info()}"
      if self.stat_train_loss.value_sm <= self.train_config.target_loss:
        done_msg = "Train loss reached target"
      if self.with_test():
        self.test()
        self.perform_delay()
        if self.test_target_reached():
          done_msg = "Test accuracy reached target"
        s += "   " + self.test_report()
      pr(s)
      self.epoch_number += 1
      if self.stop_signal_received():
        done_msg = "Stop signal received"

      next_snapshot_epoch = int(self.snapshot_next_epoch)
      if self.epoch_number >= next_snapshot_epoch:
        # Early on, generate more frequent snapshots
        if self.epoch_number > 20:
          self.snapshot_epoch_interval *= 1.2
        self.snapshot_next_epoch = self.epoch_number + self.snapshot_epoch_interval
        pr("Saving model inference snapshot")
        self.send_inference_result()

      current_time = time_ms()
      if not self.checkpoint_last_time_ms:
        self.checkpoint_interval_ms = 30000
        self.checkpoint_last_time_ms = current_time

      ms_until_save = (self.checkpoint_last_time_ms + self.checkpoint_interval_ms) - current_time
      if ms_until_save <= 0:
        self.save_checkpoint()
        self.send_inference_result()

      if self.update_timeout():
        done_msg = "Timeout expired"

      if done_msg:
        self.save_checkpoint()
        self.quit_session(done_msg)


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
    todo("have symbolic constants for special handling")
    t.special_handling = 1
    t.family_id = img_index
    t.family_size = 2
    t.family_slot = 0

    t.name = "image"
    tens = self.ndarray_to_tensor(self.recent_image_array, t)
    self.logger.add(tens, t)

    t.name = "labels"
    t.family_slot = 1
    tens = self.recent_model_output
    self.logger.add(tens, t)


  def ndarray_to_tensor(self, ndarr, ti:LogItemBuilder):
    tens =  torch.from_numpy(ndarr)
    return tens



  def update_timeout(self)->bool:
    if self.timeout_length is None:
      return False
    return time_ms() >= self.start_time + self.timeout_length * 1000


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
      pr("Restored checkpoint at epoch:", self.epoch_number)


  def save_checkpoint(self):
    diff = self.epoch_number - self.last_checkpoint_epoch
    check_state(diff >= 0,"epoch number less than last saved")
    # Don't save a checkpoint if we haven't done some minimum number of epochs
    if diff <= 3:
      if diff > 0:
        pr("(...not bothering to save checkpoint for only", diff, "new epochs)")
      return

    path = self.construct_checkpoint_path_for_epoch(self.epoch_number)
    check_state(not os.path.exists(path), "checkpoint already exists")
    pr("Saving checkpoint:",path)
    # Save to a temporary file and rename afterward, to avoid leaving partially written files around in
    # case user quits program or something
    path_tmp = path + ".tmp"
    torch.save({
                'epoch': self.epoch_number,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
                }, path_tmp)
    os.rename(path_tmp, path)
    self.last_checkpoint_epoch = self.epoch_number
    self.checkpoint_last_time_ms = time_ms()
    self.checkpoint_interval_ms = min(int(self.checkpoint_interval_ms * 1.2), 10 * 60 * 1000)


  def log(self, *args):
    if self.verbose:
      pr("(verbose:)", *args)


  def perform_delay(self):
    t = self.train_config.min_batch_time
    if t <= 0:
      return
    warning("Imposing minimum batch time:", t)
    c = time_ms()
    p = none_to(self.prev_batch_time, c)
    self.prev_batch_time = c
    if c - p < t:
      time.sleep(t / 1000.0)


# Determine if train_set is not None and not the default instance
#
def train_set_defined(train_set:TrainSet):
  return train_set is not None and train_set.directory != ""

# Negation of train_set_defined
#
def train_set_undefined(train_set:TrainSet):
  return not train_set_defined(train_set)
