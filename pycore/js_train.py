#!/usr/bin/env python3

from pycore.pytorch_util import *
import os
import os.path
from gen.image_set_info import *
from gen.train_set import *
from gen.data_type import *
from gen.compile_images_config import *
from gen.train_param import *
from pycore.stats import Stats


class JsTrain:

  def __init__(self, train_script_file):
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

    script_path = os.path.realpath(train_script_file)
    self._proj_path = os.path.dirname(script_path)

    t = self.proj_path("model_data")
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

    t = self.proj_path("compileimages-args.json")
    config:CompileImagesConfig = read_object(CompileImagesConfig.default_instance, t)
    self.checkpoint_dir = config.target_dir_checkpoint

    train_set_count = self.train_config.max_train_sets - 1  # Service tries to provide one more than needed
    check_state(train_set_count > 1)
    self.train_set_list: [TrainSetBuilder] = [None] * train_set_count

    self.last_set_processed = TrainSet.default_instance
    self.last_id_generated = 100 # Set to something nonzero, as the differences are what's important
    self.prev_train_set_dir = None  # directory to be used for testing model

    self.last_checkpoint_epoch = None   # epoch last saved as checkpoint
    self.checkpoint_interval_ms = None  # interval between checkpoints; increases nonlinearly up to a max value
    self.checkpoint_last_time_ms = None # time last checkpoint was written


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

    self.loss_fn = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum = 0.9)


  def define_model(self):
    die("Supply a define_model method")
    return None


  # -------------------------------------------------------------------------------------
  # For logging purposes only
  #

  def show_train_set_elem(self, elem) -> [str]:
    if elem is None:
      return "<none>"
    return base_name(elem.directory) + ":" + str(elem.used)


  def show_train_set(self) -> [str]:
    x = []
    for idx, y in enumerate(self.train_set_list):
      s = self.show_train_set_elem(y)
      x.append(s)
    return x

  #
  # -------------------------------------------------------------------------------------


  def discard_stale_train_sets(self):
    # Discard any sets that have already been used the max number of times
    for idx, x in enumerate(self.train_set_list):
      if x is None:
        continue
      recycle_factor = self.train_config.recycle
      check_state(x.used <= recycle_factor)
      if x.used == recycle_factor:
        self.log("discarding stale:", self.show_train_set_elem(x))
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


  def discard_signature(self):
    p = self.signature_path()
    remove_if_exists(p)


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
    self.log("looking for training set; list:",self.show_train_set())

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


  def labels_are_ints(self):
    die("Add implementation for labels_are_ints")
    return False


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

      # Read image, converting to floats if necessary
      #

      dt = self.network.image_data_type
      if dt == DataType.FLOAT32:
        floats_per_image = self.train_info.image_length_bytes // BYTES_PER_FLOAT
        images = read_floats(train_images_path, floats_per_image * img_index, floats_per_image, self.batch_size)
      elif dt == DataType.UNSIGNED_BYTE:
        bytes_per_image = self.train_info.image_length_bytes
        images = read_bytes(train_images_path, bytes_per_image * img_index, bytes_per_image, self.batch_size)
        pr("images type:",type(images))
        pr("bytes per image:",bytes_per_image)
        pr("length:",len(images))
        # Convert bytes to floats, where 0=0.0, 255=1.0
        #
        images = images.astype(np.float32)
        todo("scale from 0..255 to 0.0 ... 1.0")
        pr("images type:",type(images))
        pr("length:",len(images))
        pr("shape:",images.size)
      else:
        die("Unsupported image data type:", dt)

      # Convert the numpy array it returned to a pytorch tensor
      #
      images = images.reshape((self.batch_size, self.img_channels, self.img_height, self.img_width))
      tensor_images = torch.from_numpy(images)

      if self.labels_are_ints():
        record_size = self.train_info.label_length_bytes // BYTES_PER_INT
        labels = read_ints(train_labels_path, img_index, record_size, self.batch_size)
        warning("not sure why this reshape is necessary, or why the one in floats is failing")
        labels = labels.reshape(self.batch_size)
        tensor_labels = torch.from_numpy(labels)
        tensor_labels = tensor_labels.long()
      else:
        record_size = self.train_info.label_length_bytes // BYTES_PER_FLOAT
        labels = read_floats(train_labels_path, img_index, record_size, self.batch_size)
        pr("read floats, shape:",labels.shape)
        labels = labels.reshape(self.batch_size)
        pr("after further reshape:",labels.shape)
        tensor_labels = torch.from_numpy(labels)

      tensor_images, tensor_labels = tensor_images.to(self.device), tensor_labels.to(self.device)

      # Compute prediction error
      pred = self.model(tensor_images)

      # See: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
      loss = self.loss_fn(pred, tensor_labels)

      # NOTE: this assumes the loss function returned is independent of the batch size
      # Does reading the loss value mess things up?
      self.stat_train_loss.set_value(loss.item())

      # Backpropagation
      self.optimizer.zero_grad()
      loss.backward()

      self.optimizer.step()


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
      floats_per_image = self.train_info.image_length_bytes // BYTES_PER_FLOAT
      images = read_floats(test_images_path, 0, floats_per_image, test_image_count)

      # Convert the numpy array it returned to a pytorch tensor
      #
      images = images.reshape((test_image_count, self.img_channels, self.img_height, self.img_width))
      tensor_images = torch.from_numpy(images)
      labels = read_ints(test_labels_path, 0, 1, test_image_count)
      labels = labels.reshape(test_image_count)
      tensor_labels = torch.from_numpy(labels)

      # We need the tensor labels to be 64-bits, just as we did for training
      tensor_labels = tensor_labels.long()
      tensor_images, tensor_labels = tensor_images.to(self.device), tensor_labels.to(self.device)
      pred = self.model(tensor_images)
      loss = self.loss_fn(pred, tensor_labels).item()
      self.stat_test_loss.set_value(loss)
      self.update_test(pred, tensor_labels, test_image_count)


  def quit_session(self, reason):
    if not self.abort_flag:
      pr("...quitting training session, reason:", reason)
      self.abort_flag = True
      self.discard_signature()  # so streaming service stops as well


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
      s = f"Epoch {self.epoch_number:4}   {self.stat_train_loss.info()}"
      if self.stat_train_loss.value_sm <= self.train_config.target_loss:
        done_msg = "Train loss reached target"
      if self.with_test():
        self.test()
        if self.test_target_reached():
          done_msg = "Test accuracy reached target"
        s += "   " + self.test_report()
      pr(s)
      self.epoch_number += 1

      current_time = time_ms()
      if not self.checkpoint_last_time_ms:
        self.checkpoint_interval_ms = 30000
        self.checkpoint_last_time_ms = current_time

      ms_until_save = (self.checkpoint_last_time_ms + self.checkpoint_interval_ms) - current_time
      if ms_until_save <= 0:
        self.save_checkpoint()

      if done_msg:
        self.save_checkpoint()
        self.quit_session(done_msg)


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


# Determine if train_set is not None and not the default instance
#
def train_set_defined(train_set:TrainSet):
  return train_set is not None and train_set.directory != ""

# Negation of train_set_defined
#
def train_set_undefined(train_set:TrainSet):
  return not train_set_defined(train_set)
