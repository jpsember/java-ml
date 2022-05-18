from pycore.pytorch_util import *
from gen.neural_network import NeuralNetwork
from pycore.printsize import *

# The model for the classifier_train.py program
#
class ClassifierModel(nn.Module):


  def __init__(self, network: NeuralNetwork):
    super(ClassifierModel, self).__init__()

    self.tensors = []
    self.add_size("image input")

    self.layer = None
    for lyr in network.layers:
      self.layer = lyr

      if lyr.type == "conv":
        our_stride = lyr.stride.tuple()
        t = nn.Conv2d(in_channels=lyr.input_volume.depth,
                      out_channels=lyr.output_volume.depth,
                      kernel_size=lyr.kernel_width,
                      stride=our_stride,
                      padding=lyr.kernel_width//2, # half padding?
                      )
        todo("confirm stride parameter order")
        self.add_layer(t)
      elif lyr.type == "leaky_relu":
        self.add_layer(nn.LeakyReLU())
        todo("what parameters do we need to pass to LeakyRelU?")
      elif lyr.type == "maxpool":
        # I am used to thinking of a maxpool has having a stride, but they expect a kernel_size parameter
        stride = none_to(lyr.stride, network.stride)
        if warning("using a constant stride of 2"):
          stride = 2
        self.add_layer(nn.MaxPool2d(kernel_size=stride))
      elif lyr.type == "fc":
        self.add_layer(self.construct_fc())
      elif lyr.type == "output":
        x = self.construct_output()
        if x is not None:
          self.add_layer(x)
        else:
          self.add_size(lyr.type)
      else:
        die("unsupported layer type:",lyr.type)
    self.layer = None

    self.stop_size_messages()
    self.layers = nn.Sequential(*self.tensors)


  def add_size(self, message):
    self.tensors.append(PrintSize(message))


  def stop_size_messages(self):
    self.add_size("!stop!")


  def exit_when_built(self):
    self.add_size("!exit!")


  def add_layer(self, layer, size_label=None):
    if layer is not None:
      self.add_size(none_to(size_label, self.layer.type))
      self.tensors.append(layer)


  def last_tensor(self):
    return self.tensors[-1]


  def construct_fc(self):
    # We need to 'convert' the input volume into a set of confidences, one for each category.
    #
    # Flatten the input volume into a fibre, then apply a linear layer?
    #
    # See: https://stackoverflow.com/a/60372416
    #
    # Flatten:  https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
    #
    ly = self.layer
    self.add_layer(nn.Flatten(),"fc.Flatten")
    self.add_layer(nn.Linear(vol_volume(ly.input_volume), ly.filters),"fc.Linear")


  def construct_output(self):
    t = None
    in_vol = self.layer.input_volume
    # Reshape to fibre, if necessary
    #
    if in_vol.width != 1 or in_vol.height != 1:
      t = nn.reshape(self.last_tensor(), (-1,))
    return t


  def reshape_if_nec(self, in_tensor):
    t = None
    size = in_tensor.size()
    error_unless(len(size) == 3, "tensor is not 3-dim")
    if size[0] != 1 or size[1] != 1:
      t = nn.reshape(in_tensor,(-1,))
    return t


  def forward(self, x):
    return self.layers(x)
