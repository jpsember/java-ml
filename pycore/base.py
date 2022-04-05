#!/usr/bin/env python3

import re
import os
import time
import traceback
import json
import jstyleson   # pip install jstyleson
import inspect


# Define some constants
#
BYTES_PER_FLOAT = 4
BYTES_PER_INT = 4


class _StackTrace:
  """
  Sentinel object for generating stack traces via st()
  """
  def __str__(self):
    return exception_info(None, 4, 8)


def st():
  return _StackTrace()


class _LocalVars:
  """
  Used to produce an instance to hold the tools' local variables
  """

  def __init__(self):
    self.rxFilenameOnly = re.compile(".*/([^/]*)$")
    self.repMap = set()
    self.cmd_line_args = None
    self.ca_args = None
    self.ca_extras = None
    self.home_dir = None
    self.app = None
    self.app_oper_map = None
    self.command_count = 0

# Construct a singleton instance for the static variables
_v = _LocalVars()
_v.arg_value_map = None

def pr(*args):
  """
  Print objects to stdout
  """
  buffer = spr(*args)
  print(buffer, end='', flush=True)


def spr(*args):
  """
  Print objects to a string
  """
  buffer = ''
  mid_line = False
  for x in args:
    s = d(x)
    has_cr = "\n" in s
    if has_cr:
      s2 = s.rstrip()
      if mid_line:
        buffer = buffer + "\n"
      buffer = buffer + s2
      buffer = buffer + "\n"
      mid_line = False
    else:
      if mid_line:
        buffer = buffer + ' '
      buffer = buffer + s
      mid_line = True
  if mid_line:
    buffer = buffer + "\n"
  return buffer


def sprintf(fmt, *args):
  return fmt % args


def d(s):
  """
  Convert an object to a string (by calling str(x)), or returns '<None>' if
  object is None
  """
  if s is None:
    return "<None>"
  elif isinstance(s, dict):
    return pretty_pr(s)
  elif isinstance(s, float):
    return df(s)
  else:
    return str(s)


def df(value):
  """Convert a float to a string, with fixed-width format"""
  s = "{:9.3f}".format(value)
  if s.endswith(".000"):
    s = s[:-4] + "    "
  return s

def parse_int(string, default=None):
  result = default
  try:
    result = str(string)
  except ValueError:
    pass
  if result is None:
    die("Failed to parse integer from string '"+string+"'")
  return result

class OurJSONEncoder(json.JSONEncoder):
  def default(self, obj):
    if hasattr(obj, "to_json"):
      return obj.to_json()
    if type(obj).__name__ == "memmap":
      return obj.__str__().split("\n")
    return vars(obj)


def to_json(x):
  return json.dumps(x, sort_keys=True, cls=OurJSONEncoder)


def pretty_pr(s):
  return json.dumps(s, sort_keys=True, indent=4, cls=OurJSONEncoder)


def dtype(s):
  """
  Print string representation of object (with d(x)), followed by type name
  """
  if s is None:
    return '<None>'
  else:
    return d(s) + " (type=" + str(s.__class__.__name__) + ")"


def simple_name(filename, line):
  """ Builds a nice string from a pathname and line number; removes directories from pathname.  """
  m = _v.rxFilenameOnly.match(filename)
  if m:
    filename = m.group(1)
  return "(" + filename.ljust(12) + str(line).rjust(4) + ")"


def one_time_alert(skip_count, prompt_msg, *args):
  """Prints an alert, if hasn't yet been printed."""
  loc = get_caller_location(skip_count + 2)
  s = "*** " + prompt_msg + " " + loc + ": "
  if len(args):
    msg = s + spr(*args)
  else:
    msg = s
  _one_time_only(msg)
  return True


def _one_time_only(msg):
  if msg not in _v.repMap:
    _v.repMap.add(msg)
    print(msg, end='', flush=True)


def clear_screen(title = None):
  os.system('clear')
  if title:
    pr(title)
    pr("==============================================================================================")


def opt_value(dictionary, key, default_value=None):
  if key in dictionary:
    return dictionary[key]
  if default_value is None:
    die("Missing key:",key)
  return default_value


def warning(*args):
  one_time_alert(1, "WARNING", *args)
  return True

def todo(*args):
  one_time_alert(1, "TODO", *args)
  return True



def get_caller_location(skip_count=2):
  h = inspect.stack()
  if 0 <= skip_count < len(h):
    fi = h[skip_count]  # inspect.getframeinfo(h[1])
    loc = simple_name(fi[1], fi[2])  # fi.__file__,fi.f_lineno)
  else:
    loc = "(UNKNOWN LOCATION)"
  return loc


def my_assert(cond):
  one_time_alert(1, "checking assertion")
  error_if(not cond)


def not_none(arg):
  """
  Raise exception if argument is 'None';
  otherwise, return the argument
  """
  if arg is None:
    error("Argument is 'None'")
  return arg


def check_state(cond, *args):
  if cond:
    return
  error(_log_msg("Illegal state", *args))


def error(msg=None):
  """
  Raise an exception, optionally with a message
  """
  if msg is None:
    msg = "error occurred"
  raise Exception(msg)


def error_if(cond, msg=None):
  """
  Raise an exception, optionally with a message, if a condition is true
  """
  print("...refactor to use 'check_state' instead of 'error_if'...")
  if cond:
    error(msg)


def error_unless(cond, msg=None):
  error_if(not cond, msg)


def die(*args):
  error(_log_msg("...exiting program!", *args))


def halt(*args):
  loc = get_caller_location(2)
  s = "...at " + loc + ":" + _log_msg("   halting program early (development only)", *args)
  print(s)
  quit()


def _log_msg(default_msg, *args):
  if len(args) == 0:
    args = [default_msg]
  return spr(*args)


def current_dir():
  """
  Get absolute path of current directory, as unicode string
  """
  return os.path.abspath(os.curdir)


_last_timer_mark = None


def time_ms():
  return int(round(time.time() * 1000))


def timer_start(message=None):
  global _last_timer_mark
  _last_timer_mark = time_ms()
  output = ""
  if message:
    output = "(Starting timer: " + message + ")"
  return output


def timer_mark(message=None):
  global _last_timer_mark
  if not _last_timer_mark:
    return timer_start(message)

  current = time_ms()
  diff = current - _last_timer_mark
  _last_timer_mark = current
  message_expr = ""
  if message:
    message_expr = " : " + message
  return "({:6.3f}s {})".format(diff / 1000.0, message_expr)


def report_exception(e, skip=1, limit=20):
  message = exception_info(e, skip + 1, limit)
  pr(message)


def exception_info(e, skip=1, limit=20):
  """
  Get subset of stack trace from exception
  """
  if e is None:
    info = traceback.extract_stack(None, limit + skip)
  else:
    info = traceback.extract_tb(e.__traceback__)
  info = info[skip:skip + limit]

  # Build list of prefixes we'll omit, including lines within system libraries
  # that we'll omit altogether
  #
  prefix_map = {
    home_dir() : home_dir() + "/",
    "/usr" : "!"
  }

  # Filter stack trace entries
  #
  # Construct list of lists [location, text]
  #                      or [skip count, None]
  #
  filtered = []
  omit_count = 0

  for x in info:
    f2 = x.filename

    for key, val in prefix_map.items():
      if x.filename.startswith(key):
        if val == "!":
          f2 = None
        else:
          if x.filename.startswith(val):
            f2 = x.filename[len(val):]
    if f2 is None:
      omit_count += 1
      continue

    if omit_count > 0:
      filtered.append([omit_count, None])
      omit_count = 0

    line_info = x.line
    loc_str = '{:s} ({:d})'.format(f2, x.lineno)
    filtered.append([loc_str, line_info])

  if omit_count > 0:
    filtered.append([omit_count, None])

  # Determine maximum length of line info,
  # and replace skip counts with appropriate text
  #
  max_len = 20
  for x in filtered:
    if x[1] is None:
      x[0] = '...{:d} omitted...'.format(x[0])
    max_len = max(max_len, len(x[0]))

  result = ""
  if e:
    result += "=" * 125 + "\n"
  fmt = '{:' + str(max_len) + 's} : {:s}'
  for elem in filtered:
    loc_str, line = elem
    if line is None:
      s = loc_str
    else:
      s = fmt.format(loc_str, line)
    result = result + s + "\n"

  if e:
    result += "***\n"
    as_str = str(e)
    # Deal with exception chaining; remove excess comments
    clip = as_str.find("Caused by")
    if clip > 0:
      as_str = as_str[0:clip]
    result += "*** " + as_str.strip()
  return result


def ca_builder(app=None):
  c = _v.cmd_line_args
  if c is None:
    _v.app = app
    _v.app_oper_map = {}
    if app:
      _v.app_oper_map = _determine_app_oper_map(app)
    import argparse
    from argparse import RawDescriptionHelpFormatter
    epilog = None
    if len(_v.app_oper_map) != 0:
      help_messages = "Operations include:\n"
      for oper_key in _v.app_oper_map:
        oper_map = _v.app_oper_map[oper_key]
        msg = " " + oper_key
        help_fn = oper_map.get("help")
        if help_fn:
          msg += ": "
          msg += " " * max(20 - len(msg), 0)
          msg += help_fn()
        help_messages += msg + "\n"
      epilog = help_messages
    c = argparse.ArgumentParser(epilog=epilog, formatter_class=RawDescriptionHelpFormatter)
    _v.cmd_line_args = c
  return c


def ca_args():
  if _v.ca_args is None:
    _v.ca_args, _v.ca_extras = ca_builder().parse_known_args()
    for s in _v.ca_extras:
      if s.startswith("-"):
        die("Unknown argument:", s)
  return _v.ca_args


def assert_args_done():
  ca_args()
  extras = _v.ca_extras
  if len(extras) != 0:
    die("extraneous arguments:", *extras)


def has_next_arg():
  """
  Return true iff more arguments exist
  """
  ca_args()
  return len(_v.ca_extras) != 0


def next_arg(default_value=None):
  """
  Read next argument; if none exist, return default value; fail if none provided
  """
  if has_next_arg():
    arg = _v.ca_extras.pop(0)
  else:
    if default_value is None:
      die("missing argument")
    arg = default_value
  return arg


def _parse_value_to_match_type(value, value_of_type):
  if isinstance(value_of_type, float):
    return float(value)
  if isinstance(value_of_type, int):
    return int(value)
  return value


def next_arg_if(name, default_value=False):
  """
  If next argument doesn't exist, or doesn't match the provided name, return the default value.
  Otherwise, read the next argument, and:
   If default value is false, return true (i.e. it's a boolean flag argument);
   Otherwise:
     read the next argument (failing if there isn't one)
     parse its argument based on the type of the default value
     return the parsed value
  """
  effective_default = _v.arg_value_map.get(name) or default_value
  if has_next_arg() and _v.ca_extras[0] == name:
    _v.done_handling_args = False
    next_arg()
    if type(effective_default) != bool:
      value = _parse_value_to_match_type(next_arg(), effective_default)
      _v.arg_value_map[name] = value
    else:
      value = True
      _v.arg_value_map[name] = value
  else:
    value = effective_default
  return value


def handling_args():
  """
  Conditional for 'while handling_args()' to repeat
  a loop that handles a sequence of arguments.
  If no explicit arguments were provided, this will return false
  """
  _v.done_handling_args = not _v.done_handling_args
  return _v.done_handling_args


def ca_perform_operations(default_opers=None):
  warning("Deprecated; use execute_commands instead")
  execute_commands(default_opers)


def execute_commands(default_opers=None):
  if not has_next_arg():
    if default_opers:
      _v.ca_extras = default_opers.split()
  try:
    while has_next_arg():
      arg = next_arg()
      oper_info = _v.app_oper_map.get(arg)
      if oper_info is None:
        die("no such operation:", arg)
      _v.done_handling_args = False
      _v.arg_value_map = {}
      oper_info["fn"]()
  except KeyboardInterrupt as exc:
    raise Exception("KeyboardInterrupt") from exc


def _determine_app_oper_map(app):
  m = {}
  help_fns = []

  # Add perform_ functions found in the app
  #
  for func in dir(app):
    func_callable = getattr(app, func)
    if not callable(func_callable):
      continue
    if func.startswith("help_"):
      help_fns.append(func)
      continue
    if not func.startswith("perform_"):
      continue
    oper_key = func[len("perform_"):]
    m[oper_key] = {"fn": func_callable}

  # Add help functions associated with the operations
  #
  for help_fn in help_fns:
    oper_key = help_fn[len("help_"):]
    oper_map = m.get(oper_key)
    if oper_map is None:
      warning("no perform_ function found for", help_fn)
      continue
    oper_map["help"] = getattr(app, help_fn)
  return m


def chomp(string, suffix):
  if string.endswith(suffix):
    return string[:-len(suffix)]
  return string


def txt_read(path, defcontents=None):
  if defcontents and not os.path.isfile(path):
    return defcontents
  with open(path) as f:
    contents = f.read()
  return contents

# Read json from text file, optionally strip comments and extraneous commas
#
def json_read(path, strip_comments=True):
  content = txt_read(path)
  if strip_comments:
    return jstyleson.loads(content)
  return json.loads(content)

def txt_write(path, contents):
  with open(path, "w") as text_file:
    text_file.write(contents)

def mkdir(path, name = None):
  if not os.path.exists(path):
    if name:
      pr("Creating",name,":",path)
    os.makedirs(path)


def clean_directory(directory, extension):
  """
  If directory exists, removes files with particular extension from within it.
  Does not recurse into subdirectories
  """
  if not os.path.exists(directory):
    return
  suffix = "." + extension
  for f in os.listdir(directory):
    if f.endswith(suffix):
      os.remove(os.path.join(directory, f))



def bin_write_floats(path, float_array, append=False):
  import struct
  flags = "wb"
  if append:
    flags = "ab"
  f = open(path, flags)
  f.write(struct.pack("<%df" % len(float_array), *float_array))
  f.close()



def ensure_file_length(path, expected_length, comment="(no comment provided)"):
  file_length = os.stat(path).st_size
  if file_length == expected_length:
    return
  halt("File", path, "length is", file_length, "; expected", expected_length, ";", comment)


def home_dir():
  if _v.home_dir is None:
    _v.home_dir = os.path.abspath(os.environ["HOME"])
  return _v.home_dir


def _extract_name_and_ext(path, must_exist=False):
  prefix, ext = os.path.splitext(path)
  ext = ext[1:]  # Skip the period (if there is one)
  if must_exist and ext == "":
    die("no extension found for:", path)
  return prefix, ext


def get_extension(path, must_exist=False):
  _, ext = _extract_name_and_ext(path, must_exist)
  return ext


def remove_extension(path):
  prefix, ext = _extract_name_and_ext(path)
  return prefix


def set_extension(path, new_ext):
  prefix, ext = _extract_name_and_ext(path)
  if new_ext != "":
    return prefix + "." + new_ext
  return prefix


def clamp(value, min_val, max_val):
  if value < min_val:
    return min_val
  if value > max_val:
    return max_val
  return value

def where(depth=1):
  """
  Construct a string describing the current location in the program
  """
  info = inspect.stack()[depth]
  filename = info[1]
  line_no = info[2]
  line_text = info[3]
  s = "%s (%d)" % (filename, line_no)
  return "%-22s: %s" % (s,line_text)

def pw(depth = 1):
  """
  Display the current location in the program
  """
  print(where(1+depth))


def find_command_filename(directory, prefix):
  """
  Generate a command filename "<prefix>dddddd" and increment the command number
  """
  _v.command_count = (_v.command_count + 1) % 1000000
  path = "%s/%s%09d" % (directory, prefix, _v.command_count)
  if os.path.exists(path):
    die("Command path already exists: ", path)
  return path

def rebuild_dir(path):
  import shutil
  if os.path.isdir(path):
    shutil.rmtree(path)
  os.makedirs(path)
  return path


def remove_if_exists(path, prompt=None):
  if os.path.exists(path):
    if prompt:
      pr("Removing existing file;",prompt," path:",path)
    os.remove(path)
  return path


# Return path_expr if absolute, else relative to root_directory
#
def relative_path(root_directory, path_expr):
  if path_expr[0] != '/':
    return root_directory + "/" + path_expr
  return path_expr


def name_of(var, back_count=1, default_value=None):
  # Get name of variable
  frame = inspect.currentframe()
  for i in range(back_count):
    frame = frame.f_back
  callers_local_vars = frame.f_locals.items()
  names = [var_name for var_name, var_val in callers_local_vars if var_val is var]
  if len(names) != 1:
    if default_value is not None:
      return default_value
    return str(type(var))
  return names[0]



def dict_to_object(name, dict_param, additional_dict=None, default_values_dict=None):
  source_dict = dict_param
  if additional_dict is not None:
    source_dict = {**dict_param, **additional_dict}
  if default_values_dict is not None:
    for key in default_values_dict.keys():
      if key not in source_dict:
        source_dict[key] = default_values_dict[key]
  from collections import namedtuple
  return namedtuple(name, source_dict.keys())(*source_dict.values())


def read_object(abstract_data_class_prototype, path):
  content = json_read(path)
  return abstract_data_class_prototype.parse(content)





class AbstractData:
  """
  Base class for generated types, as well as BaseObject
  """

  def __str__(self):
    """
    Get string representation of object.  Default implementation returns pretty-printed json
    """
    return self.to_string()


  def to_json(self):
    """
    Construct representation of object that is compatible with json.dumps
    """
    return die("to_json is not implemented by:", type(self))


  def to_string(self, pretty=True):
    """
    Get string representation of object.  Default implementation returns pretty-printed json
    """
    _json = self.to_json()
    if pretty:
      return json.dumps(_json, indent=2, sort_keys=True)
    else:
      return json.dumps(_json, sort_keys=True)


  def parse(self, obj):
    return die("parse is not implemented by:", type(self))


  def to_builder(self):
    return die("to_builder is not implemented by:", type(self))


  def build(self):
    return self



class BaseObject(AbstractData):
  """
  A convenient base class that supports selective logging, naming
  """

  def __init__(self, **kwargs):
    self._verbose = kwargs.get("verbose", False)
    self._name = kwargs.get("name", None)

  def set_verbose(self, verbose=True):
    self._verbose = verbose
    return self

  def set_name(self, name):
    check_state(name)
    self._name = name
    return self

  def name(self):
    if self._name is None:
      n = self.provide_name()
      check_state(n, "failed to get name for "+str(self))
      self._name = n
    return self._name

  def provide_name(self):
    n = self.__class__.__name__
    k = n.rfind(".")
    n = n[k + 1:]
    return n

  def log(self, *args):
    if self._verbose:
      pr("("+self.name()+")", *args)

