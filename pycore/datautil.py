from pycore.base import *

class DataUtil:

  @classmethod
  def parse_list_of_objects(cls, default_instance, source_list_or_none, none_if_source_none):

    if source_list_or_none is None:
      if none_if_source_none:
        return None
      return []

    items = []
    for obj in source_list_or_none:
      result = default_instance.parse(obj)
      items.append(result)
    return items
  