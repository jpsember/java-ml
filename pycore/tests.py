from pycore.base import *


clear_screen("just called clear_screen")
pr(home_dir())
pr("\n\n\n========== base.py: basic python tools =========")

pr("Here's a stack trace:", st())
warning("This is a warning;", 5, 12, 42.123456, "alpha", "bravo", dtype(__file__))
todo("this code is unimplemented")
if True:
  print(timer_start("Beginning a time consuming operation"))
  time.sleep(0.5)
  print(timer_mark("Slept for 1/2 sec"))
  time.sleep(0.2)
  print(timer_mark("Slept for a bit more"))
pr("Current dir:", current_dir())

x = "this is a string"
pr("name_of:",name_of(x))


