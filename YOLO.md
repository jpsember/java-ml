# Documentation about my YOLO implementation



##  Image labels format

This is the format of the labels for an image.

+ Gx, Gy = dimensions of the grid

+ B = number of anchor boxes

+ C = number of categories (or classes)

For each grid cell (Gy rows of Gx cells, in standard bitmap order), there are B contiguous entries


Let gx, gy be indices of grid cell, and bi the index of the anchor box.

Then at index

(((gy * Gx) + gx) * B) + bi

we store

[ X | Y | W | H | J | c_0 | c_1 | ... | c_C-1 ]

X, Y :  locations of center of box relative to the grid cell, 0...1
W, H :  box size relative to anchor box size

J    : 'objectness' :  1,  if there is an object in this slot; otherwise, 0

c_0, c_n : 0, except for i = the box's category, when we store 1




## Investigate labelling boxes on boundaries as well, so model doesn't get penalized for edge cases

The suppression filtering will remove the duplicates that can result from it.
