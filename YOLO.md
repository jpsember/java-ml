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

X, Y : logit( locations of box relative to the grid cell, 0...1 )

W, H : ln( box size relative to anchor box size )

J    : 'objectness' :  logit(1),  if there is an object in this slot; otherwise, logit(0)

c_0, c_n : logit(0), except for i = the box's category, when we store logit(1)

***TODO: do we really want to store logits of these values, or leave that up to the train/test code?***

If C = 1, we omit the [ c_0 | .... ] elements, since the category is known (if box exists)

