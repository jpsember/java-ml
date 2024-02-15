Specify which variant of the network to use by editing `subproject.txt`.
For example, to use `project-toyimage.json`, set `subproject.txt` to `toyimage`.


Procedurally generate sets of images for training and evaluation:

```
gen_images.sh
```

Delete existing checkpoints (this usually happens automatically, but if code has changed, maybe not):
```
zapcpt.sh
```

Train model:
```
train.sh
```

You should press `^c` to stop streaming the logging output, otherwise subsequent commands will be ignored.

Evaluate the model:
```
inf.sh
scredit inference_results
```
