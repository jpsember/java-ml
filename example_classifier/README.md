Procedurally generate a set of images:

```
ml genimages
```

Generate a set of images for evaluating (performing inference after training):

```
comp_inf.sh
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
