# Insights for working with Neural Networks

## Problems with training progress

If accuracy is not improving or loss is not decreasing, perhaps it is overfitting... try using the training set instead of the test set for the reporting to see if it is at least learning the training set.

## Training set vs test set

Since I'm procedurally generating a large number of training images, and then during training am streaming transformed/augmented versions of these, I don't think it is worth the trouble of supporting a separate testing set and the code that goes with it.  I can instead just look at the training loss (smoothed) as a hopefully reliable indicator of the model's strength.

## Miscellaneous

+ What is the purpose of convolutional layers?  Are they to scale a large image down to an effectively smaller one, and then rely on some fully connected layers to take it from there?

+ What is the purpose of dropout layers?  Is it a sort of loosely proven heuristic that *may* improve the model?  For simplicity, I should consider leaving this feature out, at least until I am confident in my code and the models it produces.

+ Without a GPU to train on, maybe I need to stick to much smaller images, and assume that I can scale them back up later when a GPU becomes available?  Or is there a qualitative shift in the approach that will be necessary, or that won't apply?

## Classification models

+ My experiments have seen cases where the loss function decreases steadily (albeit slowly) while the accuracy is stuck at zero, but then does a big jump to say 50 or more.  I think this is because the continuous floats are slowly moving up to the cutoff point of 0.5 where a classification switches from being 0 to 1, which makes an 'all zeros' become 'approximately half are 1s', which could explain this behaviour.


## Logits and whatnot

Logits, exponentials, and logarithms are often used in models so an unbounded trained value (-inf...+inf) can be bounded to something like -1...+1, or 0...1.  The labels fed in to the training should be the original values, and the appropriate conversions should be performed within the model's loss function.  One exception is at inference time, the Java code can take the model's output and apply the appropriate inverse functions (i.e. to convert -inf...+inf to a probability 0..1).


## Label conversions

The format of training labels (e.g. bounding boxes with categories) can take multiple forms within the workflow:

+ scredit annotations
+ training labels
+ model outputs, before applying the inverse of functions (such as logistic or exp);
+ model outputs, after applying the above inverses to recover values equivalent to the training labels


## PyTorch quirks


### Exploding loss function (issue 56)

Sometimes the loss function suddenly gets crazy large.  I activated 'gradient clipping' and the problem went away.


## Performance tuning

### Batch normalization

Adding batch normalization to the YOLO model reduces training time significantly.  Without normalization, training until loss reached 0.05 took 2h8m32s (epoch 3532).  With normalization, it took 1h24m25s (epoch 2304).

Checkpoint size for with    is 124274579   (not significantly larger)
Checkpoint size for without is 124243793

## Gradient normalization

This fixes some crazy large loss values that I encountered while developing the YOLO algorithm, but it imposes a significant performance cost.  I turned it off and the model trained in only 23 minutes:

```
Target loss reached, stopping training
Elapsed time training: 21m56.2s
Epoch 573
```

But hold on.... I am now continuing to train and the loss seems to go up and doesn't seem to return to the `0.05` region.

```
Saving model inference snapshot
Saving checkpoint: /home/eio/js_dep/ml/example_yolo/checkpoints/000575.pt
Epoch: 574  Loss:  0.035  Loss_class:  0.003  Loss_obj_f:  0.017  Loss_obj_t:  0.004  Loss_wh:  0.057  Loss_xy:  0.001
Epoch: 575  Loss:  0.037  Loss_class:  0.003  Loss_obj_f:  0.017  Loss_obj_t:  0.004  Loss_wh:  0.057  Loss_xy:  0.001
Saving model inference snapshot
Epoch: 576  Loss:  0.036  Loss_class:  0.003  Loss_obj_f:  0.017  Loss_obj_t:  0.004  Loss_wh:  0.058  Loss_xy:  0.001
Epoch: 577  Loss:  0.036  Loss_class:  0.003  Loss_obj_f:  0.017  Loss_obj_t:  0.004  Loss_wh:  0.056  Loss_xy:  0.001
Saving model inference snapshot
Epoch: 578  Loss:  0.036  Loss_class:  0.003  Loss_obj_f:  0.016  Loss_obj_t:  0.004  Loss_wh:  0.055  Loss_xy:  0.001
Epoch: 579  Loss:  0.036  Loss_class:  0.003  Loss_obj_f:  0.016  Loss_obj_t:  0.004  Loss_wh:  0.055  Loss_xy:  0.001
Epoch: 580  Loss:  0.036  Loss_class:  0.003  Loss_obj_f:  0.016  Loss_obj_t:  0.004  Loss_wh:  0.052  Loss_xy:  0.001
Saving model inference snapshot
Epoch: 581  Loss:  0.036  Loss_class:  0.003  Loss_obj_f:  0.016  Loss_obj_t:  0.004  Loss_wh:  0.051  Loss_xy:  0.001
Epoch: 582  Loss:  0.036  Loss_class:  0.003  Loss_obj_f:  0.016  Loss_obj_t:  0.004  Loss_wh:  0.048  Loss_xy:  0.001
Epoch: 583  Loss:  0.036  Loss_class:  0.003  Loss_obj_f:  0.016  Loss_obj_t:  0.004  Loss_wh:  0.051  Loss_xy:  0.001
Epoch: 584  Loss:  0.037  Loss_class:  0.003  Loss_obj_f:  0.016  Loss_obj_t:  0.004  Loss_wh:  0.051  Loss_xy:  0.001
Saving model inference snapshot
Epoch: 585  Loss:  0.038  Loss_class:  0.003  Loss_obj_f:  0.016  Loss_obj_t:  0.004  Loss_wh:  0.049  Loss_xy:  0.001
Epoch: 586  Loss:  0.037  Loss_class:  0.003  Loss_obj_f:  0.016  Loss_obj_t:  0.004  Loss_wh:  0.047  Loss_xy:  0.001
Saving checkpoint: /home/eio/js_dep/ml/example_yolo/checkpoints/000588.pt
Epoch: 587  Loss:  0.037  Loss_class:  0.003  Loss_obj_f:  0.016  Loss_obj_t:  0.004  Loss_wh:  0.045  Loss_xy:  0.001
Epoch: 588  Loss:  0.038  Loss_class:  0.003  Loss_obj_f:  0.016  Loss_obj_t:  0.004  Loss_wh:  0.043  Loss_xy:  0.001
Saving model inference snapshot
Epoch: 589  Loss:  0.044  Loss_class:  0.003  Loss_obj_f:  0.016  Loss_obj_t:  0.004  Loss_wh:  0.040  Loss_xy:  0.001
Epoch: 590  Loss:  0.044  Loss_class:  0.003  Loss_obj_f:  0.015  Loss_obj_t:  0.004  Loss_wh:  0.041  Loss_xy:  0.001
Epoch: 591  Loss:  0.053  Loss_class:  0.003  Loss_obj_f:  0.015  Loss_obj_t:  0.004  Loss_wh:  0.039  Loss_xy:  0.001
Epoch: 592  Loss:  0.078  Loss_class:  0.003  Loss_obj_f:  0.015  Loss_obj_t:  0.004  Loss_wh:  0.040  Loss_xy:  0.001
Epoch: 593  Loss:  0.084  Loss_class:  0.003  Loss_obj_f:  0.015  Loss_obj_t:  0.005  Loss_wh:  0.049  Loss_xy:  0.001
Saving model inference snapshot
Epoch: 594  Loss:  0.081  Loss_class:  0.003  Loss_obj_f:  0.015  Loss_obj_t:  0.005  Loss_wh:  0.051  Loss_xy:  0.001
Epoch: 595  Loss:  0.081  Loss_class:  0.003  Loss_obj_f:  0.015  Loss_obj_t:  0.005  Loss_wh:  0.048  Loss_xy:  0.001
Epoch: 596  Loss:  0.077  Loss_class:  0.003  Loss_obj_f:  0.015  Loss_obj_t:  0.005  Loss_wh:  0.045  Loss_xy:  0.001
Epoch: 597  Loss:  0.073  Loss_class:  0.003  Loss_obj_f:  0.015  Loss_obj_t:  0.005  Loss_wh:  0.045  Loss_xy:  0.001
Epoch: 598  Loss:  0.070  Loss_class:  0.003  Loss_obj_f:  0.015  Loss_obj_t:  0.005  Loss_wh:  0.044  Loss_xy:  0.001
Epoch: 599  Loss:  0.069  Loss_class:  0.003  Loss_obj_f:  0.015  Loss_obj_t:  0.005  Loss_wh:  0.045  Loss_xy:  0.001
Epoch: 600  Loss:  0.066  Loss_class:  0.003  Loss_obj_f:  0.015  Loss_obj_t:  0.005  Loss_wh:  0.044  Loss_xy:  0.001
Saving model inference snapshot
Epoch: 601  Loss:  0.065  Loss_class:  0.003  Loss_obj_f:  0.015  Loss_obj_t:  0.005  Loss_wh:  0.045  Loss_xy:  0.001
Epoch: 602  Loss:  0.069  Loss_class:  0.003  Loss_obj_f:  0.014  Loss_obj_t:  0.006  Loss_wh:  0.047  Loss_xy:  0.001
Epoch: 603  Loss:  0.069  Loss_class:  0.003  Loss_obj_f:  0.014  Loss_obj_t:  0.006  Loss_wh:  0.050  Loss_xy:  0.001
Epoch: 604  Loss:  0.078  Loss_class:  0.003  Loss_obj_f:  0.015  Loss_obj_t:  0.006  Loss_wh:  0.049  Loss_xy:  0.001
Saving checkpoint: /home/eio/js_dep/ml/example_yolo/checkpoints/000606.pt
Epoch: 605  Loss:  0.098  Loss_class:  0.003  Loss_obj_f:  0.015  Loss_obj_t:  0.006  Loss_wh:  0.049  Loss_xy:  0.001
Epoch: 606  Loss:  0.103  Loss_class:  0.003  Loss_obj_f:  0.015  Loss_obj_t:  0.006  Loss_wh:  0.055  Loss_xy:  0.001
Epoch: 607  Loss:  0.104  Loss_class:  0.003  Loss_obj_f:  0.015  Loss_obj_t:  0.006  Loss_wh:  0.053  Loss_xy:  0.001
Epoch: 608  Loss:  0.109  Loss_class:  0.003  Loss_obj_f:  0.015  Loss_obj_t:  0.007  Loss_wh:  0.062  Loss_xy:  0.001
Saving model inference snapshot
Epoch: 609  Loss:  0.118  Loss_class:  0.003  Loss_obj_f:  0.015  Loss_obj_t:  0.007  Loss_wh:  0.068  Loss_xy:  0.001
Epoch: 610  Loss:  0.114  Loss_class:  0.004  Loss_obj_f:  0.016  Loss_obj_t:  0.007  Loss_wh:  0.073  Loss_xy:  0.001
Epoch: 611  Loss:  0.113  Loss_class:  0.004  Loss_obj_f:  0.016  Loss_obj_t:  0.008  Loss_wh:  0.077  Loss_xy:  0.001
Epoch: 612  Loss:  0.106  Loss_class:  0.004  Loss_obj_f:  0.016  Loss_obj_t:  0.008  Loss_wh:  0.078  Loss_xy:  0.001
Epoch: 613  Loss:  0.106  Loss_class:  0.004  Loss_obj_f:  0.016  Loss_obj_t:  0.008  Loss_wh:  0.080  Loss_xy:  0.001
Epoch: 614  Loss:  0.105  Loss_class:  0.004  Loss_obj_f:  0.016  Loss_obj_t:  0.008  Loss_wh:  0.079  Loss_xy:  0.001
Epoch: 615  Loss:  0.102  Loss_class:  0.004  Loss_obj_f:  0.016  Loss_obj_t:  0.008  Loss_wh:  0.074  Loss_xy:  0.001

   :
   :
   :

Epoch: 692  Loss:  0.111  Loss_class:  0.003  Loss_obj_f:  0.013  Loss_obj_t:  0.003  Loss_wh:  0.024  Loss_xy:  0.001
Epoch: 693  Loss:  0.104  Loss_class:  0.003  Loss_obj_f:  0.013  Loss_obj_t:  0.003  Loss_wh:  0.024  Loss_xy:  0.001
Epoch: 694  Loss:  0.098  Loss_class:  0.003  Loss_obj_f:  0.013  Loss_obj_t:  0.003  Loss_wh:  0.024  Loss_xy:  0.001
Epoch: 695  Loss:  0.092  Loss_class:  0.003  Loss_obj_f:  0.013  Loss_obj_t:  0.003  Loss_wh:  0.022  Loss_xy:  0.001
Epoch: 696  Loss:  0.086  Loss_class:  0.003  Loss_obj_f:  0.014  Loss_obj_t:  0.003  Loss_wh:  0.022  Loss_xy:  0.001
Epoch: 697  Loss:  0.080  Loss_class:  0.003  Loss_obj_f:  0.014  Loss_obj_t:  0.003  Loss_wh:  0.022  Loss_xy:  0.002
Epoch: 698  Loss:  0.075  Loss_class:  0.003  Loss_obj_f:  0.014  Loss_obj_t:  0.003  Loss_wh:  0.021  Loss_xy:  0.001
Epoch: 699  Loss:  0.070  Loss_class:  0.003  Loss_obj_f:  0.014  Loss_obj_t:  0.004  Loss_wh:  0.023  Loss_xy:  0.001
Epoch: 700  Loss:  0.066  Loss_class:  0.003  Loss_obj_f:  0.014  Loss_obj_t:  0.004  Loss_wh:  0.024  Loss_xy:  0.001
Saving checkpoint: /home/eio/js_dep/ml/example_yolo/checkpoints/000702.pt
Epoch: 701  Loss:  0.063  Loss_class:  0.003  Loss_obj_f:  0.014  Loss_obj_t:  0.004  Loss_wh:  0.027  Loss_xy:  0.001
Epoch: 702  Loss:  0.060  Loss_class:  0.003  Loss_obj_f:  0.014  Loss_obj_t:  0.004  Loss_wh:  0.029  Loss_xy:  0.001
Epoch: 703  Loss:  0.057  Loss_class:  0.003  Loss_obj_f:  0.014  Loss_obj_t:  0.004  Loss_wh:  0.028  Loss_xy:  0.001
Epoch: 704  Loss:  0.054  Loss_class:  0.003  Loss_obj_f:  0.014  Loss_obj_t:  0.005  Loss_wh:  0.032  Loss_xy:  0.001
Epoch: 705  Loss:  0.051  Loss_class:  0.003  Loss_obj_f:  0.014  Loss_obj_t:  0.005  Loss_wh:  0.031  Loss_xy:  0.001
Epoch: 706  Loss:  0.048  Loss_class:  0.003  Loss_obj_f:  0.014  Loss_obj_t:  0.005  Loss_wh:  0.034  Loss_xy:  0.001
Epoch: 707  Loss:  0.046  Loss_class:  0.003  Loss_obj_f:  0.013  Loss_obj_t:  0.005  Loss_wh:  0.039  Loss_xy:  0.001
Saving model inference snapshot
Epoch: 708  Loss:  0.044  Loss_class:  0.003  Loss_obj_f:  0.013  Loss_obj_t:  0.006  Loss_wh:  0.043  Loss_xy:  0.001
Epoch: 709  Loss:  0.043  Loss_class:  0.003  Loss_obj_f:  0.013  Loss_obj_t:  0.006  Loss_wh:  0.041  Loss_xy:  0.001

```
...or at least it seems to be unstable.

