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



