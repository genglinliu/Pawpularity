# pawpularity
Hybrid CNN on Pawpularity Kaggle Contest

Kaggle contest [link](https://www.kaggle.com/c/petfinder-pawpularity-score) 

Attempt to use the shortfused hybrid conv layer to improve a baseline VGG model with multi-modal inputs

### Notes:
After setting the correct data directory, run `main.py` using a TitanX gpu on gypsum - this works without error.

For the competition, the performance is not very satisfactory, with a public score of `rmse = 22.7` on the baseline. Possible reaons include 
 - the outdated VGG model
 - high memory complexity for more structured covariates to be used
 - training from scratch on a dataset that's not large enough might not be better than transfer learning.

### Problem and Challenges
 - There is some messy code for the hybrid conv layer implementation. Although this script ran without error when executed on gypsum using a titanX-long gpu, it threw errors when in the Kaggle environment. It's related to torch tensors being on different devices but I could not personally locate or resolve the issue. 
 - VGG baseline model is visibly outdated for the learning-based computer vision tasks nowadays, but the challenge is that it's non-trivial to incorporate shortfused layer to more complex CNN architectures such as EffecientNet, etc.
 - Vision transformer models are very dominant across kaggle competitions and our approach is not known to be effective with transformers, at least not yet.