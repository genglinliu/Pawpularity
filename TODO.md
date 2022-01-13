## TODO list

-----------
12/24/21
-----------
 - train on vgg16 and get a baseline result
 - provide interface for CNN with covariates

-----------
1/8/22
-----------
 - training with ALL the covariates makes the model parameter number way too large. 
 - It's infeasible and we should only try a few
 - If our goal is to see whether covariates help with predication, one way to conduct experiment is to use a lighter network as baseline
 as we really don't need vgg16 particularly. We could use vgg 11 or something even lighter
 - We could also select the most informative covariates using a random forest or something similar.

-----------
1/12/22
-----------
 - ~~alright we boutta take an L on this project lol~~
 - Looked into lighter networks but I decided to stick with vgg 11 which is the lightest vgg for now.
 - I will have a baseline running and a hybrid net with three covariates