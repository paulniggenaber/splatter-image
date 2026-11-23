
Quick update on the baseline reproduction before we start working on our extension.

### 1. Got the original model running

I set up the Splatter Image repository on Bocconi HPC, downloaded the ShapeNet-SRN Cars dataset, and got the pretrained model working.

There were a few issues with dependencies, dataset paths, and the Gaussian rasterizer, so I had to make some small changes to the code.

### 2. Reproduced the paper's results!

I ran evaluation on all **704 cars** in the test set.

| Metric | Paper | Our result |
|---|---|---|
| PSNR ↑ | 24.00 | 23.9991 |
| SSIM ↑ | 0.92 | 0.9228 |
| LPIPS ↓ | 0.078 | 0.0770 |

So basically, we managed to reproduce the original single-view Cars results almost exactly 

The evaluation initially ran out of RAM, so I split it into two runs and combined the scores afterward.
Full evaluation scores for all 704 test objects are available in baseline_cars_full_scores.txt.
### 3. Generated visualizations

I also saved the predicted images and ground-truth images for five cars, with 251 views per car.

From these, I made rotating-car videos and a side-by-side comparison of the reconstruction and ground truth.

I also exported one car as a 3D Gaussian Splat PLY file (16,384 Gaussians). The interactive viewer isn't displaying it correctly yet, so that part still needs fixing.

### 4. What's next?

Now that the baseline works, we can start implementing our extension that we still have to decide on, my suggestion is:

**Multi-View Importance-Aware Gaussian Pruning** thats basically what we've discussed amount reducing amount of gaussians. 

The idea is to remove less useful Gaussians while keeping reconstruction quality as high as possible.

We can start by comparing random pruning, opacity-based pruning, and an importance-based method, then measure PSNR, SSIM, LPIPS, rendering speed, and memory usage.

The baseline is ready though.                                        
