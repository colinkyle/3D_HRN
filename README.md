### 3D-HRN

##### 3D Histology Reconstruction Networks

######
To get started:
The pre-trained example Tensorflow models are too large to host on github without the large file service.  First install git-lfs by following the directions [here](https://git-lfs.github.com/).


Remember to finalize the git-lfs install with:


```bash

git lfs install
```


then clone the repo using git lfs clone:


```bash

git lfs clone https://github.com/colinkyle/3D-HRN

```



Open the script example_reconstruction.py and modify the paths to reflect your local environment.  The first if block of example_reconstruction.py constructs the histology image stack object and precomputes the distributions for the smoothing regularization term (see below).  The second if block searches for the histological reconstruction.  The final if block saves and plots the results.



This library was developed using Python 3 and Tensorflow 1.0, and requires various open-source python libraries (e.g., numpy, scipy, nibabel for nifti IO, openCV).




##### Method overview:


3D-HRN uses convolutional neural networks to accurately match a stack of serial-section histology to a 3D template (nifti image).  The primary advance of this method is to substitute traditional image registration algorithms with Convolutional Neural Networks followed by Spatial Transformer Networks (Jaderberg et al., 2015).  Once these networks are trained, they perform image registration 1000+ times faster than traditional image registration methods with little to no loss in accuracy.  Here, we use a simplistic histology reconstruction model involving 1. invertable transformations of a template to better match the shape of the histology, 2. registering the histology to the template via rigid and non-rigid transformations. The search scheme involves a reinforcement-learning style retraining of the registration networks which ensures accurate reconstruction despite poor initial matching of histology and the template.



###### Search steps:


1. Sample the Template image at the z-position of each histological section and with warp parameters: z, theta_x, theta_y, dxyz.  Which account for positioning on the microtome during sectioning and shrinking/swelling during fixation.

2. Retrain the registration networks to optimize registration of the histology to the template.

3. Search for the transformation of the Template image (rigid + scaling) which maximizes the objective function. The objective function measures template and stack similarity once the stack has been registered to the template.  It has two regularization terms measuring a. smoothness between neighboring sections in the stack (precomputed in first if block of example_reconstruction.py) and b. deformation energy of non-rigid registration from histology to the template (an output of the Tps non-rigid registration neural network).

4. Go to step 1, sample at best transformation from step 3.



###### Between section smoothness regularization:

In order to efficiently compute the regularization term for the smoothness between neighboring sections, a precomputation step takes place before searching for the optimum template transformation.  This step estimates image similarity between neighboring sections as a function of rigid registration parameters:  Similarity(x,y,theta).  

This step has two stages:

1. Compute empirical between-section image similarity at full grid of x, y, theta positions for matrix: S_empirical(x,y,theta).

2. Assume independence between rigid parameters and estimate distributions S(x), S(y), and S(theta) so that S(x)\*S(y)\*S(theta) approximates S(x,y,theta).

\*note: due to asymetry in the brain's y-direction (inferior-superior), we intentionally underfit S(y) to allow for model flexibility in the y-direction between neighboring sections.



At runtime, we compute the registrations of histology to the template (in search step 3. using the neural networks), we then compute the relative positions between neighboring sections, look up the similarity S for each neighbor pair in the stack, and compute the sum of log[S(x,y,z)] as a regularization term.

###### Training Registration Networks:

You can see example training scripts in train_supervised_RigidNet.py, train_unsupervised_RigidNet.py, and train_unsupervised_TpsNet.py.  Some of the code in this library is hardcoded to deal with 265x257 pixel histology sections.  Please send me a pull request if you go through the trouble of altering the code for different network sizes, or log an issue if you need help.
