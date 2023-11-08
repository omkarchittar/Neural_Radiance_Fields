NeRF
========================
**Name: Omkar Chittar**  
------------------------
```
Neural_Radiance_Fields
+-configs
+-data
+-images
+-outputs
+-README.md
+-report
+-ta_images
__init__.py
checkpoints.pth
data_utils.py
dataset.py
environment.yml
implicit.py
main.py
ray_utils.py
render_functions.py
renderer.py
sampler.py

```

# **Installation**

- Download and extract the files.
- Make sure you meet all the requirements given on: https://github.com/848f-3DVision/assignment2/tree/main
- Or reinstall the necessary stuff using 'environment.yml':
```bash
conda env create -f environment.yml
conda activate l3d
```
- The **data** folder consists of all the data necessary for the code.
- Uncompress the file `lego.png.zip` in the **data** folder
- The **images** folder has all the images/gifs generated after running the codes.
- All the necessary instructions for running the code are given in **README.md**.
- The folder **report** has the html file that leads to the webpage.


# **1. Differentiable Volume Rendering**
## **1.1. Familiarize yourself with the code structure**
## **1.2. Outline of tasks**
## **1.3. Ray sampling**
After making changes to:
1. `get_pixels_from_image` in `ray_utils.py` and
2. `get_rays_from_pixels` in `ray_utils.py`

Run the code:  
```bash
python main.py --config-name=box
```
The code renders the xy_grid and the ray bundle is saved as *xygrid.png* and *raybundle.png* respectively in the images folder. 

## **1.4. Point sampling**
After making changes in `StratifiedSampler` in `sampler.py`

Run the code:  
```bash
python main.py --config-name=box
```
The code renders the sample points and is saved as *sample_points.png* in the images folder. 

## **1.5. Volume rendering**
After making the necessary changes in `VolumeRenderer._compute_weights`, `VolumeRenderer._aggregate` and `VolumeRenderer.forward`,

Run the code:  
```bash
python main.py --config-name=box
```
The code renders the box volume defined in the `configs/box.yaml` and the renderings are saved as *part_1.gif* and the depth is saved as *depth.png* in the images folder. 

# **2. Optimizing a basic implicit volume**
## **2.1. Random ray sampling**
Implement the `get_random_pixels_from_image` method in `ray_utils.py`

## **2.2. Loss and training**
Change the 'loss' initially set to 'None' to 'MSELoss'
The 'MSELoss' calculates the mean squared error between the predicted colors and ground truth colors `rgb_gt`.

## **2.3. Visualization**
Train the model by running the code:
```bash
python main.py --config-name=train_box
```
The code renders a spiral sequence of the optimized volume in `images/part_2.gif`

# **3. Optimizing a Neural Radiance Field (NeRF)**
To train a NeRF on the lego bulldozer dataset, 
Run the code:

```bash
python main.py --config-name=nerf_lego
```

This creates a NeRF with the `NeuralRadianceField` class in `implicit.py`, and uses it as the `implicit_fn` in `VolumeRenderer`. 

The NeRF is trained for 250 epochs on 128x128 images.
In order to make changes in the model parameters, image_size etc., visit the `configs/nerf_lego.yaml` file.

# **4. Webpage**
The html code for the webpage is stored in the *report* folder along with the images/gifs.
Clicking on the *webpage.md.html* file will take you directly to the webpage.




