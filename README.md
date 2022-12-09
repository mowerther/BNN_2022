# About

This repo contains the code to use the Bayesian Neural Networks for Sentinel-3 OLCI and Sentinel-2 MSI as described in:  
[A Bayesian approach for remote sensing of chlorophyll-a and associated retrieval uncertainty in oligotrophic and mesotrophic lakes](https://www.sciencedirect.com/science/article/pii/S0034425722004011).

BNNs were developed for oligotrophic and mesotrophic lakes (**maximum chla 68 $\text{mg m}^{-3}$**) and for **optical water types (OWTs) [2, 3, 4, 5, 9]** described in [Spyrakos et al. (2018)](https://aslopubs.onlinelibrary.wiley.com/doi/full/10.1002/lno.10674). Usage outside of this application scope is not intended. The OWT classification is implemented in this repository and an OWT flag generated for each input observation.

| ![Alt Text](/.repo/figure_14.jpg)| 
|:--:| 
| *Figure 14 from the manuscript: OLCI BNN chla and uncertainty products over southern New Zealand, 9th of February 2020. Chla was measured in situ at the location of Lake Hawea (star symbol) as 0.89 mg^m−3 during the overpass.* |

# Installation

You want to have [Anaconda](https://www.anaconda.com/) installed and use a dedicated Anaconda environment for this code. To install all required packages and versions, follow these steps (recommended):

Clone this repository (e.g., through GitHub Desktop). Alternatively download the repository. <br>
Open your cmd and navigate to the directory where you cloned or downloaded the repository to, for example: <br>
cd C:\github_repos\BNN_2022<br>

Then: <br>
`conda create -n "bnn_2022_env" python=3.8.15`. This creates a fresh conda environment with the correct Python version to load the BNN models.
Activate it: `conda activate bnn_2022_env`. And update it using the bnn_requirements file specific to this repository: `conda env update --file bnn_requirements.yml`<br>

This installes all the repository requirements into your "bnn_2022_env" environment. 
You can then use any Python IDE, such as VSCode, activate/set the conda environment “bnn_2022_env”, and run `example_usage.py` that includes the main script structure.

Optional (not tested): <br>
`pip install git+https://github.com/mowerther/BNN_2022.git` is enabled but does not automatically install the requirements, which you need to then install using the bnn_requirements.yml or .txt file available here.

## Usage

To use this code, you will need a .csv dataset with band configurations as detailed in `sensor_meta_info.py`. You can see an example of the correct file format in `example_data.csv`, which can also be used to run the BNN models and generate outputs using `example_usage.py`.

The input data from a satellite sensor should be the remote sensing reflectance $(\text{sr}^{-1})$ derived through prior atmospheric correction.
The BNNs can also be applied to _in situ_ remote sensing reflectance observations that must correspond to the relative spectral response (RSR) of OLCI or MSI S2A/S2B, and measured as or transformed to above-water reflectance. Negative values < 665 nm will be flagged, and a warning issued. The code raises ValueErrors when observations contain NaN values.

Example code for a satellite image will follow later. Just pick the models, follow the structure of the functions and apply it to your image tiles! 

Happy usage!
Any bugs reported in the issues tab or send me an email directly: mortimer(dot)werther(at)eawag.ch.
