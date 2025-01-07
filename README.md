# Spike Sorting Pipeline using the Locally Competitive Algorithm (LCA)

Our study aims to create a quick and energy-efficient spike sorting pipeline by leveraging the capabilities of LCA for implementation on neuromorphic hardware.

This Python project aims to perform spike sorting based on the Locally Competitive Algorithm (LCA) network.
We used simulated electrophysiological signals generated with the MEArec framework. 
The pipeline consists of two main steps: feature extraction using the LCA and a clustering with HDBSCAN.
To keep things simple, we have omitted the steps of spike detection and template matching, which will be explored in future work.

![alt text](figures/lca_sorting_neuropixels-24.png?)

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Pipeline Overview](#pipeline-overview)
- [Acknowledgements](#acknowledgements)

## Installation

To use this spike sorting pipeline, follow these steps:

1. Clone this repository to your local machine using the following command:
   ```bash
   git clone https://github.com/NECOTIS/LCA-Spike-Sorting
   ```

2. Navigate to the project directory:
   ```bash
   cd LCA-Spike-Sorting
   ```

3. Install the required dependencies (see [Dependencies](#dependencies) section) using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the spike sorting pipeline, execute the main script:
```
python run.py
```
Make sure to provide the required input files and adjust any necessary parameters within the script.
You can run the following command to get info on the parameters :
```
python run.py --help
```

## Dependencies

The spike sorting pipeline requires the following dependencies:

- Python (version >= 3.9)
- Pytorch (version >= 2.0 )
- spikeinterface (version >= 0.97.1)
- HDBSCAN (version >= 0.8.29)

Please refer to the respective documentation and installation guides of each library to ensure proper setup.

## Pipeline Overview

The spike sorting pipeline follows these main steps:

1. Load the simulated electrophysiological signals generated using the MEArec framework.
2. Extract features from the signals using the Locally Competitive Algorithm (LCA).
3. Cluster the sparse encoded the features with the density-based clustering algorithm, HDBSCAN.
4. Construct the inferred spike trains b combining the clustering results and the ground truth spike timings.
5. Compare the ground truth spike train with the inferred spike trains using a comparison metric taken from the spikeinterface framework.

# Acknowledgements

Special thanks to Sean WOOD, Fabien ALIBART and Pierre YGER

