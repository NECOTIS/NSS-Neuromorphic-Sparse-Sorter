# Spike Sorting Pipeline using the Locally Competitive Algorithm (LCA)

Our study aims to create a quick and energy-efficient spike sorting pipeline by leveraging the capabilities of LCA for implementation on neuromorphic hardware.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)

## Installation

To use this spike sorting pipeline, follow these steps:

1. Clone this repository to your local machine using the following command:
   ```bash
   git clone https://github.com/NECOTIS/NSS-Neuromorphic-Sparse-Sorter
   ```

2. Navigate to the project directory:
   ```bash
   cd NSS-Neuromorphic-Sparse-Sorter
   ```

3. Install the required dependencies (see [Dependencies](#dependencies) section) using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the spike sorting pipeline, execute the main script:
```
python scripts/run_nss.py
```

## Dependencies

The spike sorting pipeline requires the following dependencies:

- Python (version >= 3.9)
- Pytorch (version >= 2.0 )
- spikeinterface (version >= 0.97.1)

Please refer to the respective documentation and installation guides of each library to ensure proper setup.

# Acknowledgements

Special thanks to Sean WOOD, Fabien ALIBART and Pierre YGER

