# Will the Cloud Run Dry?

## Mapping the Global Water Risks of Data Centers in a Warming World

## Table of Contents

- [Mapping the Global Water Risks of Data Centers in a Warming World](#mapping-the-global-water-risks-of-data-centers-in-a-warming-world)
- [Overview](#overview)
  - [Problem Statement](#problem-statement)
  - [Project structure](#project-structure)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the notebooks](#running-the-notebooks)
  - [Data sources](#data-sources)
- [License](#license)
- [Contact](#contact)

## Overview

This repository contains the code for the MSc Industrial Ecology thesis project by Audrey Wientjes. The project analyzes the water use of data centers using spatial GIS modeling and environmental footprinting methodologies, and assesses their impact on local water scarcity.

See the [pyproject.toml](pyproject.toml) file for complete package information such as project metadata and dependencies.

The input data for this project is available on Zenodo: [10.5281/zenodo.14914472](https://zenodo.org/records/149144720)


### Problem Statement

Data centers require significant water resources for cooling systems and power generation, yet the global scale and distribution is still unknown. With water scarcity risks growing as global warming rises, data centers are increasingly vulnerable, while contributing to local water stress. This project analyzes their vulnerability and contributions to local water scarcity through:

- Web scraping of facility data
- Estimating water and energy use metrics
- Modeling water scarcity under climate change scenarios
- Assessing the impact of data centers on water scarcity

### Project structure


```bash
├── data
│   ├── inputs                              # See Zenodo link for input data
│   └── outputs                             # See Zenodo link for output data
├── functions
│   ├── data_etl                            # Module for data cleaning and transformation functions
│   ├── energy_and_water_use                # Module for energy and water use analysis functions
│   ├── figures                             # Module for creating figures functions
│   ├── water_scarcity                      # Module for water scarcity assessment functions
│   └── project_settings.py                 # Project-wide variables
├── 0_scrape_datacenters_com.ipynb          # Web scraping notebook
├── 1_data_etl.ipynb                        # Data cleaning and transformation notebook
├── 2_energy_and_water_use_analysis.ipynb   # Energy and water use analysis notebook
├── 3_water_scarcity_assessment.ipynb       # Water scarcity assessment notebook
├── 4_figures.ipynb                         # Figures notebook
├── LICENSE                                 # License file
├── README.md                               # This file
└── pyproject.toml                          # Project metadata and dependencies
```

## Setup

### Prerequisites

- The easiest way to install the project dependencies is via [uv](https://docs.astral.sh/uv/). You can [install](https://docs.astral.sh/uv/getting-started/installation/) it using your favorite package manager.

### Installation

- Clone the repository:

```bash
git clone https://github.com/amwientjes/<data-center-water-footprint>
cd <data-center-water-footprint>
```

- Install project dependencies in a virtual environment using uv: `uv sync`

### Running the notebooks

Choose one of the following options to run the Jupyter notebooks:

- Command line: `uv run jupyter lab` to start a Jupyter Lab server.
- [VS Code](https://code.visualstudio.com/): Open notebook and select kernel from virtual environment.

### Data sources

The input data for this project is available on Zenodo: [10.5281/zenodo.14914472](https://zenodo.org/records/149144720)

## License

The code in this repository is licensed under the GNU AGPLv3 License. See the [LICENSE](LICENSE) file for details.

## Contact

For further inquiries, pleast contact the author through LinkedIn: [Audrey Wientjes](https://www.linkedin.com/in/audrey-wientjes/)
