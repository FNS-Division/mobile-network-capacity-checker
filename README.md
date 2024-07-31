
![Logo](https://www.itu.int/web/pp-18/assets/logo/itu_logo.png)

![BSD-3 License](https://img.shields.io/pypi/l/prtg-pyprobe) [![Python](https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white)](https://www.python.org/)

# Mobile Network Capacity Model

The Mobile Network Capacity Model is a geospatial tool designed to assess the adequacy of cellular network connectivity for critical locations such as hospitals, schools, and residential areas (collectively known as points of interest or POIs). This model evaluates whether the available bandwidth from cell towers is sufficient to meet the internet usage demands of these points of interest (POIs).

## Usage example

You may find an [example](notebooks/analysis.ipynb) with test data on how to use this tool in the [notebooks](notebooks/) folder.

The image below is an example of a mobile network coverage and capacity map created by our Mobile Network Capacity Model. It shows the locations of cell towers and their service areas, along with the points of interest that have or don't have sufficient mobile cellular service capacity.

![Logo](https://i.ibb.co/hBbrdLy/cell-towers.png)

## Technical documentation

See our [technical documentation](https://fns-division.github.io/mobile-network-capacity-model-documentation/) for in-depth information on our mobile network capacity models, including instructions on how to use each function, as well as the relationships between each function.

## Repository structure

```sh
mobile-network-capacity-model
├── README.md
├── LICENSE.txt
├── data
│   ├── input_data
│   │   ├── ESP-1697915895-xs2u-pointofinterest.csv
│   │   ├── ESP-1697916284-6wv8-cellsite.csv
│   │   ├── ESP-1719230627-szhl-visibility.csv
│   │   ├── MobileBB_Traffic_per_Subscr_per_Month.csv
│   │   ├── active-mobile-broadband-subscriptions_1711147050645.csv
│   │   ├── area.gpkg
│   │   ├── bwdistance_km.csv
│   │   ├── bwdlachievbr_kbps.csv
│   │   ├── bwrsrp_dbm.csv
│   │   ├── bwulachievbr_kbps.csv
│   │   └── population.tif
│   └── output_data
│       ├── MobileBB_Traffic_per_Subscr_per_Month.csv
│       └── network_capacity.csv
├── documentation
├── environment.yml
├── logs
├── mobile_capacity
│   ├── capacity.py
│   ├── rmalos.py
│   ├── spatial.py
│   └── utils.py
├── notebooks
│   ├── analyses.ipynb
└── tests
    ├── conftest.py
    └── unit
        └── test_class.py
```

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.x
- Conda
- Jupyter Notebook or JupyterLab

You can verify the installations by running these commands:

```bash
python --version
conda --version
jupyter --version
```

If any of these commands fail or show an outdated version:
- Python: Download from [python.org](www.python.org)
- Conda: Follow the official [Conda installation guide](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)
- Jupyter: Install via Conda after setting up your environment.

```bash
conda install jupyter
```

## Installation steps

1. Clone the Repository:
    Clone the repository to your local machine using the following command:

```bash
git clone git@ssh.dev.azure.com:v3/ITUINT/ConnectivityToolkit/mobile-network-capacity-model
```

2. Navigate to the directory:
```bash
cd mobile-network-capacity-model
```

3. Create a virtual environment with the required dependencies via [conda](https://www.anaconda.com/download):
```bash
conda env create --file environment.yml
conda activate mobilecapacityenv
```

## Running your analysis

To conduct your analysis using the Mobile Network Capacity Model, follow these steps:

1. Prepare Your Data: 
   Place your input data files in the `data/input_data` directory. Ensure your data is in the correct format as specified in the technical documentation.

2. Configure Analysis Parameters: 
   Open the `notebooks/analyses.ipynb` notebook. Locate the configuration cell and adjust the parameters according to your specific analysis requirements.

3. Execute the Analysis:
   Run through the notebook cells sequentially. Each cell contains explanations and code for different stages of the analysis.

4. Review Results: 
   After execution, find your output data and visualizations in the `data/output_data` directory. The notebook will also display key results and graphs inline.

5. Iterate if Necessary: 
   Based on your initial results, you may want to adjust parameters or input data. Simply update the relevant sections and re-run the affected cells or the entire notebook.

For a more detailed walkthrough, refer to our [technical documentation](https://fns-division.github.io/mobile-network-capacity-model-documentation/).

## Contributing

Contributions are always welcome.

See our [contributing page](CONTRIBUTING.md) for ways to contribute to the project.

Please adhere to this project's [code of conduct](CODE_OF_CONDUCT.md).

### Keeping analysis outputs private

To ensure that all Jupyter Notebook outputs are cleared before committing changes to the repository, we use `nbstripout`. By following these instructions, contributors to your project will ensure that Jupyter Notebook outputs are cleared before committing changes, helping to keep the repository clean and free of unnecessary data. Follow the steps below to install and enable `nbstripout`.

First, you'll need to install nbstripout. You can do this using `pip`:

```bash
pip install nbstripout
```

Once `nbstripout` is installed, you need to enable it for your Git repository. Run the following command in the root directory of your repository:

```bash
nbstripout --install
```

This will configure nbstripout to automatically strip output from Jupyter Notebooks when you commit them to your repository.

```bash
nbstripout --status
```

## Support
If you need help or have any questions, please contact [fns@itu.int](fns@itu.int).


## License

[BSD-3-Clause](LICENSE.txt)