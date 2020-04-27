

[![Abcdspec-compliant](https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg)](https://github.com/brain-life/abcd-spec)
[![Run on Brainlife.io](https://img.shields.io/badge/Brainlife-bl.app.1-blue.svg)](https://doi.org/10.25663/brainlife.app.306)

# Network Measurements
This app generates simple visualizations for networks by using a force-directed algorithm. The current implementation uses the Large Graph Layout (LGL) algorithm.

### Authors
- [Filipi N. Silva](filsilva@iu.edu)

<!-- ### Contributors
- Franco Pestilli (franpest@indiana.edu) -->

<!-- ### Funding  -->
<!-- [![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1734853) -->


### Citations

1. Adai, Alex T., Shailesh V. Date, Shannon Wieland, and Edward M. Marcotte. "LGL: creating a map of protein function with an algorithm for visualizing very large biological networks." Journal of molecular biology 340, no. 1 (2004): 179-190. [https://doi.org/10.1016/j.jmb.2004.04.047](https://doi.org/10.1016/j.jmb.2004.04.047)

## Running the App 

### On Brainlife.io

You can submit this App online at [https://doi.org/10.25663/brainlife.app.306](https://doi.org/10.25663/brainlife.app.306) via the "Execute" tab.

### Running Locally (on your machine)
Singularity is required to run the package locally.

1. git clone this repo.

```bash
git clone <repository URL>
cd <repository PATH>
```

2. Inside the cloned directory, edit `config-sample.json` with your data or use the provided data.

3. Rename `config-sample.json` to `config.json` .

```bash
mv config-sample.json config.json
```

4. Launch the App by executing `main`

```bash
./main
```

### Sample Datasets

A sample dataset is provided in folder `data` and `config-sample.json`

## Output

The output is a folder containing the pdfs of the static visualizations for each network in input.

#### Product.json

The `product.json` contains previews of the generated figures.

### Dependencies

This App only requires [singularity](https://www.sylabs.io/singularity/) to run. If you don't have singularity, you will need to install the python packages defined in `environment.yml`, then you can run the code directly from python using:  

```bash
./main.py config.json
```
