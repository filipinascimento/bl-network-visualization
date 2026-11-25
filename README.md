

[![Abcdspec-compliant](https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg)](https://github.com/brain-life/abcd-spec)
[![Run on Brainlife.io](https://img.shields.io/badge/Brainlife-bl.app.1-blue.svg)](https://doi.org/10.25663/brainlife.app.306)

# Network Visualization

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Draws a static 2D visualisation of a network with the Large Graph Layout (LGL) force-directed algorithm in igraph, colouring and sizing nodes by a chosen property, and saves it as a PDF.**

Loads a network in JGF format, lays it out with LGL and plots it with node colour and size taken from a node property (degree by default, or communities detected by another App), optionally with labels. Use it for a quick visual check of connectivity matrices or derived networks; edge weights must be non-negative.

| | |
|---|---|
| 📥 **Inputs** | `generic/network` |
| 📤 **Outputs** | `report/pdf` |
| 🐳 **Container** | `filsilva/cxnetwork:0.2.0` |
| ⚙️ **Resources** | 1 CPUs |
| 🔗 **Run it** | [doi.org/10.25663/brainlife.app.306](https://doi.org/10.25663/brainlife.app.306) |

#### 📥 Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `network` | `generic/network` | yes | Input network. Currently, visualization only works if the edge weights are non negative. Please use the preprocess app to prepare the network. |

#### 📤 Outputs

| Output | Datatype | Description |
|---|---|---|
| `output` | `report/pdf` | Static visualization of the network in pdf. |

<details>
<summary><b>⚙️ Parameters</b> (3)</summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `color-property` | string | `degree` | Uses the provided property as color. If the network has communities detected by the "Network Communities" App, you can provide "communitiy" to plot communities as colors. |
| `size-property` | string | `degree` | Makes the size of nodes proportional to the provided property. |
| `plot-labels` | boolean | `true` | Plot labels on top of nodes. |

</details>

📚 **References used by this App:**
- Avesani P et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. Sci Data 6:69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)
- Adai AT, Date SV, Wieland S, Marcotte EM. LGL: creating a map of protein function with an algorithm for visualizing very large biological networks. J Mol Biol 340(1):179-190 (2004). [doi:10.1016/j.jmb.2004.04.047](https://doi.org/10.1016/j.jmb.2004.04.047)

<!-- brainlife-app:end -->

This app generates simple visualizations for networks by using a force-directed algorithm. The current implementation uses the Large Graph Layout (LGL) algorithm.

### Authors
- [Filipi N. Silva](https://filipinascimento.github.io)

### Contributors
- [Franco Pestilli](https://liberalarts.utexas.edu/psychology/faculty/fp4834)


### Funding Acknowledgement
brainlife.io is publicly funded and for the sustainability of the project it is helpful to Acknowledge the use of the platform. We kindly ask that you acknowledge the funding below in your publications and code reusing this code.

[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-1636893](https://img.shields.io/badge/NSF_BCS-1636893-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-ACI-1916518](https://img.shields.io/badge/NSF_ACI-1916518-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1916518)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://grantome.com/grant/NIH/R01-EB029272-01)


### Citations
1. Avesani, P., McPherson, B., Hayashi, S. et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. Sci Data 6, 69 (2019). [https://doi.org/10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

2. Adai, Alex T., Shailesh V. Date, Shannon Wieland, and Edward M. Marcotte. "LGL: creating a map of protein function with an algorithm for visualizing very large biological networks." Journal of molecular biology 340, no. 1 (2004): 179-190. [https://doi.org/10.1016/j.jmb.2004.04.047](https://doi.org/10.1016/j.jmb.2004.04.047)

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

The output folder contains the pdfs of the static visualizations for each network in input.

#### Product.json

The `product.json` contains previews of the generated figures.

### Dependencies

This App only requires [singularity](https://www.sylabs.io/singularity/) to run. If you don't have singularity, you will need to install the python packages defined in `environment.yml`, then you can run the code directly from python using:  

```bash
./main.py config.json
```
