# Exa.TrkX Analysis

Environment and tool set to analysis result of Exa.TrkX pipeline.

## Setup

### Environment

Default environment is python 3.8.9. Other version can be use but might encounter some dependency issue.

### Dependencies

Most of required packages are listed in requirements.txt. Use

```
pip install -r requirements.txt
```

to install those packages. Additionally, you need to install `PyTorch Geometric` manually. The recommanded version is 1.7.2. See [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) from `PyTorch Geometric` documents.

Aside from those packages, there are two submodules need to be install:

- `Exa.TrkX DataIO`:

    This submodule is used to simplify data reading. You can either install with develop mode if you want to make modification:

    ```
    git clone https://github.com/rlf23240/ExaTrkXDataIO.git
    cd ExaTrkXDataIO
    pip install -e .
    ```

    or install it directly:

    ```
    pip install git+https://github.com/rlf23240/ExaTrkXDataIO
    ```

- `Exa.TrkX Plotting`:

    This submodule provide most of plotting functions. You can either install with develop mode if you want to make modification:

    ```
    git clone https://github.com/rlf23240/ExaTrkXPlotting.git
    cd ExaTrkXPlotting
    pip install -e .
    ```

    or install it directly:

    ```
    pip install git+https://github.com/rlf23240/ExaTrkXPlotting
    ```

## Scripts

Analyze scripts are located under subdirectory of HSF:

- hits:

    Plot event overview. Such as hit 2D positions.

- logs:

    Plot train logs. Such as loss function for each iteration.

- performance:

    Plot performance figure such as ROC curve.

- tracks:

    Run track reconstruction from GNN output and evaluate performance.

Those script reference a configuration file in `HSF/configs/reading` to get valid data.
You can view those configuration file and place your data in corresponding location under `data` folder or change it to point to your data. 
Note that base directory can be change in script if you plan to point your data to outside of project.
It is recommended to copy and maintain your own version of config along with your dataset or model as metadata.
It will be easier to orgnize for experiments with multiple training.

## Data

The input data are stage output of [HSF pipeline](https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX). There are training configurations as reference under `HSF/training` using Heavy Neutral Lepton process with HNL mass = 15 GeV, lifetime = 100 mm without pileup. Note that input features are normalized cylindrical coordinates of hits with scale (3000, pi, 400).

## License

Tracking code is adapt from [gnn4itk](https://gitlab.cern.ch/xju/gnn4itk/-/tree/xju/develop/). I include here just for completeness and should not be republish in any form. You should check and give credit to gnn4itk repo.

## See Also

- [gnn4itk](https://gitlab.cern.ch/xju/gnn4itk/-/tree/xju/develop/)

- [Exa.TrkX Data IO](https://github.com/rlf23240/ExaTrkXDataIO)

- [Exa.TrkX Plotting](https://github.com/rlf23240/ExaTrkXPlotting)
