## 

# Code Repository for "Causal guided multi-task learning for enhancing urban traffic flow probabilistic forecasting with neighborhood-aware disentangling network"

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Last Commit](https://img.shields.io/github/last-commit/WangBo-2000/Causal-guided-multi-task-learning-for-traffic-flow-probabilistic-forecasting.svg)](https://github.com/WangBo-2000/Causal-guided-multi-task-learning-for-traffic-flow-probabilistic-forecasting/commits/main)

This repository contains the implementation code, datasets, and supplementary materials for the paper:

> **Causal guided multi-task learning for enhancing urban traffic flow probabilistic forecasting with neighborhood-aware disentangling network**  
> Wenbo Zhang, Bo Wang, Yikai Fang 
> *AAAI conference*, 2026  

## Overview

    This codebase implements the [method/algorithm/model] proposed in our paper. It includes:
    - Core implementation of the proposed approach
    - Baseline methods for comparison
    - Scripts for data preprocessing and evaluation
    - Experimental results reproduction pipeline

## Baseline Model
    - **SSTBAN**: Self-supervised Spatial-Temporal Bottleneck Attentive Network, which leverages a self-supervised learning module and bottleneck attention to improve generalization, capture long-term dependencies, and reduce computational cost for efficient long-term traffic forecasting.

    - **STGCN**: Spatio-Temporal Graph Convolutional Networks that model traffic forecasting on graphs using convolutional structures instead of traditional convolutional and recurrent units, enabling efficient training and capturing multi-scale spatio-temporal dependencies.

    - **TGC-LSTM**: Traffic Graph Convolutional Long Short-Term Memory network that models traffic interactions based on physical road network topology and integrates graph convolution with LSTM to capture spatial-temporal dependencies effectively.

    - **DGCRN**: Dynamic Graph Convolutional Recurrent Network that employs hyper-networks to dynamically generate graph filter parameters at each time step, capturing fine-grained temporal graph topology while integrating static and dynamic graphs for improved traffic forecasting.

## Project Structure
    repository/
    │  0.research_area.py
    │  1.flow_interpolated.py
    │  Func.py
    │  list.txt
    │  Loop_adjust.py
    │  main.py
    │  
    ├─.idea
    │  │  .gitignore
    │  │  misc.xml
    │  │  Model.iml
    │  │  modules.xml
    │  │  vcs.xml
    │  │  workspace.xml
    │  │  
    │  └─inspectionProfiles
    │          profiles_settings.xml
    │          Project_Default.xml
    │          
    └─data
            flow_interpolated.csv
            gdf.csv
            gdf_processed.csv
            gdf_processed_with_location.csv
            loop.zip
            road_network_segment_level.csv

## Reqirements
    filterpy==1.4.5
    geopandas==0.14.4
    matplotlib==3.10.5
    networkx==3.2.1
    numpy==2.3.2
    pandas==2.3.1
    scikit_learn==1.7.1
    scipy==1.16.1
    Shapely==1.8.5.post1
    torch==2.6.0+cu118
    transbigdata==0.5.3
## Reproduction pipeline
    To execute these files for prediction, follow these three steps:

    1. Run `0.research_area.py` to determine the specific area for traffic prediction, defining the scope for subsequent data processing and model prediction.

    2. Execute `1.flow_interpolated.py` to perform interpolation preprocessing on traffic data `loop.csv` in `loop.zip` and form `flow_interpolated.csv` in the `data` folder. This step fills in data gaps to ensure data integrity and continuity, providing high-quality input for the prediction.

    3. Run `main.py` to conduct specific traffic prediction operations. It leverages the previously determined prediction area and preprocessed traffic data, combined with functional functions and adjustment logic from files like `Func.py`.

## References
    ```bibtex
    @inproceedings{STGCN,
    author = {Lau, Yuen Hoi and Wong, Raymond Chi-Wing},
    title = {Spatio-Temporal Graph Convolutional Networks for Traffic Forecasting: Spatial Layers First or Temporal Layers First?},
    year = {2021},
    isbn = {9781450386647},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3474717.3484207},
    doi = {10.1145/3474717.3484207},
    booktitle = {Proceedings of the 29th International Conference on Advances in Geographic Information Systems},
    pages = {427–430},
    numpages = {4},
    keywords = {traffic forecasting, temporal dependencies, spatio-temporal graph convolutional networks, spatial dependencies, sequence of modeling},
    location = {Beijing, China},
    series = {SIGSPATIAL '21}
    }

    @ARTICLE{TGC-LSTM,
      author={Cui, Zhiyong and Henrickson, Kristian and Ke, Ruimin and Wang, Yinhai},
      journal={IEEE Transactions on Intelligent Transportation Systems}, 
      title={Traffic Graph Convolutional Recurrent Neural Network: A Deep Learning Framework for Network-Scale Traffic Learning and Forecasting}, 
      year={2020},
      volume={21},
      number={11},
      pages={4883-4894},
      keywords={Convolution;Forecasting;Predictive models;Roads;Machine learning;Feature extraction;Artificial neural networks;Traffic forecasting;spatial–temporal;graph convolution;LSTM;recurrent neural network},
      doi={10.1109/TITS.2019.2950416}}

    @article{DGCRN,
    author = {Li, Fuxian and Feng, Jie and Yan, Huan and Jin, Guangyin and Yang, Fan and Sun, Funing and Jin, Depeng and Li, Yong},
    title = {Dynamic Graph Convolutional Recurrent Network for Traffic Prediction: Benchmark and Solution},
    year = {2023},
    issue_date = {January 2023},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {17},
    number = {1},
    issn = {1556-4681},
    url = {https://doi.org/10.1145/3532611},
    doi = {10.1145/3532611},
    journal = {ACM Trans. Knowl. Discov. Data},
    month = feb,
    articleno = {9},
    numpages = {21},
    keywords = {traffic benchmark, dynamic graph construction, Traffic prediction}
    }

    @misc{STFGNN,
          title={Spatial-Temporal Fusion Graph Neural Networks for Traffic Flow Forecasting}, 
          author={Mengzhang Li and Zhanxing Zhu},
          year={2021},
          eprint={2012.09641},
          archivePrefix={arXiv},
          primaryClass={cs.LG},
          url={https://arxiv.org/abs/2012.09641}, 
    }

    @article{ASTGCN,
    author = {Guo, Shengnan and Lin, Youfang and Feng, Ning and Song, Chao and Wan, Huaiyu},
    year = {2019},
    month = {07},
    pages = {922-929},
    title = {Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting},
    volume = {33},
    journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
    doi = {10.1609/aaai.v33i01.3301922}
    }
    ```

```markdown
