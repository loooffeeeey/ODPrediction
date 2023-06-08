# ODPrediction

![Illustration of OD construction](assets/problem_pre.png)

**Problem Definition.** Given the regional urban characteristics of the city ${\lbrace} X_r | r\in\mathcal{R} \rbrace$ and observed OD flows $\lbrace f_{ij}|\langle r_i, r_j\rangle\in\mathcal{X} \rbrace$ between part of OD pairs $\mathcal{X}$ , construct a model to predict the remaining unknown OD flows $\lbrace f_{ij}|\langle r_i,r_j\rangle\notin\mathcal{X}\rbrace$.

## Requirements

- python 3.8
- tqdm == 4.64.0
- pytorch == 2.0.0
- dgl == 0.8
- pandas == 1.4.4
- geopandas == 0.12.2
- matplotlib == 3.5.3
- mlflow == 2.3.2
- networkx == 2.8.6
- pyproj == 3.4.1
- scikit-learn == 1.1.2
- scipy == 1.9.1
- tensorboard == 2.12.3

## Systematic Summary

| Models | Techniques | Required Features | Feature Type |
| ---- | ---- | ---- | ---- |
| gravity | Physical Model | Population, distance | Numerical | 
|IOM | Social Model | Opportunities | Numerical |
| radiation model | Physical Model | Population | Numerical |
| SVR | Kernal-based Model | Socioeconomics<br>distance | Numerical |
| GBRT | Tree-based Model | Socioeconomics | Numerical<br>categorical |
| Random Forest | Tree-based Model | Socioeconomics | Numerical |
| ANN | Neural Network | Socioeconomics | Numerical |
| SI-GCN | Deep Learning | Socioeconomics | Numerical<br>categorical |
| GMEL | Deep Learning | Socioeconomics | Numerical |
| GCN-MLP | Deep Learning | POIs | Numerical |
| spatialGAT | Deep Learning | Population<br>road density<br>POIs<br>railway users | Numerical |
| ConvGCN-RF | Deep Learning | Population<br>landuse | Numerical<br>categorical |
| SIRI | Deep Learning<br>Causal Inference | Socioeconomics<br>POIs | Numerical |


## Performance Comparison

| Models | RMSE | MAE | CPC |
| ---- | ---- | ---- | ---- |
| gravity | 6.944 | 2.179 | 0.602 |
| random forest | 6.273 | 2.436 | 0.638 |
| GBRT | 5.454 | 1.974 | 0.707 |
| XGB | 5.726 | 1.998 | 0.689 |
| ANN | 5.503 | 2.001 | 0.708 |
| GNN | 5.026 | 1.773 | 0.722 |
| GMEL | 4.887 | 1.747 | 0.741 |

## The complete performance table is coming soon.