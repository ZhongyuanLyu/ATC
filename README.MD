## Adaptive Transfer Clustering: A Uniﬁed Framework

### Authors:
- Yuqi Gu
- Zhongyuan Lyu
- Kaizheng Wang

### Date:
- Oct 29, 2024

### Contents:
- **Simulations**: Contains all necessary code to reproduce the results of the simulation studies. Specifically, `algo_new.py` includes functions to run the Adaptive Transfer Clustering (ATC) algorithm.
- **Realdata**: Contains all necessary code to reproduce the results for the Lazega Lawyers Network, TIMSS 2019, and Business Relation Network.

  - **Lazega Lawyers Network**:
    - `LawyersNetwork_preprocess.py` includes preprocessing functions for the Lazega Lawyers Network dataset.
    - `LawyersNetwork.py` also serves as a demo for ATC.

  - **TIMSS 2019**:
    - `TIMSS_preprocess.py` includes preprocessing functions for the TIMSS 2019 dataset.
    - Data for other countries can be downloaded from the [TIMSS 2019 International Database](https://timss2019.org/international-database/).
  
  - **Business Relation Network**:
    - `Data.json` contains data collected and cleaned by Shirong Xu, Yaoming Zhen, and Junhui Wang. For more details, see Xu, S., Zhen, Y., & Wang, J. (2023). *Covariate-assisted community detection in multi-layer networks*. *Journal of Business & Economic Statistics, 41*(3), 915–926. Taylor & Francis.
