## Adaptive Transfer Clustering: A Uniﬁed Framework

### Authors:
- Yuqi Gu
- Zhongyuan Lyu
- Kaizheng Wang

### Paper:
- Available on [arXiv: https://arxiv.org/pdf/2410.21263](https://arxiv.org/pdf/2410.21263)

### Contents:
- **Simulations**: Contains all necessary code to reproduce the results of the simulation studies. Specifically, `algo_new.py` includes functions to run the Adaptive Transfer Clustering (ATC) algorithm.
- **Realdata**: Contains all necessary code to reproduce the results for the Lazega Lawyers Network, TIMSS 2019, and Business Relation Network.

  - **Lazega Lawyers Network**:
    - `LawyersNetwork_preprocess.py` includes preprocessing functions for the Lazega Lawyers Network dataset.
    - `LawyersNetwork.py` also serves as a demo for ATC.
    - For more details on the dataset, see: Lazega, E. (2001). *The Collegial Phenomenon: The Social Mechanisms of Cooperation Among Peers in a Corporate Law Partnership*. Oxford University Press, Oxford.

  - **TIMSS 2019**:
    - `TIMSS_preprocess.py` includes preprocessing functions for the TIMSS 2019 dataset.
    - Data for other countries can be downloaded from the [TIMSS 2019 International Database](https://timss2019.org/international-database/).
    - For more details on the TIMSS 2019 data, see: Mullis, I.V.S., Martin, M.O., Foy, P., Kelly, D.L., & Fishbein, B. (2020). *TIMSS 2019 International Results in Mathematics and Science*. Boston College, TIMSS & PIRLS International Study Center.

  - **Business Relation Network**:
    - `Data.json` contains data collected and cleaned by Shirong Xu, Yaoming Zhen, and Junhui Wang. For more details, see Xu, S., Zhen, Y., & Wang, J. (2023). *Covariate-assisted community detection in multi-layer networks*. *Journal of Business & Economic Statistics, 41*(3), 915–926. Taylor & Francis.
