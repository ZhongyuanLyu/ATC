## Adaptive Transfer Clustering: A Unified Framework

### Authors
- Yuqi Gu
- Zhongyuan Lyu
- Kaizheng Wang

### Paper
- arXiv: https://arxiv.org/pdf/2410.21263

---

## Repository structure

### Simulations/
Code to reproduce simulation figures in the paper.
- `algo_new.py`: implementation of Adaptive Transfer Clustering (ATC) and utilities.
- `Simulation_vs_eps.py`: generates simulation results for the **error vs. discrepancy** settings.
  (Header comments in the file specify the exact figure/panel mapping in the paper.)
- `Simulation_vs_n.py`: generates simulation results for the **error vs. sample size** settings.
  (Header comments in the file specify the exact figure/panel mapping in the paper.)
- `Simulation_plot.py`: plotting script. Header comments specify the corresponding figure numbers/panels.

### Realdata/
Code to reproduce real-data experiments.


#### Lazega Lawyers Network
- `LawyersNetwork_preprocess.py`: preprocessing for the Lazega Lawyers dataset.
- `LawyersNetwork.py`: main script / demo for ATC on Lazega Lawyers.

Reference:
Lazega, E. (2001). *The Collegial Phenomenon: The Social Mechanisms of Cooperation Among Peers in a Corporate Law Partnership*.
Oxford University Press.

#### TIMSS 2019 (Grade 8)
- `TIMSS_preprocess.py`: preprocessing for TIMSS 2019 (SPSS format).

Data download (official site):
- Main page (includes Grade 8 downloads and documentation):
  https://timss2019.org/international-database/

To reproduce our results, we use **Grade 8 (Singapore)**:
1. Go to https://timss2019.org/international-database/
2. Under “Download TIMSS 2019 Data Files by Country”, choose **Grade 8**, tick **Singapore**,
   then click **Download SPSS Data**.
3. Download the item documentation from the same page:
   - “TIMSS 2019 Item Information - Grade 8”
   - (Optional but recommended) “TIMSS 2019 Codebooks - Grade 8” and the User Guide.

Direct documentation/download links (as listed on the TIMSS page):
- TIMSS 2019 User Guide (2nd ed):
  https://timss2019.org/international-database/downloads/TIMSS-2019-User-Guide-for-the-International-Database-2nd-Ed.pdf
- Grade 8 Item Information (zip):
  https://timss2019.org/international-database/downloads/T19_G8_Item%20Information.zip
- Grade 8 Codebooks (zip):
  https://timss2019.org/international-database/downloads/T19_G8_Codebooks.zip

Note: depending on browser/security settings, you may see a warning when visiting external sites.
The links above point to the official TIMSS 2019 International Database.

Reference:
Mullis, I.V.S., Martin, M.O., Foy, P., Kelly, D.L., & Fishbein, B. (2020).
*TIMSS 2019 International Results in Mathematics and Science*. Boston College, TIMSS & PIRLS International Study Center.

#### Business Relation Network (BRN)
- `BRN.json`: business relation network data (renamed from `Data.json` for clarity).

Reference:
Xu, S., Zhen, Y., & Wang, J. (2023). *Covariate-assisted community detection in multi-layer networks*.
*Journal of Business & Economic Statistics*, 41(3), 915–926.

### R scripts (real-data comparisons)
In addition to the Python pipeline, we provide optional R scripts to reproduce the real-data comparison summaries:
- `Laywer.R`: Lazega Lawyers Network real-data comparison (R implementation/metrics).
- `Business.R`: Business Relation Network (BRN) real-data comparison (R implementation/metrics).
- `Useful_functions.R`: helper functions shared by the two R scripts.
