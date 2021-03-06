## Mixture-based best region search

#### Overview

Given a collection of geospatial point entities of different types, _mixture-based best region search_ aims at discovering spatial regions exhibiting either very high or very low mixture with respect to the types of enclosed points. Existing works for this problem are limited in the shape of regions they can discover, typically supporting only fixed-shape regions, such as circles or rectangles. However, this is in contrast to interesting regions occurring in real-world data, which often have arbitrary shapes. 

This software supports mixture-based best region search for _arbitrarily shaped regions_ directly from the underlying entities, not only based on their spatial locations, but also on their thematic classification. Our formulation introduces certain desired properties to ensure the cohesiveness and completeness of the detected regions. However, computing exact solutions to this problem has exponential cost with respect to the number of points, making it infeasible in practice. To overcome this issue, we propose anytime algorithms that efficiently search the space of candidate solutions to produce high-scoring regions under any user-specified time budget. These methods can discover _top-k regions_ where there is a concentration of entities either mostly from the same category (a _low-mixture_ region) or from diverse categories (a _high-mixture_ region). Regions are ranked with appropriate scores taking into consideration the size of the region and the diversity of entities therein. Experiments on several real-world datasets show that our algorithms can produce high-quality results even within time constraints that are suitable for near-interactive data exploration.

This software can be used for discovery of top-k mixture-based regions over a collection of geolocated entities characterized by various categories, like Points of Interest (shops, restaurants, museums, etc.), or locations of various crime incidents, or check-ins at different types of venues, etc. We also provide a graphical interface that enables users to specify their preferences and provides map-based visualizations that highlight latent information in the data.

Full details about the problem setting and the proposed algorithms are available in this paper: D. Skoutas, D. Sacharidis, and K. Patroumpas. [Discovering Mixture-Based Best Regions of Arbitrary Shapes](https://doi.org/10.1145/3474717.3484215). In Proceedings of the 29th International Conference on Advances in Geographic Information Systems ([ACM SIGSPATIAL 2021](https://sigspatial2021.sigspatial.org/)), Beijing, China, November 2021.

More details about MAGE and its user interface are available in thie demo paper: K. Patroumpas, D. Skoutas, and D. Sacharidis. [MAGE: Discovering Mixture-based Areas of Interest over Geolocated Entities](https://openproceedings.org/2022/conf/edbt/paper-147.pdf). In Proceedings of the 25th International Conference on Extending Database Technology ([EDBT/ICDT 2022 Joint Conference](https://conferences.inf.ed.ac.uk/edbticdt2022/)), Edinburgh, Scotland, UK, March 2022.


#### Source code

The source code is written in Python. 

A Jupyter [notebook](notebooks/MixtureBestRegionSearch-NYC.ipynb) replicates the process utilized for conducting the experiments for the published [paper](https://doi.org/10.1145/3474717.3484215) and indicates how to: 

- ingest a point dataset from a CSV file (containing Points of Interest in New York City extracted from [OpenStreetMap](https://www.openstreetmap.org/));
- specify values for the various parameters,
- run the methods; and 
- automatically create plots that compare performance among all proposed methods.

#### Graphical interface

We demonstrate the entire framework with [MAGE](notebooks/MAGE-Demo-App.ipynb), a graphical user interface that exposes the full functionality and allows users to easily specify their preferences and inspect the returned top-k regions on map. Also built in Python and powered by [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) for Jupyter notebooks, this interactive HTML page can be deployed using [Voila](https://voila.readthedocs.io/en/stable/using.html). This GUI is organized in three panels and offers the ability to load a dataset with geolocated entities, preprocess it, and discover mixture-based regions with user-specified parametrization.

Please watch this short [video](https://drive.google.com/file/d/1-BwoBE1zrDvktTJIn2lcInTdDIZIO_WN/view) (5 minutes) that offers a quick tour of MAGE for discovering mixture-based areas of interest over Points of Interest (shops, restaurants, museums, etc.) in Athens [extracted in CSV format](http://download.slipo.eu/results/osm-to-csv/) from [OpenStreetMap](https://www.openstreetmap.org/).

#### Presentations

You may watch this [video presentation](https://www.youtube.com/watch?v=8EURgp9nJqc) (duration: 12 minutes) at SIGSPATIAL Conference 2021 regarding the proposed methodology and its evaluation over real-world datasets. 

You may watch this [demonstration](https://www.youtube.com/watch?v=rjoCjMnvZ-4) (duration: 8 minutes) at EDBT Conference 2022 regarding MAGE and its application over different datasets.

#### Acknowledgment

This work was supported by the EU H2020 project [SmartDataLake](http://smartdatalake.eu/) (825041).

#### License

The contents of this project are licensed under the [Apache License 2.0](https://github.com/SLIPO-EU/loci/blob/master/LICENSE).
