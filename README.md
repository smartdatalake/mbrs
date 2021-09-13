## Mixture-based best region search

#### Overview

Given a collection of geospatial points of different types, mixture-based best region search aims at discovering spatial regions exhibiting either very high or very low mixture with respect to the types of enclosed points. Existing works for this problem are limited in the shape of regions they can discover, typically supporting only fixed-shape regions, such as circles or rectangles. However, this is in contrast to interesting regions occurring in real-world data, which often have arbitrary shapes. In this paper, we formulate the problem of mixture-based best region search for arbitrarily shaped regions. Our formulation introduces certain desired properties to ensure the cohesiveness and completeness of detected regions. We then observe that computing exact solutions to this problem has exponential cost with respect to the number of points, making it infeasible in practice. To overcome this issue, we propose anytime algorithms that efficiently search the space of candidate solutions to produce high-scoring regions under any given time budget. Our experiments on several real-world datasets show that our algorithms can produce high-quality results even within time constraints that are suitable for near-interactive data exploration.

Full details are available in this [paper](https://doi.org/10.1145/3474717.3484215): D. Skoutas, D. Sacharidis, and K. Patroumpas. Discovering Mixture-Based Best Regions of Arbitrary Shapes. In Proceedings of the 29th International Conference on Advances in Geographic Information Systems ([ACM SIGSPATIAL 2021](https://sigspatial2021.sigspatial.org/)), Beijing, China, November 2021.

#### Source code

The source code is written in Python. A Jupyter notebook indicates how to: 

- ingest a point dataset from a CSV file (containing Points of Interest in New York City extracted from [OpenStreetMap](https://www.openstreetmap.org/));
- specify values for the various parameters,
- run the methods; and 
- automatically create plots that compare performance among all proposed methods.

#### Acknowledgment

This work was supported by the EU H2020 project [SmartDataLake](http://smartdatalake.eu/) (825041).

#### License

The contents of this project are licensed under the [Apache License 2.0](https://github.com/SLIPO-EU/loci/blob/master/LICENSE).
