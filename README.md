# Case Study: A spatially-complete $\delta ^{18}O$ dataset using Machine Learning methods

The Arctic Ocean is one of the regions most strongly affected by climate change; it may be ice free in summer by the middle of the century. As the region warms, erosion of surrounding coasts is accelerating, changing the composition of material released into the ocean from rivers and glaciers and impacting the fragile Arctic ecosystems. Understanding how freshwater and riverborne nutrients enter the Arctic Ocean and where they go is vital to predict and plan for the consequences. Nature has provided us with a tool to do this. All water ($H_2O$) is composed of hydrogen ($H$) and oxygen ($O$). However, there is a variety (isotope) of oxygen (referred to as $^{18}O$) that is more common in seawater and sea ice (frozen seawater) than in rivers and glaciers. By mapping the relative abundance of $^{18}O$ (referred to as $\delta^{18}O$), we can see how freshwater, and the nutrients it may carry, move around and out of the Arctic.

We present a case study where a simple machine learning model is applied to in-situ oxygen isotope tracer data, developed by the British Antarctic Survey as part of the AISIT project, to produce a spatially-complete dataset in the Arctic region.

To install, run `git clone https://github.com/NOC-MDP/AISIT.git`

The key files in the repository are
* `case_study.ipynb`: A Jupyter notebook containing the details of the ML model, its application to the BAS dataset and subsequent use to build a spatially-complete dataset of $\delta^{18}O$ using the Arctic Ocean Reanalysis. 
* `case_study.md`: A report summarising the details contained in `case_study.ipynb`
* `d18O_science.ipynb`: A Jupyter notebook that performs some simple scientific applications of the $\delta^{18}O$ dataset developed in the `case_study.ipynb`.

Other files are supplementary and allow these files to run correctly. An `environment.yml` file is included, which outlines which Python packages are required. These can all be installed via `pip`.

The BAS data is included in the repository; however, the AOR data is not included in the repository due to space restrictions. This can be downloaded from https://data.marine.copernicus.eu/product/ARCTIC_MULTIYEAR_PHY_002_003/files?subdataset=cmems_mod_arc_phy_my_topaz4_P1M_202506, or the users can choose an alternative model to use.