# LakeIce_RF_Uncertainty

MODIS Terra Level 1B calibrated radiances product (MOD02/MYD02), Collection 6.1 used for mapping lake ice, water and cloud and uncertainty mapping. MOD02QKM with a 250 m pixel spacing at bands 1-2 and MOD02HKM with a 500 m-pixel spacing at bands: 3-7 were used. We applied Trishchenko, Luo, & Khlopenkov (2006)’s  method for resampling to 250 m pixels. Bands used in Random Forest lake ice mapping are shown in the table below. 

<img width="404" alt="image" src="https://user-images.githubusercontent.com/59842707/189716327-e2aa92e6-ff64-469a-8b18-8aafdbe5b6f6.png">

Data flow and processing steps is as follows:

<img width="1017" alt="image" src="https://user-images.githubusercontent.com/59842707/189716129-3d5c1006-79f8-4b9b-a71a-9e8287a9d31a.png">

Note for developers: The original annotation file was big so couldn't push to the repo. A sample exists to show the formating. 

For neighbourhood stats of uncertainties; pixel arrangements diagram is as follows (code for this analysis hasn't been pushed):
<img width="852" alt="image" src="https://user-images.githubusercontent.com/59842707/189719142-79b141e5-e05d-40a5-88a1-1f275647becc.png">

For rejection analysis using uncertainties; the workfolw diagram is as follows (code for this analysis hasn't been pushed):

(annotions for specific days are required for such analysis)

<img width="433" alt="image" src="https://user-images.githubusercontent.com/59842707/189719470-b7f0597e-630d-40b0-b34d-3bc492fffb31.png">

[1] Wu, Y., Duguay, C. R., & Xu, L. (2021). Assessment of machine learning classifiers for global lake ice cover mapping from MODIS TOA reflectance data. Remote Sensing of Environment, 253(November 2020), 112206. https://doi.org/10.1016/j.rse.2020.112206

[2] Shaker, M. H., & Hüllermeier, E. (2020). Aleatoric and Epistemic Uncertainty with Random Forests. Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 12080 LNCS, 444–456. https://doi.org/10.1007/978-3-030-44584-3_35

