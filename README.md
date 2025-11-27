# OpenTTMD - evaluating ligand pose stability using thermal titration

#### v1.0.1
OpenTTMD - an open source implementation of Thermal Titration Molecular Dynamics (TTMD) with OpenMM. Replicates the protocol as described by
Pavan et al. 2022 (DOI: 10.1021/acs.jcim.2c00995).

Runs sequential MD with increasing temperatures (default, 300 to 450) and scores the ability of the ligand to maintain its native interactions with the target.

Scoring function (interaction fingerprint from ODDT)

$$
IFP_{cs} = \frac{A \cdot B}{\lVert A \rVert \lVert B \rVert} \times - 1
$$

MS coefficient (slope of the straight line that interpolates the first and last points of the “titration profile”):

$$
MS = \frac{\text{mean } IFP_{cs}^{\,T_{\text{end}}} - (-1)}{T_{\text{end}} - T_{\text{start}}}
$$

The lower the MS is (between 0 and 1), the stronger is the binding
	
Outputs:
- titration_profile.png (with MS value)
- titration_timeline.png (IFPcs and RMSD evolution over time)


Version 1.0.1
- instead of N reps, it sequentially performs MD runs within the specified temperature ramp (300-450K, $\Delta T$   = 10 K)
- uses fingerprint scoring (IFPcs) instead of ContactScore
- outputs "titration_profile.png" and "titration_timeline.png" instead of "results.csv"

### Background 
OpenTTMD is an open source implementation of thermal titration molecular dynamics (TTMD). It was heavily inspired by the OpenBPMD scripting with OpenMM while based scientifically in the TTMD package:
- OpenBPMD: https://github.com/dlukauskis/OpenBPMD
- TTMD: https://github.com/molecularmodelingsection/TTMD

### Installation & Usage

The dependencies needed for running the scripts can be installed with conda:

```
conda create -n openttmd
conda activate openttmd

conda install -c conda-forge openmm=8.1.2
conda install -c conda-forge mdanalysis
conda install -c conda-forge mdtraj
conda install -c conda-forge parmed
```

Once the dependencies have been installed, running OpenBPMD involves simply running one of the Python scripts. Have a look at the ```examples/``` directory for further instructions on how to run and analyse the OpenTTMD results.
