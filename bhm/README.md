# Implementation of the BHM for KiDS-1000 Data

The data is downloaded from [here](https://kids.strw.leidenuniv.nl/DR4/data_files/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits).

The catalogue contains a total of 21,262,011 sources, and is presented as a single 16GB FITS table. For more information on the format of the data, see this [webpage](https://kids.strw.leidenuniv.nl/DR4/KiDS-1000_shearcatalogue.php).

## To Do
- (Done) Script to create a random set of $N$ galaxies.
- (Done) Redshift stacking.
- (Done) Kernel density estimation.
- Script to run the KiDS-1000 likelihood.
- Plot the lensing kernel for different redshift distributions.
- Plot the different templates.
- Plot the different filters.

<br/>

- Understand Boris' notebook.
- Think about how to do tomography $n_{i}(z)$. Do we have a good technique for this yet?

### Reference Papers (BHM)
- [Benitez 2000](https://iopscience.iop.org/article/10.1086/308947)
- [Leistedt et al. 2016
  ](https://academic.oup.com/mnras/article/460/4/4258/2609193?login=false)
- [Sanchez et al. 2019](https://academic.oup.com/mnras/article/483/2/2801/5218506)
- [Salvato et al. 2019](https://www.nature.com/articles/s41550-018-0478-0)
- [Alarcon et al. 2020](https://academic.oup.com/mnras/article/498/2/2614/5893329)
- [Leistedt et al. 2022](https://arxiv.org/abs/2207.07673)
- [Alsing et al. 2022](https://arxiv.org/abs/2207.05819)

### Reference Papers (Cosmology: KiDS-1000)
- [Wright et al. 2019](https://www.aanda.org/articles/aa/full_html/2019/12/aa34879-18/aa34879-18.html) - KV-450 data (optical and infrared)
- [Joachimi et al. 2021](https://doi.org/10.1051/0004-6361/202038831) - Methodology Paper
- [Hildebrandt et al. 2021](https://doi.org/10.1051/0004-6361/202039018) - KiDS-1000 Redshift distributions
- [Asgari et al. 2021](https://doi.org/10.1051/0004-6361/202039070) - KiDS-1000 Cosmology