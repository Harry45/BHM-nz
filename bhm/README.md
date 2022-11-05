# Implementation of the BHM for KiDS-1000 Data

The data is downloaded from [here](https://kids.strw.leidenuniv.nl/DR4/data_files/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits).

The catalogue contains a total of 21,262,011 sources, and is presented as a
single 16GB FITS table. For more information on the format of the data, see this [webpage](https://kids.strw.leidenuniv.nl/DR4/KiDS-1000_shearcatalogue.php).

## To Do
- Script to create a random set of $N$ galaxies.
- Redshift stacking.
- Kernel density estimation.
- Script to run the KiDS-1000 likelihood.
<br/>
- Plot the different templates.
- Plot the different filters.
- Understand Boris' notebook.
- Think about how to do tomography $n(z)$. Do we have a good technique for this yet?

## Reference Papers
- [Benitez 2000](https://iopscience.iop.org/article/10.1086/308947)
- [Leistedt et al. 2016
  ](https://academic.oup.com/mnras/article/460/4/4258/2609193?login=false)
- [Sanchez et al. 2019](https://academic.oup.com/mnras/article/483/2/2801/5218506)
- [Alarcon et al. 2020](https://academic.oup.com/mnras/article/498/2/2614/5893329)