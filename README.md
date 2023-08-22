FOCUS & MA-FOCUS
=====
FOCUS (Fine-mapping Of CaUsal gene Sets) is software to fine-map transcriptome-wide association study statistics at genomic risk regions. The software takes as input summary GWAS data along with eQTL weights and outputs a credible set of _genes_ to explain observed genomic risk.

MA-FOCUS (Multi-Ancestry Fine-mapping Of CaUsal gene Sets) is an extension of FOCUS that leverages summary GWAS data with eQTL weights from multiple ancestries to increase the precision of credible sets for causal genes.

```diff
- We detest usage of our software or scientific outcome to promote racial discrimination.
```

FOCUS is described in:

> [Probabilistic fine-mapping of transcriptome-wide association studies](https://www.nature.com/articles/s41588-019-0367-1). Nicholas Mancuso, Malika K. Freund, Ruth Johnson, Huwenbo Shi, Gleb Kichaev, Alexander Gusev, and Bogdan Pasaniuc. ***Nature Genetics*** 51, 675-682 (2019).

MA-FOCUS is described in:

> [Multi-ancestry fine-mapping improves precision to identify causal genes in transcriptome-wide association studies](https://www.cell.com/ajhg/fulltext/S0002-9297(22)00306-8). Zeyun Lu<sup>\*</sup>, Shyamalika Gopalan<sup>\*</sup>, Dong Yuan, David V. Conti, Bogdan Pasaniuc, Alexander Gusev, Nicholas Mancuso. ***American Journal of Human Genetics*** VOLUME 109, ISSUE 8, P1388-1404, AUGUST 04, 2022.

<sup>\*</sup> indicates equal contribution

Installing
----------
The easiest way to install updated version is with pip:


    git clone https://github.com/mancusolab/ma-focus.git
    cd ma-focus
    pip install .

*We currently only support Python3.6+.*

FOCUS Example
-------
Here is an example of how to perform LDL fine-mapping while prioritizing predictive models from adipose tissues:

    focus finemap LDL_2010.clean.sumstats.gz 1000G.EUR.QC.1 gtex_v7.db --chr 1 --tissue adipose --locations 37:EUR --out LDL_2010.chr1

This command will scan `LDL_2010.clean.sumstats.gz` for risk regions `37:EUR` generated by LDetect on GRCh37 for European ancestry and then perform TWAS+fine-mapping using LD estimated from plink-formatted `1000G.EUR.QC.1` and eQTL weights from `gtex_v7.db` GRCh37.

MA-FOCUS Example
-------
Here is an example of how to perform multi-ancestry fine-mapping from European and African ancestry (just use `:` to concatenate):

    focus finemap LDL.EUR.sumstats.gz:LDL.AFR.sumstats.gz 1000G.EUR.QC.1:1000G.AFR.QC.1 genoa_EUR.db:genoa_AFR.db --chr 1 --tissue LCL --locations 37:EUR-AFR --out LDL_mafocus_chr1

This command will scan GWAS summary data `LDL.EUR.sumstats.gz` and `LDL.AFR.sumstats.gz` for risk regions `37:EUR-AFR` generated by modified LDetect on GRCh37 for European and African ancestry and then perform TWAS+fine-mapping using LD estimated from plink-formatted `1000G.EUR.QC.1` and `1000G.AFR.QC.1` and eQTL weights from `genoa_EUR.db` and `genoa_AFR.db`.

Note: `--locations` is a new required parameter compared to versions prior to v0.8. This parameter specifies the genomic regions to be fine-mapped. We recommend to use independent regions for single ancestry or multiple ancestries. Please see the [wiki](https://github.com/mancusolab/focus/wiki) for more instructions.

Please see the [wiki](https://github.com/mancusolab/ma-focus/wiki) for more details on how to use focus, ma-focus and links to database files.

Notes
-----
Version 0.802: Fix the bug that .gitignore includes *.tsv so that gencode files couldn't be pushed to github.

Version 0.801: Added gencode_map_v38 and multiple LD block files in GRCh38. Fixed prior_prob bugs.

Version 0.8: Added MA-FOCUS support. Added GWAS imputation using imp-G. Added additional choice for prior probability for causal genes (number of genes in the risk regions).

Version 0.6.10: Fixed bug where weight database allele mismatch with GWAS broke inference.

Version 0.6.5: Fixed bug in newer versions of matplotlib not accepting string for colormaps. Fixed legend bug in plot. Fixed bug that mismatched string and category when supplying custom locations.

Version 0.6: Fixed bug where only one of the two alleles was reversed complemented breaking alignment. For now these instances are dropped. Added option `--use-ens-id` for FUSION import to indicate the main model label is an Ensembl ID rather than HGNC symbol.

Version 0.5: Plotting sorts genes based on tx start. Various bug fixes that limited the number of queried SNPs and plotting when using newer matplotlib.

Version 0.4: Added FUSION import support.

Version 0.3: Initial release. More to come soon.

Software and support
-----
If you have any questions or comments please contact nicholas.mancuso@med.usc.edu and zeyunlu@usc.edu

For performing various inferences using summary data from large-scale GWASs please find the following useful software:

1. Association between predicted expression and complex trait/disease [FUSION](https://github.com/gusevlab/fusion_twas) and [PrediXcan](https://github.com/hakyimlab/PrediXcan)
2. Estimating local heritability or genetic correlation [HESS](https://github.com/huwenboshi/hess)
3. Estimating genome-wide heritability or genetic correlation [UNITY](https://github.com/bogdanlab/UNITY)
4. Fine-mapping using summary-data [PAINTOR](https://github.com/gkichaev/PAINTOR_V3.0)
5. Imputing summary statistics using LD [FIZI](https://github.com/bogdanlab/fizi)
6. TWAS simulator (https://github.com/mancusolab/twas_sim)
