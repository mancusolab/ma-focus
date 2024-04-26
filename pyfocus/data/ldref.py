import warnings

import pyfocus as pf
import numpy as np
import pandas as pd
import pkg_resources

import scipy.linalg as lin

from pandas_plink import read_plink

dft_location = {"37:EUR": "ld_blocks/grch37.eur.loci.bed",
"37:AFR": "ld_blocks/grch37.afr.loci.bed",
"37:EAS": "ld_blocks/grch37.eas.loci.bed",
"37:EUR-AFR": "ld_blocks/grch37.eur.afr.loci.bed",
"37:EUR-EAS": "ld_blocks/grch37.eur.eas.loci.bed",
"37:EUR-EAS-AFR": "ld_blocks/grch37.eur.eas.afr.loci.bed",
"38:EUR": "ld_blocks/grch38.eur.loci.bed",
"38:AFR": "ld_blocks/grch38.afr.loci.bed",
"38:EAS": "ld_blocks/grch38.eas.loci.bed",
"38:EUR-AFR": "ld_blocks/grch38.eur.afr.loci.bed",
"38:EUR-EAS": "ld_blocks/grch38.eur.eas.loci.bed",
"38:EUR-EAS-AFR": "ld_blocks/grch38.eur.eas.afr.loci.bed"}

class IndBlocks(object):
    """
    Class to wrap/iterate approximately independent regions
    """

    CHRCOL = "chrom"
    STARTCOL = "start"
    STOPCOL = "stop"

    REQ_COLS = [CHRCOL, STARTCOL, STOPCOL]

    def __init__(self, regions=None):
        if regions is None:
            raise ValueError("Please specify independent regions location or default regions with '37:EUR', etc.")
        else:
            if type(regions) is str or pf.is_file(regions):
                if regions in dft_location.keys():
                    try:
                        dtype_dict = {IndBlocks.CHRCOL: "category", IndBlocks.STARTCOL: int, IndBlocks.STOPCOL: int}
                        local_ld_blocks = pkg_resources.resource_filename(__name__, dft_location[regions])
                        self._regions = pd.read_csv(local_ld_blocks, delim_whitespace=True, dtype=dtype_dict)
                    except Exception as e:
                        raise Exception("Fail population names specified. '37:EUR', ':EUR-EAS', etc. for more values, check documentation." + str(e))
                else:
                    try:
                        dtype_dict = {IndBlocks.CHRCOL: "category", IndBlocks.STARTCOL: int, IndBlocks.STOPCOL: int}
                        self._regions = pd.read_csv(regions, delim_whitespace=True, dtype=dtype_dict)
                    except Exception as e:
                        raise Exception("Parsing LD blocks failed:" + str(e))
            elif type(regions) is pd.core.frame.DataFrame:
                self._regions = regions
            else:
                raise Exception("Fail specfication on location name. Please specify it as a file path, '37:EUR', or similar valid input. see help.")

            for column in IndBlocks.REQ_COLS:
                if column not in self._regions:
                    raise ValueError(f"{column}-column not found in regions.")

            self._regions = self._regions[IndBlocks.REQ_COLS]

        return

    def subset_by_pos(self, chrom, start, stop):
        if chrom is None:
            raise ValueError("chrom argument cannot be `None` in subset_by_pos")

        df = self._regions.loc[self._regions[IndBlocks.CHRCOL] == chrom]
        if len(df) == 0:
            raise ValueError(f"No independent blocks found on chromosome {chrom}")

        if stop is None:
            stop = max(df[IndBlocks.STOPCOL])
        if start is None:
            start = min(df[IndBlocks.STARTCOL])

        # grab the intervals that overap the provided start and stop positions on same chromosome
        locs = df.loc[df.apply(lambda x: min(x[IndBlocks.STOPCOL], stop) - max(x[IndBlocks.STARTCOL], start), axis=1) > 0]

        if len(locs) == 0:
            raise ValueError(f"No independent blocks found at region {chrom}{start}-{stop}")

        return IndBlocks(locs)

    def __iter__(self):
        for row in self._regions.itertuples(name=None):
            # drop the index
            yield row[1:]

        return

    def show_nrow(self):
        return len(self._regions.index)


gencode_location = {"gencode37": "gencode/gencode_map_v37.tsv",
"gencode38": "gencode/gencode_map_v38.tsv"}

class GencodeBlocks(object):
    """
    Class to wrap/iterate approximately gencode regions
    """

    CHRCOL = "chrom"
    STARTCOL = "start"
    STOPCOL = "stop"
    GENENAME = "gene_name"

    REQ_COLS = [CHRCOL, STARTCOL, STOPCOL, GENENAME]

    def __init__(self, regions=None):
        if regions is None:
            raise ValueError("Please specify gencode file location")
        elif regions == "gencode37" or regions == "gencode38":
            try:
                dtype_dict = {GencodeBlocks.CHRCOL: "category", GencodeBlocks.STARTCOL: int, GencodeBlocks.STOPCOL: int, GencodeBlocks.GENENAME: str}
                gencode_blocks = pkg_resources.resource_filename(__name__, gencode_location[regions])
                self._regions = pd.read_csv(gencode_blocks, dtype=dtype_dict, sep = "\t")
            except Exception as e:
                raise Exception("Data folder doesn't contain gencode files " + str(e))
        else:
            if type(regions) is str or pf.is_file(regions):
                try:
                    dtype_dict = {GencodeBlocks.CHRCOL: "category", GencodeBlocks.STARTCOL: int, GencodeBlocks.STOPCOL: int, GencodeBlocks.GENENAME: str}
                    self._regions = pd.read_csv(regions, dtype=dtype_dict, sep='\t')
                except Exception as e:
                    raise Exception("Parsing your own gencode file failed. Make sure the path is correct and column has chr, start, stop, and gene_name:" + str(e))
            else:
                raise Exception("Please check your gencode file path" + str(e))

            for column in GencodeBlocks.REQ_COLS:
                if column not in self._regions:
                    raise ValueError(f"{column}-column not found in regions.")

            self._regions = self._regions[GencodeBlocks.REQ_COLS]

        return

    def subset_by_pos(self, chrom, start, stop):
        if chrom is None:
            raise ValueError("chrom argument cannot be `None` in subset_by_pos")

        df = self._regions.loc[self._regions[GencodeBlocks.CHRCOL] == chrom]
        if len(df) == 0:
            raise ValueError(f"No gencode information found on chromosome {chrom}")

        if stop is None:
            stop = max(df[GencodeBlocks.STOPCOL])
        if start is None:
            start = min(df[GencodeBlocks.STARTCOL])

        # grab the intervals that overap the provided start and stop positions on same chromosome
        locs = df.loc[df.apply(lambda x: min(x[GencodeBlocks.STOPCOL], stop) - max(x[GencodeBlocks.STARTCOL], start), axis=1) > 0]

        if len(locs) == 0:
            prior_prob = 0.001
        else:
            prior_prob = 1 / len(locs[GencodeBlocks.GENENAME].unique())

        return prior_prob, len(locs)

    def __iter__(self):
        for row in self._regions.itertuples(name=None):
            # drop the index
            yield row[1:]

        return

class LDRefPanel(object):
    CHRCOL = "chrom"
    SNPCOL = "snp"
    BPCOL = "pos"
    A1COL = "a1"
    A2COL = "a0"

    # chrom snp  cm pos a0 a1 i

    def __init__(self, snp_info, sample_info, geno):
        self._snp_info = snp_info
        if len(snp_info) > 0 and pd.api.types.is_categorical_dtype(self._snp_info[LDRefPanel.A1COL]):
            self._snp_info.loc[:, LDRefPanel.A1COL] = self._snp_info[LDRefPanel.A1COL].astype("str")
            self._snp_info.loc[:, LDRefPanel.A2COL] = self._snp_info[LDRefPanel.A2COL].astype("str")
        self._sample_info = sample_info
        self._geno = geno
        return

    def __len__(self):
        return len(self._snp_info)

    def __str__(self):
        start_chr = self._snp_info[LDRefPanel.CHRCOL].iloc[0]
        stop_chr = self._snp_info[LDRefPanel.CHRCOL].iloc[-1]

        start_bp = self._snp_info[LDRefPanel.BPCOL].iloc[0]
        stop_bp = self._snp_info[LDRefPanel.BPCOL].iloc[-1]
        return f"{start_chr}:{int(start_bp)} - {stop_chr}:{int(stop_bp)}"

    def subset_by_pos(self, chrom, start=None, stop=None, clean_snps=True):
        df = self._snp_info
        if start is not None and stop is not None:
            snps = df.loc[(df[LDRefPanel.CHRCOL] == chrom) & (df[LDRefPanel.BPCOL] >= start) & (df[LDRefPanel.BPCOL] <= stop)]
        elif start is not None and stop is None:
            snps = df.loc[(df[LDRefPanel.CHRCOL] == chrom) & (df[LDRefPanel.BPCOL] >= start)]
        elif start is None and stop is not None:
            snps = df.loc[(df[LDRefPanel.CHRCOL] == chrom) & (df[LDRefPanel.BPCOL] <= stop)]
        else:
            snps = df.loc[(df[LDRefPanel.CHRCOL] == chrom)]

        if clean_snps and len(snps) > 0:
            valid = pf.check_valid_snp(snps[LDRefPanel.A1COL], snps[LDRefPanel.A2COL])
            snps = snps.loc[valid].drop_duplicates(subset=pf.LDRefPanel.SNPCOL)

        return LDRefPanel(snps, self._sample_info, self._geno)

    def overlap_gwas(self, gwas, enable_impg = False):
        df = self._snp_info
        how = "inner" if not enable_impg else "right"
        merged_snps = pd.merge(gwas, df, how=how, left_on=pf.GWAS.SNPCOL, right_on=pf.LDRefPanel.SNPCOL)
        return merged_snps

    def get_geno(self, snps=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            if snps is None:
                return self._geno.compute().T
            else:
                return self._geno[snps.i.values, :].compute().T

    @property
    def sample_size(self):
        return len(self._sample_info)

    def estimate_ld(self, snps=None, adjust=0.1, return_eigvals=False):
        G = self.get_geno(snps)
        n, p = G.shape
        col_mean = np.nanmean(G, axis=0)

        # impute missing with column mean
        inds = np.where(np.isnan(G))
        G[inds] = np.take(col_mean, inds[1])

        if return_eigvals:
            G = (G - np.mean(G, axis=0)) / np.std(G, axis=0)
            _, S, V = lin.svd(G, full_matrices=True)

            # adjust
            D = np.full(p, adjust)
            D[:len(S)] = D[:len(S)] + (S**2 / n)

            return np.dot(V.T * D, V), D
        else:
            return np.corrcoef(G.T) + np.eye(p) * adjust

    @classmethod
    def parse_plink(cls, path):
        bim, fam, bed = read_plink(path, verbose=False)
        return LDRefPanel(bim, fam, bed)
