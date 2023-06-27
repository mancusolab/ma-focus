#!/usr/bin/env python

# This is a modified munge_sumstat.py from LDSC project
# FOCUS needed a similar tool but with a few extra output columns
# Credit to Brendan Bulik-Sullivan and Hilary Finucane

from __future__ import division

import argparse
import bz2
import gzip
import logging
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

from scipy.stats import chi2
from sqlalchemy import exc as sa_exc

import pyfocus


np.seterr(invalid="ignore")


numeric_cols = [
    "P",
    "N",
    "N_CAS",
    "N_CON",
    "Z",
    "OR",
    "BETA",
    "LOG_ODDS",
    "INFO",
    "FRQ",
    "SIGNED_SUMSTAT",
    "NSTUDY",
]


def parse_pint(str):
    try:
        value = int(str)
        if value <= 0:
            raise ValueError()
    except ValueError:
        raise argparse.ArgumentTypeError("Value needs to be positive interger")

    return value


def parse_prob(str):
    try:
        value = float(str)
        if value < 0 or value > 1:
            raise ValueError()
    except ValueError:
        raise argparse.ArgumentTypeError("Value has to be between 0 and 1")

    return value


def parse_chisq(str):
    try:
        value = float(str)
        if value < 0:
            raise ValueError()
    except ValueError:
        raise argparse.ArgumentTypeError("Value has to be at least 0")

    return value


def parse_pos(pos, option):
    """
    Parse a specified genomic position.
    Should be digits followed optionally by case-insensitive Mb or Kb modifiers.
    """
    match = re.match("^(([0-9]*[.])?[0-9]+)(mb|kb)?$", pos, flags=re.IGNORECASE)
    if match:
        pos_tmp = float(match.group(1))  # position
        pos_mod = match.group(3)  # modifier
        if pos_mod:
            pos_mod = pos_mod.upper()
            if pos_mod == "MB":
                pos_tmp *= 1000000
            elif pos_mod == "KB":
                pos_tmp *= 1000

        position = pos_tmp
    else:
        raise ValueError(
            "Option {} {} is an invalid genomic position".format(option, pos)
        )

    return position


def parse_locations(locations, chrom=None, start_bp=None, stop_bp=None):
    """
    Parse user-specified BED file with [CHR, START, STOP] windows defining where to perform
    imputation.

    If user also specified chr, start-bp, or stop-bp arguments filter on those as well.
    """
    for idx, line in enumerate(locations):
        # skip comments
        if "#" in line:
            continue

        row = line.split()

        if len(row) < 3:
            raise ValueError(
                "Line {} in locations file does not contain [CHR, START, STOP]".format(
                    idx
                )
            )

        chrom_arg = row[0]
        start_arg = parse_pos(row[1], "start argument in locations file")
        stop_arg = parse_pos(row[2], "stop argument in locations file")

        if chrom is not None and chrom_arg != chrom:
            continue
        elif start_bp is not None and start_arg < start_bp:
            continue
        elif stop_bp is not None and stop_arg > stop_bp:
            continue

        yield [chrom_arg, start_arg, stop_arg]

    return


def get_command_string(args):
    """
    Format cli.py call and options into a string for logging/printing

    :return: string containing formatted arguments to cli.py
    """

    base = f"focus {args[0]}{os.linesep}"
    rest = args[1:]
    rest_strs = []
    needs_tab = True
    for cmd in rest:
        if "-" == cmd[0]:
            if cmd in [
                "--quiet",
                "-q",
                "--verbose",
                "-v",
                "--plot",
                "--strict-tissue",
                "--use-ens-id",
                "--from-gencode",
            ]:
                rest_strs.append("\t{}{}".format(cmd, os.linesep))
                needs_tab = True
            else:
                rest_strs.append("\t{}".format(cmd))
                needs_tab = False
        else:
            if needs_tab:
                rest_strs.append("\t{}{}".format(cmd, os.linesep))
                needs_tab = True
            else:
                rest_strs.append(" {}{}".format(cmd, os.linesep))
                needs_tab = True

    return base + "".join(rest_strs) + os.linesep


def read_header(fh):
    """Read the first line of a file and returns a list with the column names."""
    openfunc, compression = get_compression(fh)
    return [x.rstrip("\n") for x in openfunc(fh).readline().split()]


def get_cname_map(flag, default, ignore):
    """
    Figure out which column names to use.

    Priority is
    (1) ignore everything in ignore
    (2) use everything in flags that is not in ignore
    (3) use everything in default that is not in ignore or in flags

    The keys of flag are cleaned. The entries of ignore are not cleaned. The keys of defualt
    are cleaned. But all equality is modulo clean_header().

    """
    clean_ignore = {clean_header(x) for x in ignore}
    both = clean_ignore | set(flag)
    cname_map = {x: flag[x] for x in flag if x not in clean_ignore}
    cname_map.update({x: default[x] for x in default if x not in both})

    return cname_map


def get_compression(fh):
    """
    Read filename suffixes and figure out whether it is gzipped,bzip2'ed or not compressed
    """

    """Which sort of compression should we use with read_csv?"""
    if hasattr(fh, "name"):
        _, ext = os.path.splitext(fh.name)
    elif isinstance(fh, str):
        _, ext = os.path.splitext(fh)
    else:
        raise ValueError("get_compression: argument must be file handle or path")

    if ext.endswith("gz"):
        compression = "gzip"
        openfunc = lambda x: gzip.open(x, "rt")
    elif ext.endswith("bz2"):
        compression = "bz2"
        openfunc = lambda x: bz2.open(x, "rt")
    else:
        openfunc = open
        compression = None

    return openfunc, compression


def clean_header(header):
    """
    For cleaning file headers.
    - convert to uppercase
    - replace dashes '-' with underscores '_'
    - replace dots '.' (as in R) with underscores '_'
    - remove newlines ('\n')
    """
    return header.upper().replace("-", "_").replace(".", "_").replace("\n", "")


def filter_pvals(pvals, log):
    """Remove out-of-bounds P-values"""

    ii = (pvals > 0) & (pvals <= 1)
    bad_p = (~ii).sum()
    if bad_p > 0:
        msg = "{N} SNPs had P outside of (0,1]. The P column may be mislabeled"
        log.warning(msg.format(N=bad_p))

    return ii


def filter_info(info, log, args):
    """Remove INFO < args.info_min (default 0.9) and complain about out-of-bounds INFO."""

    if type(info) is pd.Series:  # one INFO column
        jj = ((info > 2.0) | (info < 0)) & info.notnull()
        ii = info >= args.info_min
    elif type(info) is pd.DataFrame:  # several INFO columns
        jj = ((info > 2.0) & info.notnull()).any(axis=1) | (
            (info < 0) & info.notnull()
        ).any(axis=1)
        ii = info.sum(axis=1) >= args.info_min * (len(info.columns))
    else:
        raise ValueError("Expected pd.DataFrame or pd.Series.")

    bad_info = jj.sum()
    if bad_info > 0:
        msg = "{N} SNPs had INFO outside of [0,1.5]. The INFO column may be mislabeled"
        log.warning(msg.format(N=bad_info))

    return ii


def filter_frq(frq, log, args):
    """
    Filter on MAF. Remove MAF < args.maf_min and out-of-bounds MAF.
    """
    jj = (frq < 0) | (frq > 1)
    bad_frq = jj.sum()
    if bad_frq > 0:
        msg = "{N} SNPs had FRQ outside of [0,1]. The FRQ column may be mislabeled"
        log.warning(msg.format(N=bad_frq))

    frq = np.minimum(frq, 1 - frq)
    ii = frq > args.maf_min
    return ii & ~jj


def parse_dat(dat_gen, convert_colname, log, args):
    """Parse and filter a sumstats file chunk-wise"""

    tot_snps = 0
    dat_list = []
    msg = "Reading sumstats from {F} into memory {N} SNPs at a time"
    log.info(msg.format(F=args.sumstats, N=int(args.chunksize)))
    drops = {"NA": 0, "P": 0, "INFO": 0, "FRQ": 0, "A": 0, "SNP": 0, "MERGE": 0}
    for block_num, dat in enumerate(dat_gen):
        log.info("Reading SNP chunk {}".format(block_num + 1))
        tot_snps += len(dat)
        old = len(dat)

        for c in dat.columns:
            # sometimes column types change when streaming the data
            if c in numeric_cols and not np.issubdtype(dat[c].dtype, np.number):
                log.warning(
                    "Column {} expected to be numeric. Attempting to convert".format(c)
                )
                dat[c] = pd.to_numeric(dat[c], errors="coerce")

        dat = dat.dropna(
            axis=0, how="any", subset=filter(lambda x: x != "INFO", dat.columns)
        ).reset_index(drop=True)
        drops["NA"] += old - len(dat)
        dat.columns = map(lambda x: convert_colname[x], dat.columns)

        ii = np.ones(len(dat), dtype=bool)

        if "INFO" in dat.columns:
            old = ii.sum()
            ii &= filter_info(dat["INFO"], log, args)
            new = ii.sum()
            drops["INFO"] += old - new

        if "FRQ" in dat.columns:
            old = ii.sum()
            ii &= filter_frq(dat["FRQ"], log, args)
            new = ii.sum()
            drops["FRQ"] += old - new

        old = ii.sum()
        if args.keep_maf:
            dat.drop([x for x in ["INFO"] if x in dat.columns], inplace=True, axis=1)
        else:
            dat.drop(
                [x for x in ["INFO", "FRQ"] if x in dat.columns], inplace=True, axis=1
            )

        ii &= filter_pvals(dat.P, log)
        new = ii.sum()
        drops["P"] += old - new
        old = new
        dat.A1 = dat.A1.str.upper()
        dat.A2 = dat.A2.str.upper()
        ii &= pyfocus.check_valid_snp(dat.A1, dat.A2)
        new = ii.sum()
        drops["A"] += old - new

        if ii.sum() == 0:
            continue

        dat_list.append(dat[ii].reset_index(drop=True))

    log.info("Done reading SNP chunks")
    dat = pd.concat(dat_list, axis=0).reset_index(drop=True)
    log.info("Read {N} SNPs from --sumstats file".format(N=tot_snps))
    log.info("Removed {N} SNPs with missing values".format(N=drops["NA"]))
    log.info(
        "Removed {N} SNPs with INFO <= {I}".format(N=drops["INFO"], I=args.info_min)
    )
    log.info("Removed {N} SNPs with MAF <= {M}".format(N=drops["FRQ"], M=args.maf_min))
    log.info("Removed {N} SNPs with out-of-bounds p-values".format(N=drops["P"]))
    log.info(
        "Removed {N} variants that were not SNPs or were strand-ambiguous".format(
            N=drops["A"]
        )
    )
    log.info("{N} SNPs remain".format(N=len(dat)))

    return dat


def process_n(dat, args, log):
    """Determine sample size from --N* flags or N* columns. Filter out low N SNPs.s"""

    if all(i in dat.columns for i in ["N_CAS", "N_CON"]):
        N = dat.N_CAS + dat.N_CON
        P = dat.N_CAS / N
        dat["N"] = N * P / P[N == N.max()].mean()
        dat.drop(["N_CAS", "N_CON"], inplace=True, axis=1)
        # NB no filtering on N done here -- that is done in the next code block

    if "N" in dat.columns:
        n_min = args.n_min if args.n_min else dat.N.quantile(0.9) / 1.5
        old = len(dat)
        dat = dat[dat.N >= n_min].reset_index(drop=True)
        new = len(dat)
        log.info(
            "Removed {M} SNPs with N < {MIN} ({N} SNPs remain)".format(
                M=old - new, N=new, MIN=n_min
            )
        )

    elif "NSTUDY" in dat.columns and "N" not in dat.columns:
        nstudy_min = args.nstudy_min if args.nstudy_min else dat.NSTUDY.max()
        old = len(dat)
        dat = (
            dat[dat.NSTUDY >= nstudy_min]
            .drop(["NSTUDY"], axis=1)
            .reset_index(drop=True)
        )
        new = len(dat)
        log.info(
            "Removed {M} SNPs with NSTUDY < {MIN} ({N} SNPs remain)".format(
                M=old - new, N=new, MIN=nstudy_min
            )
        )

    if "N" not in dat.columns:
        if args.N:
            dat["N"] = args.N
            log.info("Using N = {N}".format(N=args.N))
        elif args.N_cas and args.N_con:
            dat["N"] = args.N_cas + args.N_con
        else:
            raise ValueError(
                "Cannot determine N. This message indicates a bug. N should have been checked earlier in the program."
            )

    return dat


def p_to_z(pvals):
    """Convert P-value and N to standardized beta."""
    return np.sqrt(chi2.isf(pvals, 1))


def pass_median_check(m, expected_median, tolerance):
    """Check that median(x) is within tolerance of expected_median."""
    return np.abs(m - expected_median) <= tolerance


def parse_flag_cnames(args):
    """
    Parse flags that specify how to interpret nonstandard column names.

    flag_cnames is a dict that maps (cleaned) arguments to internal column names
    """

    cname_options = [
        [args.nstudy, "NSTUDY", "--nstudy"],
        [args.snp, "SNP", "--snp"],
        [args.N_col, "N", "--N"],
        [args.N_cas_col, "N_CAS", "--N-cas-col"],
        [args.N_con_col, "N_CON", "--N-con-col"],
        [args.a1, "A1", "--a1"],
        [args.a2, "A2", "--a2"],
        [args.p, "P", "--P"],
        [args.frq, "FRQ", "--nstudy"],
        [args.info, "INFO", "--info"],
    ]
    flag_cnames = {clean_header(x[0]): x[1] for x in cname_options if x[0] is not None}
    if args.info_list:
        try:
            flag_cnames.update(
                {clean_header(x): "INFO" for x in args.info_list.split(",")}
            )
        except ValueError:
            raise ValueError(
                "The argument to --info-list should be a comma-separated list of column names"
            )

    null_value = None
    if args.signed_sumstats:
        try:
            cname, null_value = args.signed_sumstats.split(",")
            null_value = float(null_value)
            flag_cnames[clean_header(cname)] = "SIGNED_SUMSTAT"
        except ValueError:
            raise ValueError(
                "The argument to --signed-sumstats should be column header comma number"
            )

    return [flag_cnames, null_value]


def munge(args):
    # This is a modified munge_sumstat.py from LDSC project
    # FIZI needed a similar tool but with a few extra output columns
    # Credit to Brendan Bulik-Sullivan and Hilary Finucane

    log = logging.getLogger(pyfocus.LOG)

    null_values = {"LOG_ODDS": 0, "BETA": 0, "OR": 1, "Z": 0}

    default_cnames = {
        # Chromosome
        "CHR": "CHR",
        "CHROM": "CHR",
        # BP
        "BP": "BP",
        "POS": "BP",
        # RS NUMBER
        "SNP": "SNP",
        "MARKERNAME": "SNP",
        "SNPID": "SNP",
        "RS": "SNP",
        "RSID": "SNP",
        "RS_NUMBER": "SNP",
        "RS_NUMBERS": "SNP",
        # NUMBER OF STUDIES
        "NSTUDY": "NSTUDY",
        "N_STUDY": "NSTUDY",
        "NSTUDIES": "NSTUDY",
        "N_STUDIES": "NSTUDY",
        # P-VALUE
        "P": "P",
        "PVALUE": "P",
        "P_VALUE": "P",
        "PVAL": "P",
        "P_VAL": "P",
        "GC_PVALUE": "P",
        # ALLELE 1
        "A1": "A1",
        "ALLELE1": "A1",
        "ALLELE_1": "A1",
        "EFFECT_ALLELE": "A1",
        "REFERENCE_ALLELE": "A1",
        "INC_ALLELE": "A1",
        "EA": "A1",
        # ALLELE 2
        "A2": "A2",
        "ALLELE2": "A2",
        "ALLELE_2": "A2",
        "OTHER_ALLELE": "A2",
        "NON_EFFECT_ALLELE": "A2",
        "DEC_ALLELE": "A2",
        "NEA": "A2",
        # N
        "N": "N",
        "NCASE": "N_CAS",
        "CASES_N": "N_CAS",
        "N_CASE": "N_CAS",
        "N_CASES": "N_CAS",
        "N_CONTROLS": "N_CON",
        "N_CAS": "N_CAS",
        "N_CON": "N_CON",
        "NCONTROL": "N_CON",
        "CONTROLS_N": "N_CON",
        "N_CONTROL": "N_CON",
        "WEIGHT": "N",  # metal does this. possibly risky.
        # SIGNED STATISTICS
        "ZSCORE": "Z",
        "Z-SCORE": "Z",
        "GC_ZSCORE": "Z",
        "Z": "Z",
        "OR": "OR",
        "B": "BETA",
        "BETA": "BETA",
        "LOG_ODDS": "LOG_ODDS",
        "EFFECTS": "BETA",
        "EFFECT": "BETA",
        "SIGNED_SUMSTAT": "SIGNED_SUMSTAT",
        # INFO
        "INFO": "INFO",
        # MAF
        "EAF": "FRQ",
        "FRQ": "FRQ",
        "MAF": "FRQ",
        "FRQ_U": "FRQ",
        "F_U": "FRQ",
    }

    describe_cname = {
        "CHR": "Chromsome",
        "BP": "Base position",
        "SNP": "Variant ID (e.g., rs number)",
        "P": "p-Value",
        "A1": "Allele 1, interpreted as ref allele for signed sumstat",
        "A2": "Allele 2, interpreted as non-ref allele for signed sumstat",
        "N": "Sample size",
        "N_CAS": "Number of cases",
        "N_CON": "Number of controls",
        "Z": "Z-score (0 --> no effect; above 0 --> A1 is trait/risk increasing)",
        "OR": "Odds ratio (1 --> no effect; above 1 --> A1 is risk increasing)",
        "BETA": "[linear/logistic] regression coefficient (0 --> no effect; above 0 --> A1 is trait/risk increasing)",
        "LOG_ODDS": "Log odds ratio (0 --> no effect; above 0 --> A1 is risk increasing)",
        "INFO": "INFO score (imputation quality; higher --> better imputation)",
        "FRQ": "Allele frequency",
        "SIGNED_SUMSTAT": "Directional summary statistic as specified by --signed-sumstats",
        "NSTUDY": "Number of studies in which the SNP was genotyped",
    }

    try:
        file_cnames = read_header(args.sumstats)  # note keys not cleaned
        flag_cnames, signed_sumstat_null = parse_flag_cnames(args)
        if args.ignore:
            ignore_cnames = [clean_header(x) for x in args.ignore.split(",")]
        else:
            ignore_cnames = []

        # remove LOG_ODDS, BETA, Z, OR from the default list
        if args.signed_sumstats is not None or args.a1_inc:
            mod_default_cnames = {
                x: default_cnames[x]
                for x in default_cnames
                if default_cnames[x] not in null_values
            }
        else:
            mod_default_cnames = default_cnames

        cname_map = get_cname_map(flag_cnames, mod_default_cnames, ignore_cnames)
        cname_translation = {
            x: cname_map[clean_header(x)]
            for x in file_cnames
            if clean_header(x) in cname_map
        }
        cname_description = {
            x: describe_cname[cname_translation[x]] for x in cname_translation
        }

        if args.signed_sumstats is None and not args.a1_inc:
            sign_cnames = [
                x for x in cname_translation if cname_translation[x] in null_values
            ]
            if len(sign_cnames) > 1:
                raise ValueError(
                    "Too many signed sumstat columns. Specify which to ignore with the --ignore flag."
                )
            if len(sign_cnames) == 0:
                raise ValueError("Could not find a signed summary statistic column.")

            sign_cname = sign_cnames[0]
            signed_sumstat_null = null_values[cname_translation[sign_cname]]
            cname_translation[sign_cname] = "SIGNED_SUMSTAT"
        else:
            sign_cname = "SIGNED_SUMSTATS"

        # check that we have all the columns we need
        if not args.a1_inc:
            req_cols = ["CHR", "BP", "SNP", "A1", "A2", "P", "SIGNED_SUMSTAT"]
        else:
            req_cols = ["CHR", "BP", "SNP", "A1", "A2", "P"]

        for c in req_cols:
            if c not in cname_translation.values():
                raise ValueError("Could not find {C} column.".format(C=c))

        # check aren't any duplicated column names in mapping
        for field in cname_translation:
            numk = file_cnames.count(field)
            if numk > 1:
                raise ValueError(
                    "Found {num} columns named {C}".format(C=field, num=str(numk))
                )

        # check multiple different column names don't map to same data field
        check = set([])
        for head in cname_translation.values():
            if head in check:
                raise ValueError("Found multiple {C} columns".format(C=head))
            else:
                check.add(head)

        if (
            (not args.N)
            and (not (args.N_cas and args.N_con))
            and ("N" not in cname_translation.values())
            and (any(x not in cname_translation.values() for x in ["N_CAS", "N_CON"]))
        ):
            raise ValueError("Could not determine N column.")
        if (
            "N" in cname_translation.values()
            or all(x in cname_translation.values() for x in ["N_CAS", "N_CON"])
        ) and "NSTUDY" in cname_translation.values():
            nstudy = [x for x in cname_translation if cname_translation[x] == "NSTUDY"]
            for x in nstudy:
                del cname_translation[x]
        if not all(x in cname_translation.values() for x in ["A1", "A2"]):
            raise ValueError("Could not find A1/A2 columns.")

        log.info("Interpreting column names as follows:")
        for x in cname_description:
            log.info(x + ": " + cname_description[x])

        _, compression = get_compression(args.sumstats)

        # figure out which columns are going to involve sign information, so we can ensure
        # they're read as floats
        signed_sumstat_cols = [
            k for k, v in cname_translation.items() if v == "SIGNED_SUMSTAT"
        ]
        dtypes = {c: np.float64 for c in signed_sumstat_cols}
        dtypes["CHR"] = "category"
        dat_gen = pd.read_csv(
            args.sumstats,
            delim_whitespace=True,
            header=0,
            compression=compression,
            usecols=cname_translation.keys(),
            na_values=[".", "NA"],
            iterator=True,
            chunksize=args.chunksize,
            dtype=dtypes,
        )

        dat = parse_dat(dat_gen, cname_translation, log, args)
        if len(dat) == 0:
            raise ValueError("After applying filters, no SNPs remain.")

        old = len(dat)
        dat = dat.drop_duplicates(subset="SNP").reset_index(drop=True)
        new = len(dat)
        log.info(
            "Removed {M} SNPs with duplicated rs numbers ({N} SNPs remain).".format(
                M=old - new, N=new
            )
        )
        # filtering on N cannot be done chunkwise
        dat = process_n(dat, args, log)
        dat.P = p_to_z(dat.P)
        dat.rename(columns={"P": "Z"}, inplace=True)
        if not args.a1_inc:
            m = np.median(dat.SIGNED_SUMSTAT)
            if not pass_median_check(m, signed_sumstat_null, 0.1):
                msg = "Median value of {F} is {V} (should be close to {M}). This column may be mislabeled."
                msg = msg.format(F=sign_cname, M=signed_sumstat_null, V=round(m, 2))
                log.warning(msg)
            else:
                msg = "Median value of {F} was {C}, which seems sensible.".format(
                    C=m, F=sign_cname
                )
                log.info(msg)

            dat.Z *= (-1) ** (dat.SIGNED_SUMSTAT < signed_sumstat_null)
            dat.drop("SIGNED_SUMSTAT", inplace=True, axis=1)

        out_fname = args.output + ".sumstats.gz"
        print_colnames = []
        for c in ["CHR", "SNP", "BP", "A1", "A2", "FRQ", "Z", "N"]:
            if c in dat.columns:
                if c == "FRQ" and not args.keep_maf:
                    continue

                print_colnames.append(c)

        msg = (
            "Writing summary statistics for {M} SNPs ({N} with nonmissing beta) to {F}."
        )
        log.info(msg.format(M=len(dat), F=out_fname, N=dat.N.notnull().sum()))

        dat.to_csv(
            out_fname,
            sep="\t",
            index=False,
            columns=print_colnames,
            float_format="%.3f",
            compression="gzip",
        )

        CHISQ = dat.Z**2
        mean_chisq = CHISQ.mean()

        log.info("METADATA - Mean chi^2 = " + str(round(mean_chisq, 3)))
        if mean_chisq < 1.02:
            log.warning("METADATA: Mean chi^2 may be too small")

        log.info("METADATA - Lambda GC = " + str(round(CHISQ.median() / 0.4549, 3)))
        log.info("METADATA - Max chi^2 = " + str(round(CHISQ.max(), 3)))
        log.info(
            "METADATA - {N} Genome-wide significant SNPs (some may have been removed by filtering)".format(
                N=(CHISQ > 29).sum()
            )
        )

    except Exception as err:
        log.error(err)
    finally:
        log.info("Conversion finished")

    return


def region_sanity_check(args_chr, args_start, args_stop):
    chrom = None
    start_bp = None
    stop_bp = None
    if any(x is not None for x in [args_chr, args_start, args_stop]):
        if args_start is not None and args_chr is None:
            raise ValueError("Option --start cannot be set unless --chr is specified")
        if args_stop is not None and args_chr is None:
            raise ValueError("Option --stop cannot be set unless --chr is specified")
        chrom = args_chr
        # parse start/stop positions and make sure they're ordered (if exist)
        if args_start is not None:
            start_bp = parse_pos(args_start, "--start")
        else:
            start_bp = None

        if args_stop is not None:
            stop_bp = parse_pos(args_stop, "--stop")
        else:
            stop_bp = None

        if args_start is not None and args_stop is not None:
            if start_bp >= stop_bp:
                raise ValueError(
                    "Specified --start position must be before --stop position"
                )

    return chrom, start_bp, stop_bp


def run_twas(args):
    log = logging.getLogger(pyfocus.LOG)

    try:
        # perform sanity arguments checking before continuing
        chrom, start_bp, stop_bp = region_sanity_check(args.chr, args.start, args.stop)

        # load GWAS summary data for each population
        # log.info("Preparing GWAS summary data.")

        df_paths = args.gwas.split(":")
        n_pop = len(df_paths)
        log.info(f"Detecting {n_pop} populations for fine-mapping.")

        if n_pop > 1:
            log.info(
                "Running single-population FOCUS on each population, and then MA-FOCUS across all populations."
            )
        else:
            log.info("Running single-population FOCUS.")

        gwas = [None] * n_pop
        for i in range(n_pop):
            log.info(f"Preparing GWAS summary file for population at {df_paths[i]}.")
            gwas_tmp = pyfocus.GWAS.parse_gwas(df_paths[i])
            if len(gwas_tmp) == 0:
                raise ValueError(
                    f"No GWAS summary file for population at {df_paths[i]}."
                )
            gwas[i] = gwas_tmp

        # if chrom is supplied just filter here
        if chrom is not None:
            for i in range(n_pop):
                gwas[i] = gwas[i].subset_by_pos(chrom)

                if len(gwas[i]) == 0:
                    err_str = (
                        f"No GWAS SNPs found at chromosome {chrom} at {df_paths[i]}."
                    )
                    raise ValueError(err_str)

        # load reference genotype data for each population
        # log.info("Preparing reference SNP data.")
        df_refs = args.ref.split(":")

        if len(df_refs) != n_pop:
            raise Exception(
                "The number of LD refernece panel is different from the number of GWAS data."
            )

        ref = [None] * n_pop
        for i in range(n_pop):
            log.info(f"Preparing reference SNP data for population at {df_refs[i]}.")
            ref_tmp = pyfocus.LDRefPanel.parse_plink(df_refs[i])
            if len(ref_tmp) == 0:
                raise ValueError(
                    f"No reference SNP data for population at {df_refs[i]}."
                )
            ref[i] = ref_tmp

        # if chrom is supplied just filter here
        if chrom is not None:
            for i in range(n_pop):
                ref[i] = ref[i].subset_by_pos(chrom)
                if len(ref[i]) == 0:
                    err_str = f"No reference LD SNPs found at chromosome {chrom} at {df_refs[i]}."
                    raise Exception(err_str)

        df_dbs = args.weights.split(":")

        if len(df_dbs) != n_pop:
            raise Exception(
                "The number of weight database is different from the number of GWAS data."
            )

        session = [None] * n_pop
        for i in range(n_pop):
            # not the best approach, but load_db creates an empty db if the file does not exist
            # so check here that we actually -have- some db
            if not os.path.isfile(df_dbs[i]):
                raise ValueError(f"Cannot find database at {df_dbs[i]}.")

        for i in range(n_pop):
            log.info(f"Preparing weight database at {df_dbs[i]}.")
            session[i] = pyfocus.load_db(df_dbs[i], idx=i)

        # alias
        Weight = pyfocus.models.Weight
        Model = pyfocus.models.Model
        MolecularFeature = pyfocus.models.MolecularFeature
        RefPanel = pyfocus.models.RefPanel

        with open("{}.cli.py.tsv".format(args.output), "w") as output:
            log.info(f"Preparing user-defined locations at {args.locations}.")

            partitions = pyfocus.IndBlocks(args.locations)
            log.info(
                f"Found {partitions.show_nrow()} independent regions on the entire genome."
            )

            if chrom is not None:
                partitions = partitions.subset_by_pos(chrom, start_bp, stop_bp)
                _local_n = partitions.show_nrow()
                log.info(
                    f"{_local_n} independent regions detected after filtering on chromosome, start, and stop."
                )

            written = False
            _thold = args.p_threshold
            for region in partitions:
                chrom, start, stop = region
                block = f"{chrom}:{int(start)}-{chrom}:{int(stop)}"
                log.info(
                    f"Preparing data at region {block}. Skipping if following warning occurs."
                )

                # Decide prior prob for a gene to be causal
                log.info("Deciding prior probability for a gene to be causal.")

                # conver prior_prob to float if it's float
                try:
                    arg_prior_prob = float(args.prior_prob)
                    if 0 < arg_prior_prob < 1:
                        prior_prob = arg_prior_prob
                        log.info(
                            f"Using fixed numeric prior probability {args.prior_prob}."
                        )
                    else:
                        raise ValueError(
                            f"Numeric prior probability {arg_prior_prob} is invalid."
                        )
                except ValueError:
                    gencodeBlocks = pyfocus.GencodeBlocks(args.prior_prob)
                    prior_prob = gencodeBlocks.subset_by_pos(chrom, start, stop)
                    log.info(f"Using gencode file prior probability {prior_prob}.")

                # grab local GWAS data
                local_gwas = [None] * n_pop
                skip = False
                ct = 0
                for i in range(n_pop):
                    local_gwas_tmp = gwas[i].subset_by_pos(
                        chrom, start=start, stop=stop
                    )
                    # only fine-map regions that contain GWAS data
                    if len(local_gwas_tmp) == 0:
                        log.warning(
                            f"No GWAS data found found at region {block} at {df_paths[i]}. Skipping."
                        )
                        skip = True
                        break

                    # only fine-map regions that contain GWAS signal
                    if min(local_gwas_tmp.P) >= args.p_threshold:
                        # The idea is for single pop, it gives warning skipping message later
                        # For multiple pops, it gives no GWAS info here, and give warning skipping message later
                        if n_pop != 1:
                            log.info(
                                f"No GWAS SNPs with p-value < {_thold} found at region {block} at {df_paths[i]}."
                            )
                        ct += 1
                    local_gwas[i] = local_gwas_tmp

                if skip:
                    continue

                if ct == n_pop:
                    # Make sure no duplicated warnings for single pop

                    if n_pop != 1:
                        log.warning(
                            f"No GWAS SNPs with p-value < {_thold} at region {block} for all popluations. Skipping."
                        )
                    else:
                        log.warning(
                            f"No GWAS SNPs with p-value < {_thold} at region {block} at {df_paths[0]}. Skipping."
                        )
                    continue
                elif ct != 0 and bool(args.all_gwas_sig):
                    log.warning(
                        (
                            f"{ct}/{n_pop} populations have no GWAS SNPs with p-value < {_thold} at region {block} ",
                            "when `all_gwas_sig parameter` is specified `True`. Skipping.",
                        )
                    )
                    continue

                # grab local reference genotype data
                local_ref = [None] * n_pop
                for i in range(n_pop):
                    local_ref_tmp = ref[i].subset_by_pos(chrom, start=start, stop=stop)
                    if len(local_ref_tmp) == 0:
                        log.warning(
                            f"No reference LD SNPs found at region {block} at {df_refs[i]}. Skipping"
                        )
                        skip = True
                        break
                    local_ref[i] = local_ref_tmp

                if skip:
                    continue

                # grab local SNP weights
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=sa_exc.SAWarning)

                    # sqlite has a limit of at most 999 parameters for queries
                    # just split up the query into chunks and append at the end
                    # it's going to be slower, but not much we can do
                    LIMIT = 999
                    snp_weights = [None] * n_pop
                    for i in range(n_pop):
                        snp_weights_tmp = []
                        for gwas_chunk in np.array_split(
                            local_gwas[i], np.ceil(len(local_gwas[i]) / LIMIT)
                        ):
                            snp_weights_tmp.append(
                                pd.read_sql(
                                    session[i]
                                    .query(Weight, Model, MolecularFeature, RefPanel)
                                    .filter(Weight.snp.in_(gwas_chunk.SNP.values))
                                    .join(Model, Model.id == Weight.model_id)
                                    .join(
                                        MolecularFeature,
                                        MolecularFeature.id == Model.mol_id,
                                    )
                                    .join(RefPanel, RefPanel.id == Model.ref_id)
                                    .statement,
                                    con=session[i].connection(),
                                )
                            )

                        snp_weights_tmp = pd.concat(snp_weights_tmp, ignore_index=True)
                        # if there are no SNPs found in this region or none of the SNPs overlap the GWAS skip it
                        if len(snp_weights_tmp) == 0:
                            log.warning(
                                f"No overlapping weights at region {block} at {df_dbs[i]}. Skipping"
                            )
                            skip = True
                            break
                        snp_weights[i] = snp_weights_tmp
                if skip:
                    continue

                tissue_pr_gene = True if args.tissue is not None else False
                if args.tissue is not None:
                    for i in range(n_pop):
                        log.info(
                            (
                                f"Prioritizing genes by {args.tissue} tissue then on predictive performance ",
                                f"at region {local_ref[i]} of {df_refs[i]}.",
                            )
                        )
                        snp_weights[i] = get_tissue_prioritized_genes(
                            snp_weights[i],
                            args.tissue,
                            session[i],
                            metric="cv.R2",
                            strict=args.strict_tissue,
                        )

                # fine-map, my dudes
                result = pyfocus.fine_map(
                    local_gwas,
                    snp_weights,
                    local_ref,
                    block,
                    intercept=args.intercept,
                    max_genes=args.max_genes,
                    ridge=args.ridge_term,
                    prior_prob=prior_prob,
                    credible_level=args.credible_level,
                    plot=args.plot,
                    min_r2pred=args.min_r2pred,
                    max_impute=args.max_impute,
                    tissue_pr_gene=tissue_pr_gene,
                    trait=args.trait,
                )

                # fine-map can break and return early if there are issues so check for non-None result
                if result is not None:
                    if args.plot:
                        result, plots = result
                        for i in range(n_pop):
                            for j, plot in enumerate(plots[i]):
                                fig_name = f"{args.output}.chr{chrom}.{int(start)}.{int(stop)}.pop{i+1}.pdf"
                                plot.savefig(fig_name, format="pdf", dpi=600)
                    pyfocus.write_output(result, output, append=written)
                    written = True

    except Exception as err:
        log.error(err)
    finally:
        log.info(
            "Finished TWAS & fine-mapping. Thanks for using FOCUS, and have a nice day!"
        )

    return 0


def get_tissue_prioritized_genes(
    snp_weights, tissue, session, metric="cv.R2", strict=False
):
    """
    Reduce the set of models to a unique set of genes. If a gene has multiple models, the function first tries
    to select the relevant-tissue (i.e. `tissue`) model first. If multiple models are in the relevant `tissue`
    then the model with the best `metric` is selected. If no model is found for a gene in the relevant `tissue`
    then the model with the best `metric` in a proxy tissue is selected.

    :param snp_weights: pandas.DataFrame snp_weights found from query
    :param tissue: str the relevant tissue to prioritize
    :param session: sqlalchemy.Session the database session
    :param metric: str the metric to select models in proxy tissues,
        or when there are multiple models in the relevant tissue
    :param strict: bool whether to use strict-matching (i.e. exact match) for tissue name in the database. When strict
    is False models only need to have case-insensitive `tissue` as a substring in the database entry (default False).

    :return: pandas.DataFrame tissue-prioritized snp_weights for downstream TWAS + fine-mapping
    """
    log = logging.getLogger(pyfocus.LOG)
    keep = dict()
    score = dict()
    tissue_type = dict()

    mids = snp_weights.model_id.unique().astype(object)

    attrs = pd.read_sql(
        session.query(pyfocus.ModelAttribute)
        .filter(pyfocus.ModelAttribute.model_id.in_(mids))
        .statement,
        con=session.connection(),
    )

    # "ens_tx_id" is likely to be None which breaks this
    # just use gene-id for now until Tx models become fully incorporated
    for data, model in snp_weights.groupby(
        ["ens_gene_id", "tissue", "inference", "ref_name"]
    ):
        eid, mtissue, inference, ref_name = data

        local_mid = model.model_id.unique()[0]
        attr = attrs[attrs.model_id == local_mid]

        if metric not in attr.attr_name.values:
            log.warning(
                f"Gene {eid} does not have {metric} and cannot evaluate performance. Skipping."
            )
            continue

        # if there are multiple models for a gene we need to select the prioritized tissue first
        # and if it isn't represented select the model with best metric (e.g., cv.R2, h2g, etc.)
        metric_value = attr[attr.attr_name == metric]["value"].values[0]

        if pd.isna(metric_value):
            log.warning(
                f"Gene {eid} havs NA {metric} and cannot evaluate performance. Skipping."
            )
            continue

        if eid not in keep:
            keep[eid] = model
            score[eid] = metric_value
            tissue_type[eid] = mtissue
            continue

        # we can probably replace the regex with just the 'in' statement and a lowercase call at some point...
        if re.match(".*" + tissue + ".*", mtissue, flags=re.IGNORECASE) or (
            strict and tissue == mtissue
        ):
            # we have a match on preferred tissue
            if re.match(
                ".*" + tissue + ".*", tissue_type[eid], flags=re.IGNORECASE
            ) or (strict and tissue == tissue_type[eid]):
                if metric_value > score[eid]:
                    # current model is from the same tissue and has a better predictive value
                    keep[eid] = model
                    score[eid] = metric_value
                    tissue_type[eid] = mtissue
            else:
                # current model is from the preferred tissue and
                # should replace non-preferred model tissue previously stored
                keep[eid] = model
                score[eid] = metric_value
                tissue_type[eid] = mtissue
        else:
            # current model is non-preferred tissue
            if re.match(
                ".*" + tissue + ".*", tissue_type[eid], flags=re.IGNORECASE
            ) or (strict and tissue == tissue_type[eid]):
                # stored model was from preferred tissue; skip
                continue

            if metric_value > score[eid]:
                # current model is not from non-preferred tissue, but is better than the previously stored
                # non-preferred tissue model
                keep[eid] = model
                score[eid] = metric_value
                tissue_type[eid] = mtissue

    if len(keep.values()) != 0:
        snp_weights = pd.concat(keep.values())
    else:
        raise ValueError(
            "No genes have valid weights. It might be the case that database's evaluation metrics have NA value."
        )

    return snp_weights


def build_weights(args):
    log = logging.getLogger(pyfocus.LOG)
    try:
        log.info("Preparing genotype data")
        ref_panel = pyfocus.ExprRef.from_plink(args.genotype)

        log.info("Preparing phenotype data")
        ref_panel.parse_pheno(args.pheno)

        log.info("Preparing covariate data")
        ref_panel.parse_covar(args.covar)

        log.info("Preparing expression meta-data")
        ref_panel.parse_gene_info(args.info)

        log.info("Preparing weight database")
        session = pyfocus.load_db(f"{args.output}.db")

        db_ref_panel = pyfocus.RefPanel(
            ref_name=args.name, tissue=args.tissue, assay=args.assay
        )

        for train_data in ref_panel:
            # here is where we will iterate over genes, train models, and add to database
            y, X, G, snp_info, gene_info = train_data

            # fit predictive model using specified method
            log.info(f"Performing {gene_info['geneid']} model inference")
            result = pyfocus.train_model(
                y, X, G, args.method, args.include_ses, args.p_threshold
            )

            if result is None:
                continue

            weights, ses, attrs = result

            # build database object and commit
            model = pyfocus.build_model(
                gene_info, snp_info, db_ref_panel, weights, ses, attrs, args.method
            )
            session.add(model)
            try:
                session.commit()
            except Exception:
                session.rollback()
                raise

    except Exception as err:
        log.error(err)
    finally:
        session.close()
        log.info("Finished building prediction models")

    return 0


def import_weights(args):
    log = logging.getLogger(pyfocus.LOG)
    try:
        log.info("Preparing weight database")
        session = pyfocus.load_db(f"{args.output}.db")

        if args.type == "predixcan":
            pyfocus.import_predixcan(
                args.path,
                args.name,
                args.tissue,
                args.assay,
                args.predixcan_method,
                session,
            )
        else:
            pyfocus.import_fusion(
                args.path,
                args.name,
                args.tissue,
                args.assay,
                args.use_ens_id,
                args.from_gencode,
                args.rsid_table,
                session,
            )

    except Exception as err:
        log.error(err)
    finally:
        session.close()
        log.info("Finished importing prediction models")

    return 0


def build_munge_parser(subp):
    munp = subp.add_parser(
        "munge",
        description="Munge summary statistics input to conform to FOCUS requirements",
    )
    munp.add_argument("sumstats", help="Input filename.")
    munp.add_argument(
        "--N",
        default=None,
        type=float,
        help="Sample size If this option is not set, will try to infer the sample "
        "size from the input file. If the input file contains a sample size "
        "column, and this flag is set, the argument to this flag has priority.",
    )
    munp.add_argument(
        "--N-cas",
        default=None,
        type=float,
        help="Number of cases. If this option is not set, will try to infer the number "
        "of cases from the input file. If the input file contains a number of cases "
        "column, and this flag is set, the argument to this flag has priority.",
    )
    munp.add_argument(
        "--N-con",
        default=None,
        type=float,
        help="Number of controls. If this option is not set, will try to infer the number "
        "of controls from the input file. If the input file contains a number of controls "
        "column, and this flag is set, the argument to this flag has priority.",
    )
    munp.add_argument("--info-min", default=0.9, type=float, help="Minimum INFO score.")
    munp.add_argument("--maf-min", default=0.01, type=float, help="Minimum MAF.")
    munp.add_argument(
        "--n-min",
        default=None,
        type=float,
        help="Minimum N (sample size). Default is (90th percentile N) / 2.",
    )
    munp.add_argument("--chunksize", default=5e6, type=int, help="Chunksize.")

    # optional args to specify column names
    munp.add_argument(
        "--snp",
        default=None,
        type=str,
        help="Name of SNP column (if not a name that pyfocus understands). NB: case insensitive.",
    )
    munp.add_argument(
        "--N-col",
        default=None,
        type=str,
        help="Name of N column (if not a name that pyfocus understands). NB: case insensitive.",
    )
    munp.add_argument(
        "--N-cas-col",
        default=None,
        type=str,
        help="Name of N column (if not a name that pyfocus understands). NB: case insensitive.",
    )
    munp.add_argument(
        "--N-con-col",
        default=None,
        type=str,
        help="Name of N column (if not a name that pyfocus understands). NB: case insensitive.",
    )
    munp.add_argument(
        "--a1",
        default=None,
        type=str,
        help="Name of A1 column (if not a name that pyfocus understands). NB: case insensitive.",
    )
    munp.add_argument(
        "--a2",
        default=None,
        type=str,
        help="Name of A2 column (if not a name that pyfocus understands). NB: case insensitive.",
    )
    munp.add_argument(
        "--p",
        default=None,
        type=str,
        help="Name of p-value column (if not a name that pyfocus understands). NB: case insensitive.",
    )
    munp.add_argument(
        "--frq",
        default=None,
        type=str,
        help="Name of FRQ or MAF column (if not a name that pyfocus understands). NB: case insensitive.",
    )
    munp.add_argument(
        "--signed-sumstats",
        default=None,
        type=str,
        help="Name of signed sumstat column, comma null value (e.g., Z,0 or OR,1). NB: case insensitive.",
    )
    munp.add_argument(
        "--info",
        default=None,
        type=str,
        help="Name of INFO column (if not a name that pyfocus understands). NB: case insensitive.",
    )
    munp.add_argument(
        "--info-list",
        default=None,
        type=str,
        help="Comma-separated list of INFO columns. Will filter on the mean. NB: case insensitive.",
    )
    munp.add_argument(
        "--nstudy",
        default=None,
        type=str,
        help="Name of NSTUDY column (if not a name that pyfocus understands). NB: case insensitive.",
    )
    munp.add_argument(
        "--nstudy-min",
        default=None,
        type=float,
        help="Minimum # of studies. Default is to remove everything below the max, unless there is an N column,"
        " in which case do nothing.",
    )
    munp.add_argument(
        "--ignore",
        default=None,
        type=str,
        help="Comma-separated list of column names to ignore.",
    )
    munp.add_argument(
        "--a1-inc",
        default=False,
        action="store_true",
        help="A1 is the increasing allele.",
    )
    munp.add_argument(
        "--keep-maf",
        default=False,
        action="store_true",
        help="Keep the MAF column (if one exists).",
    )

    # misc options
    munp.add_argument(
        "-q",
        "--quiet",
        default=False,
        action="store_true",
        help="Do not print anything to stdout.",
    )
    munp.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Verbose logging. Includes debug info.",
    )
    munp.add_argument("-o", "--output", default="FOCUS", help="Prefix for output data.")

    return munp


def build_focus_parser(subp):
    # add imputation parser
    fmp = subp.add_parser(
        "finemap",
        description="Perform TWAS and fine-map regional results using one population or multi-ancestry populations.",
    )

    # main arguments
    fmp.add_argument(
        "gwas",
        help="GWAS summary data. Supports gzip and bz2 compression. Use semicolon to separete populations.",
    )
    fmp.add_argument(
        "ref",
        help="Path to reference panel PLINK data. Use semicolon to separete populations.",
    )
    fmp.add_argument(
        "weights",
        help="Path to weights database. Use semicolon to separete populations.",
    )
    fmp.add_argument(
        "--locations",
        default=None,
        help=(
            "Path to a BED file containing windows (e.g., CHR START STOP) to fine-map over. ",
            "Start and stop values may contain kb/mb modifiers. ",
            "Or use default independent regions by specifying ",
            "'37:EUR', '37:AFR', '37:EAS', '37:EUR-AFR', '37:EUR-EAS', '37:EAS-AFR', '37:EUR-EAS-AFR', or ",
            "'38:'-prefix regions.",
        ),
    )
    fmp.add_argument("--trait", default="trait", help="Trait name for fine-mapping.")

    # fine-map location options
    fmp.add_argument(
        "--chr", default=None, help="Perform imputation for specific chromosome."
    )
    fmp.add_argument(
        "--start",
        default=None,
        help=(
            "Perform imputation starting at specific location (in base pairs). ",
            "Accepts kb/mb modifiers. Requires --chr to be specified.",
        ),
    )
    fmp.add_argument(
        "--stop",
        default=None,
        help=(
            "Perform imputation until at specific location (in base pairs). ",
            "Accepts kb/mb modifiers. Requires --chr to be specified.",
        ),
    )

    # fine-map general options
    fmp.add_argument(
        "--all-gwas-sig",
        default=False,
        help=(
            "Boolean indicator for whether fine-mapping regions that contains GWAS signal for all population; ",
            "False means GWAS signal for at least one population.",
        ),
    )
    fmp.add_argument(
        "--tissue",
        help=(
            "Name of tissue for tissue-prioritized fine-mapping. ",
            "Relaxed matching. E.g., 'adipose' matches 'Adipose_subcutaneous'.",
        ),
    )
    fmp.add_argument(
        "--strict-tissue",
        action="store_true",
        help=(
            "Use strict-matching of relevant tissue rather than relaxed match. ",
            "E.g., 'adipose' does not match 'Adipose_subcutaneous'.",
        ),
    )
    fmp.add_argument(
        "--p-threshold",
        default=5e-8,
        type=float,
        help="Minimum GWAS p-value required to perform TWAS fine-mapping.",
    )
    fmp.add_argument(
        "--ridge-term",
        default=0.1,
        type=float,
        help="Diagonal adjustment for linkage-disequilibrium (LD) estimate.",
    )
    fmp.add_argument(
        "--intercept",
        action="store_true",
        default=False,
        help="Whether to include an intercept term in the model.",
    )
    fmp.add_argument(
        "--max-genes",
        type=parse_pint,
        default=3,
        help="Maximum number of genes that can be causal.",
    )
    fmp.add_argument(
        "--prior-prob",
        default="gencode38",
        help=(
            "Type names of prior probability for a gene to be causal. ",
            "'gencode37': use one over the number of all genes in the region based on gencode v37 ",
            "(to use your own file, specify the path instead). 'gencode38': use one over the number of all genes in ",
            "the region based on gencode v38. ",
            "'numeric': use a numeric number as fixed probability, just directly specify it e.g. 1e-3. ",
        ),
    )
    fmp.add_argument(
        "--credible-level",
        type=parse_prob,
        default=0.9,
        help="Probability value to determine the credible gene set.",
    )
    fmp.add_argument(
        "--min-r2pred",
        type=float,
        default=0.7,
        help="Minimum average LD-based imputation accuracy allowed for expression weight SNP Z-scores.",
    )
    fmp.add_argument(
        "--max-impute",
        type=float,
        default=0.5,
        help="Maximum fraction of SNPs allowed to be missing per gene, and will be imputed using LD.",
    )

    # plotting options
    fmp.add_argument(
        "-p",
        "--plot",
        action="store_true",
        default=False,
        help="Generate fine-mapping plots.",
    )
    # misc options
    fmp.add_argument(
        "-q",
        "--quiet",
        default=False,
        action="store_true",
        help="Do not print anything to stdout.",
    )
    fmp.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Verbose logging. Includes debug info.",
    )
    fmp.add_argument("-o", "--output", default="FOCUS", help="Prefix for output data.")

    return fmp


def build_import_weights_parser(subp):
    # add weight-building parser
    impt = subp.add_parser(
        "import", description="Import weights trained using FUSION or PrediXcan."
    )

    # main arguments
    impt.add_argument("path", help="Path to FUSION/PrediXcan database")
    impt.add_argument(
        "type",
        choices=["fusion", "predixcan"],
        type=lambda s: s.lower(),
        help="Datatype to import (e.g., FUSION or PrediXcan)",
    )

    # technology / experiment options
    impt.add_argument(
        "--name",
        default="",
        type=lambda s: s.lower(),
        help="Name for expression reference panel (e.g., GTEx). Case-invariant",
    )
    impt.add_argument(
        "--tissue",
        default="",
        type=lambda s: s.lower(),
        help="Tissue type assayed for expression (e.g., liver). Case-invariant",
    )
    impt.add_argument(
        "--assay",
        choices=["", "rnaseq", "array"],
        default="",
        type=lambda s: s.lower(),
        help="Technology used to measure expression levels (e.g., rnaseq). Case-invariant",
    )

    impt.add_argument(
        "--use-ens-id",
        default=False,
        action="store_true",
        help="Query on ENSEMBL gene ID rather than HGNC gene symbol.",
    )
    impt.add_argument(
        "--from-gencode",
        default=False,
        action="store_true",
        help="Query on GENCODE gene ID rather than HGNC gene symbol. Must be used with --use-ens-id.",
    )
    impt.add_argument(
        "--rsid-table",
        default=None,
        help="Path to table mapping chromosome and position to SNP rsIDs.",
    )

    impt.add_argument(
        "--predixcan-method",
        default="ElasticNet",
        help="Name of the prediction method (e.g., ElasticNet) used to fit weights.",
    )

    # misc options
    impt.add_argument(
        "-q",
        "--quiet",
        default=False,
        action="store_true",
        help="Do not print anything to stdout.",
    )
    impt.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Verbose logging. Includes debug info.",
    )
    impt.add_argument(
        "-o",
        "--output",
        default="FOCUS",
        help="Prefix for output database. If database already exists imported data will be appended.",
    )

    return impt


def _main(argsv):
    # setup main parser
    argp = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subp = argp.add_subparsers(
        help="Subcommands: munge to clean up summary statistics. finemap to perform run twas & finemap."
        " import to import weights from existing databases."
    )

    # add subparsers for cli.py commands
    munp = build_munge_parser(subp)
    munp.set_defaults(func=munge)

    fmp = build_focus_parser(subp)
    fmp.set_defaults(func=run_twas)

    imtp = build_import_weights_parser(subp)
    imtp.set_defaults(func=import_weights)

    # parse arguments
    args = argp.parse_args(argsv)

    # hack to check that at least one sub-command was selected
    if not hasattr(args, "func"):
        argp.print_help()
        return 2  # command-line error

    cmd_str = get_command_string(argsv)

    v = pyfocus.VERSION
    masthead = "===================================" + os.linesep
    masthead += f"          MA-FOCUS v{v}             " + os.linesep
    masthead += "===================================" + os.linesep

    # setup logging
    log_format = "[%(asctime)s - %(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    log = logging.getLogger(pyfocus.LOG)
    if args.verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt=log_format, datefmt=date_format)

    # write to stdout unless quiet is set
    if not args.quiet:
        sys.stdout.write(masthead)
        sys.stdout.write(cmd_str)
        sys.stdout.write("Starting log..." + os.linesep)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(fmt)
        log.addHandler(stdout_handler)

    # setup log file, but write PLINK-style command first
    disk_log_stream = open(f"{args.output}.log", "w")
    disk_log_stream.write(masthead)
    disk_log_stream.write(cmd_str)
    disk_log_stream.write("Starting log..." + os.linesep)

    disk_handler = logging.StreamHandler(disk_log_stream)
    disk_handler.setFormatter(fmt)
    log.addHandler(disk_handler)

    # launch munge, impute, build-weights, import-weights, etc...
    args.func(args)

    return 0


def run_cli():
    return _main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
