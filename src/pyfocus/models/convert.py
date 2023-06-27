import logging

import pandas as pd

from sqlalchemy import create_engine

import pyfocus as pf


__all__ = ["import_fusion", "import_predixcan"]

# TODO: implement exporting to predixcan/fusion

COUNT = 250


def import_fusion(
    path, name, tissue, assay, use_ens_id, from_gencode, rsid_table, session
):
    """
    Import weights from a FUSION Rdata into the FOCUS db.

    :param path:  string path to the PrediXcan db
    :param tissue: str name of the tissue
    :param assay: technology assay to measure abundance
    :param use_ens_id: bool, query on ensembl ids instead of hgnc gene symbols
    :param from_gencode: bool, convert gencode_ids to ens_ids. Only works with use_ens_id == True
    :param rsid_table: str, name of the rsid table to use
    :param session: sqlalchemy.Session object for the FOCUS db

    :return:  None
    """
    log = logging.getLogger(pf.LOG)

    import os
    import re
    import warnings

    from collections import defaultdict

    import numpy as np

    try:
        import mygene

        # suppress warnings about R build
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import rpy2.robjects as robj
    except ImportError:
        log.error("Import submodule requires mygene and rpy2 to be installed.")
        raise

    log.info(f"Starting import from FUSION database {path}")
    db_ref_panel = pf.models.RefPanel(ref_name=name, tissue=tissue, assay=assay)
    ses = None

    load_func = robj.r["load"]
    local_dir = os.path.dirname(os.path.abspath(path))

    # we need this to grab Ensembl IDs for genes
    mg = mygene.MyGeneInfo()

    # WGT ID CHR P0 P1
    fusion_db = pd.read_csv(path, delim_whitespace=True)
    if use_ens_id and from_gencode:
        # likely importing from GTEx. Pull underlying ensemble id from gencode id
        genes = fusion_db.ID.apply(lambda x: re.sub("(.*)\.\\d+", "\\1", x))
    else:
        genes = fusion_db.ID.values

    # we need to do batch queries in order to not get throttled by the mygene servers

    log.info("Querying mygene servers for gene annotations")
    if use_ens_id:
        results = mg.querymany(
            genes,
            scopes="ensembl.gene",
            verbose=False,
            fields="ensembl.gene,genomic_pos,symbol,ensembl.type_of_gene,alias",
            species="human",
        )
    else:
        # a lot of older gene expression datasets have outdated symbols. include
        # alias here to help match them up with ens-id
        results = mg.querymany(
            genes,
            scopes="symbol,alias",
            verbose=False,
            fields="ensembl.gene,genomic_pos,symbol,ensembl.type_of_gene,alias",
            species="human",
        )

    res_map = defaultdict(list)
    for result in results:
        res_map[result["query"]].append(result)

    # load rsid_table if we have one
    dict_rsid_table = None
    if rsid_table is not None:
        log.info(f"Loading rsid table {rsid_table}")
        df_rsid_table = pd.read_csv(rsid_table, sep="\t")
        dict_rsid_table = {
            (str(chrom), pos): snp
            for chrom, pos, snp in zip(
                df_rsid_table["CHR"].values,
                df_rsid_table["POS"].values,
                df_rsid_table["SNP"].values,
            )
        }

    count = 0
    log.info("Starting individual model conversion")

    # check whether fusion_db has column DIR
    DIR_IN_HEADER = "DIR" in fusion_db.columns

    for rdx, row in fusion_db.iterrows():
        if DIR_IN_HEADER:
            wgt_dir = row.DIR
        else:
            wgt_dir = local_dir

        wgt_name, g_name, chrom, txstart, txstop = (
            row.WGT,
            row.ID,
            row.CHR,
            row.P0,
            row.P1,
        )

        # METSIM.ADIPOSE.RNASEQ/METSIM.LINC00115.wgt.RDat LINC00115 1 761586 762902
        log.debug(f"Importing {wgt_name} model")

        # this call should create the following:
        # 'wgt.matrix', 'snps', 'cv.performance', 'hsq', and 'hsq.pv'
        wgt_path = f"{wgt_dir}/{wgt_name}"

        # load the Rdat data
        load_func(wgt_path)

        gene_info = dict()
        id_dict = dict()
        if use_ens_id and from_gencode:
            lookup_g_name = re.sub("(.*)\.\\d+", "\\1", g_name)
        else:
            lookup_g_name = g_name

        # hits are ordered by match quality.
        for hit in res_map[lookup_g_name]:
            if "notfound" in hit:
                continue

            if "ensembl" not in hit:
                # nothing in db
                continue

            if (
                not use_ens_id
                and hit["symbol"] != g_name
                and "alias" in hit
                and g_name not in hit["alias"]
            ):
                # not direct match
                continue

            if "genomic_pos" not in hit:
                continue

            ens = hit["ensembl"]
            pos = hit["genomic_pos"]
            if use_ens_id and "symbol" in hit:
                g_name = hit["symbol"]

            # sometimes we have multiple ENSG entries due to diff haplotypes.
            # just reduce the single-case to the multi by a singleton list
            if type(ens) is dict:
                ens = [ens]
            if type(pos) is dict:
                pos = [pos]

            # grab the type for when we match against pos
            for e_hit in ens:
                id_dict[e_hit["gene"]] = e_hit["type_of_gene"]

            for p_hit in pos:
                if not re.match("[0-9]{1,2}|X|Y", p_hit["chr"], re.IGNORECASE):
                    continue

                g_id = p_hit["ensemblgene"]
                g_type = id_dict.get(p_hit["ensemblgene"])

                if len(gene_info) == 0:
                    # grab any info if we haven't yet
                    gene_info["geneid"] = g_id
                    gene_info["type"] = g_type
                elif "protein" in g_type:
                    # prioritize protein coding and break out if we find one
                    gene_info["geneid"] = g_id
                    gene_info["type"] = g_type

        if len(gene_info) == 0:
            # we didn't get any hits from our query
            # just use the gene-name as ens-id...
            if use_ens_id:
                log.warning(
                    f"Unable to match {g_name} to Ensembl ID. Using ID for symbol"
                )
            else:
                log.warning(
                    f"Unable to match {g_name} to Ensembl ID. Using symbol for ID"
                )
            gene_info["geneid"] = g_name
            gene_info["type"] = None

        # build info on the gene
        gene_info["txid"] = None
        gene_info["name"] = g_name
        gene_info["chrom"] = chrom
        gene_info["txstart"] = txstart
        gene_info["txstop"] = txstop

        # get the multi-SNP method with the best cvR2
        methods = np.array(robj.r["cv.performance"].colnames)
        types = list(robj.r["cv.performance"].rownames)
        if "rsq" not in types:
            raise ValueError(f"No R2 value for model {path}")
        if "pval" not in types:
            raise ValueError(f"No R2 p-value for model {path}")

        # grab the actual weights
        wgts = np.array(robj.r["wgt.matrix"])

        # sometimes weights are constant or only contain NANs; drop them
        keep = np.logical_not(np.isnan(np.std(wgts, axis=0)))
        wgts = wgts.T[keep].T
        methods = methods[keep]

        rsq_idx = types.index("rsq")
        pval_idx = types.index("pval")

        values = np.array(robj.r["cv.performance"])
        v_shape = values.shape

        # is this always stored/retrieved as 2 x M ?
        if v_shape[0] > v_shape[1]:
            values = values.T

        values = values.T[keep].T

        method = None
        r2idx = 0
        top1_idx = -1
        pval = 1
        r2 = -100  # FUSION reports the generalized R2 which can be negative
        for idx, value in enumerate(values[rsq_idx]):
            if methods[idx] == "top1":
                top1_idx = idx
                continue

            if value > r2:
                r2 = value
                method = methods[idx]
                r2idx = idx
                pval = values[pval_idx, idx]

        if method is None:
            log.warning("Only predicting model is top1 eQTL.")
            r2 = values[rsq_idx][top1_idx]
            method = "top1"
            r2idx = top1_idx
            pval = values[pval_idx, idx]

        wgts = wgts.T[r2idx]

        # keep attributes
        attrs = dict()
        attrs["cv.R2"] = r2
        attrs["cv.R2.pval"] = pval

        # SNPs data frame
        # V1 V2 V3 V4 V5 V6
        # 11 rs2729762 0 77033699 G A
        snps = robj.r["snps"]
        snp_info = pd.DataFrame(
            {
                "snp": list(snps[1]),
                "chrom": [str(chrom) for chrom in snps[0]],
                "pos": list(snps[3]),
                "a1": list(snps[4]),
                "a0": list(snps[5]),
            }
        )
        if dict_rsid_table is not None:
            # map chrom, pos to rsid
            snp_info["snp"] = snp_info.apply(
                lambda row: dict_rsid_table[(row.chrom, row.pos)]
                if (row.chrom, row.pos) in dict_rsid_table
                else None,
                axis=1,
            )
            # filter out any SNPs that don't have an rsid
            keep = snp_info.snp.notnull().values
            if not np.all(keep):
                log.warning(
                    f"Unable to map {np.sum(~keep)}/{len(snp_info)} SNPs to rsids for {g_name}"
                )
            # subset to only valid SNPs
            wgts = wgts[keep]
            snp_info = snp_info[keep]

        # if we're using a sparse model there is no need to store info on zero'd SNPs
        keep = np.logical_not(np.isclose(wgts, 0))
        wgts = wgts[keep]
        snp_info = snp_info[keep]

        model = pf.models.build_model(
            gene_info, snp_info, db_ref_panel, wgts, ses, attrs, method
        )
        session.add(model)
        try:
            session.commit()
        except Exception:
            session.rollback()
            raise Exception("Failed committing to db")
        count += 1
        if count % COUNT == 0:
            log.info(f"Committed {COUNT} models to db")

    if count % COUNT != 0:
        log.info(f"Committed {count % COUNT} models to db")

    log.info(f"Finished import from FUSION database {path}")
    return


def export_fusion(path, session):
    logging.getLogger(pf.LOG)
    raise NotImplementedError("export_fusion not implemented!")
    return


def import_predixcan(path, name, tissue, assay, method, session):
    """
    Import weights from a PrediXcan db into the FOCUS db.

    :param path:  string path to the PrediXcan db
    :param name: str name of the reference panel
    :param tissue: str name of the tissue
    :param assay: technology assay to measure abundance
    :param method: the prediction model used to fit the data
    :param session: sqlalchemy.Session object for the FOCUS db

    :return:  None
    """
    log = logging.getLogger(pf.LOG)

    import os
    import re

    from collections import defaultdict

    import numpy as np

    try:
        import mygene
    except ImportError:
        log.error("Import submodule requires mygene and rpy2 to be installed.")
        raise

    if not os.path.isfile(path):
        raise ValueError(f"Cannot find database {path}")
    log.info(f"Starting import from PrediXcan database {path}")
    pred_engine = create_engine(f"sqlite:///{path}")

    weights = pd.read_sql_table("weights", pred_engine)
    extra = pd.read_sql_table("extra", pred_engine)

    def gencode2ensmble(x):
        idx = x.rfind(".")
        return x if idx == -1 else x[:idx]

    # get unique genes
    genes = weights.gene.unique()
    genes = [gencode2ensmble(g) for g in genes]

    log.info("Querying mygene servers for gene annotations")
    mg = mygene.MyGeneInfo()
    results = mg.querymany(
        genes,
        scopes="ensembl.gene",
        verbose=False,
        fields="genomic_pos_hg19,symbol,alias",
        species="human",
    )

    res_map = defaultdict(list)
    for result in results:
        res_map[result["query"]].append(result)

    db_ref_panel = pf.models.RefPanel(ref_name=name, tissue=tissue, assay=assay)
    ses = None

    count = 0
    log.info("Starting individual model conversion")
    for gid, gene in weights.groupby("gene"):
        log.debug(f"Importing gene model {gid}")
        gene_extra = extra.loc[extra.gene == gid]

        chrom = gene.varID.values[0].split("_")[0]  # grab chromosome from varID
        pos = gene.varID.map(lambda x: int(x.split("_")[1])).values  # grab basepair pos
        txstart = txstop = np.median(pos)

        g_id = gene_extra.gene.values[0]
        g_name = gene_extra.genename.values[0]
        query_id = gencode2ensmble(g_id)

        for hit in res_map[query_id]:
            if "notfound" in hit:
                continue

            if (
                hit["symbol"] != g_name
                and "alias" in hit
                and g_name not in hit["alias"]
            ):
                continue

            if "genomic_pos_hg19" not in hit:
                continue

            gpos = hit["genomic_pos_hg19"]
            if type(gpos) is dict:
                gpos = [gpos]

            for entry in gpos:
                # skip non-primary assembles. they have weird CHR entries e.g., CHR_HSCHR1_1_CTG3
                if not re.match("[0-9]{1,2}|X|Y", entry["chr"], re.IGNORECASE):
                    continue

                txstart = entry["start"]
                txstop = entry["end"]
                break

            if txstart is not None:
                # we want to use standardized Ensembl identifiers; not GENCODE modified ones...
                g_id = query_id
                break

        gene_info = dict()
        gene_info["geneid"] = g_id
        gene_info["txid"] = None
        gene_info["name"] = g_name
        gene_info["type"] = gene_extra.gene_type.values[0]
        gene_info["chrom"] = chrom
        gene_info["txstart"] = txstart
        gene_info["txstop"] = txstop

        snp_info = pd.DataFrame(
            {
                "snp": gene.rsid.values,
                "chrom": [chrom] * len(gene),
                "pos": pos,
                "a1": gene.eff_allele.values,
                "a0": gene.ref_allele.values,
            }
        )

        wgts = gene.weight.values

        attrs = dict()
        # predixcan enet uses cv_R2_avg and nested_cv_fisher_pval
        # mashr uses pred.perf.R2 and pred.perf.pval
        if "cv_R2_avg" in gene_extra:
            attrs["cv.R2"] = gene_extra["cv_R2_avg"].values[0]
            attrs["cv.R2.pval"] = gene_extra["nested_cv_fisher_pval"].values[0]
        elif "pred.perf.R2" in gene_extra:
            attrs["cv.R2"] = gene_extra["pred.perf.R2"].values[0]
            attrs["cv.R2.pval"] = gene_extra["pred.perf.pval"].values[0]

        # build model
        model = pf.models.build_model(
            gene_info, snp_info, db_ref_panel, wgts, ses, attrs, method
        )
        session.add(model)
        try:
            session.commit()
        except Exception:
            session.rollback()
            raise Exception("Failed committing to db")

        count += 1
        if count % COUNT == 0:
            log.info(f"Committed {COUNT} models to db")

    if count % COUNT != 0:
        log.info(f"Committed {count % COUNT} models to db")

    log.info(f"Finished import from PrediXcan database {path}")
    return


def export_predixcan(path, session):
    logging.getLogger(pf.LOG)
    raise NotImplementedError("export_predixcan not implemented!")
    return
