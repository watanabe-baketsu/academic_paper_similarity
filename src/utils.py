import json

from datasets import Dataset, DatasetDict


def read_dataset(file_path: str) -> DatasetDict:
    """
    file_path: str
        Path to the dataset file
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    dataset = DatasetDict({
        "training": Dataset.from_list(data["training"]),
        "validation": Dataset.from_list(data["validation"]),
        "testing": Dataset.from_list(data["testing"])
    })

    return dataset


categories = ['cs.MA',
              'cond-mat.mes-hall',
              'math.CV',
              'q-bio.GN',
              'bayes-an',
              'math.AG',
              'hep-th',
              'math.SG',
              'cmp-lg',
              'physics.acc-ph',
              'astro-ph.HE',
              'cond-mat.str-el',
              'dg-ga',
              'econ.EM',
              'cs.NA',
              'physics.hist-ph',
              'physics.soc-ph',
              'nlin.CD',
              'cs.MM',
              'cs.OS',
              'nlin.PS',
              'cs.RO',
              'hep-ex',
              'cs.DS',
              'math.LO',
              'q-fin.ST',
              'cs.LO',
              'cs.SC',
              'astro-ph.SR',
              'physics.geo-ph',
              'physics.data-an',
              'nlin.AO',
              'eess.AS',
              'cs.LG',
              'physics.flu-dyn',
              'solv-int',
              'chem-ph',
              'math.IT',
              'cs.ET',
              'astro-ph',
              'math-ph',
              'cs.CL',
              'math.GM',
              'cs.PF',
              'cs.GL',
              'physics.app-ph',
              'physics.bio-ph',
              'math.FA',
              'patt-sol',
              'math.QA',
              'physics.pop-ph',
              'cs.DC',
              'supr-con',
              'q-fin.MF',
              'math.GT',
              'nlin.SI',
              'atom-ph',
              'physics.comp-ph',
              'nucl-th',
              'math.DS',
              'q-bio',
              'cs.AR',
              'nlin.CG',
              'stat.ME',
              'math.RT',
              'ao-sci',
              'cs.DL',
              'q-fin.RM',
              'cs.NI',
              'q-alg',
              'physics.med-ph',
              'q-bio.BM',
              'cs.CC',
              'astro-ph.EP',
              'physics.ao-ph',
              'math.CO',
              'cs.PL',
              'q-fin.TR',
              'physics.class-ph',
              'physics.atm-clus',
              'cs.CE',
              'cs.SD',
              'acc-phys',
              'cs.NE',
              'cs.CG',
              'cs.FL',
              'cs.DB',
              'chao-dyn',
              'astro-ph.IM',
              'physics.ed-ph',
              'cs.SE',
              'cs.CY',
              'comp-gas',
              'q-bio.MN',
              'cs.MS',
              'cs.GT',
              'cond-mat.other',
              'quant-ph',
              'math.CT',
              'physics.optics',
              'q-fin.PM',
              'math.KT',
              'math.SP',
              'econ.GN',
              'cs.GR',
              'cond-mat.quant-gas',
              'cs.SY',
              'cs.DM',
              'cs.AI',
              'eess.IV',
              'astro-ph.GA',
              'q-fin.EC',
              'q-bio.OT',
              'q-bio.NC',
              'math.GN',
              'cond-mat.stat-mech',
              'astro-ph.CO',
              'physics.ins-det',
              'math.AC',
              'q-bio.CB',
              'alg-geom',
              'stat.AP',
              'hep-lat',
              'q-fin.CP',
              'cs.CV',
              'cs.HC',
              'plasm-ph',
              'econ.TH',
              'physics.gen-ph',
              'physics.space-ph',
              'gr-qc',
              'q-bio.PE',
              'math.MP',
              'math.AP',
              'math.AT',
              'cs.CR',
              'math.CA',
              'math.HO',
              'q-fin.PR',
              'math.OA',
              'cs.SI',
              'cond-mat.supr-con',
              'physics.plasm-ph',
              'cond-mat.dis-nn',
              'hep-ph',
              'eess.SY',
              'physics.chem-ph',
              'adap-org',
              'math.NT',
              'nucl-ex',
              'funct-an',
              'cond-mat.soft',
              'math.OC',
              'math.MG',
              'math.GR',
              'cs.IT',
              'stat.OT',
              'math.RA',
              'math.DG',
              'stat.ML',
              'cs.IR',
              'cond-mat',
              'cond-mat.mtrl-sci',
              'physics.atom-ph',
              'stat.CO',
              'q-bio.SC',
              'stat.TH',
              'mtrl-th',
              'q-fin.GN',
              'math.NA',
              'math.ST',
              'math.PR',
              'q-bio.TO',
              'q-bio.QM',
              'eess.SP',
              'cs.OH']
