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


categories = [
    'cs.CL',
    'cs.MS',
    'cond-mat.soft',
    'cs.MA',
    'cs.GL',
    'math.FA',
    'nlin.CG',
    'physics.gen-ph',
    'stat.AP',
    'math.AT',
    'hep-ph',
    'cs.NA',
    'cond-mat.quant-gas',
    'q-fin.TR',
    'physics.pop-ph',
    'physics.optics',
    'math.HO',
    'cond-mat.supr-con',
    'math.CV',
    'q-bio.MN',
    'q-fin.PM',
    'eess.AS',
    'eess.SP',
    'physics.geo-ph',
    'econ.EM',
    'math.QA',
    'physics.comp-ph',
    'cs.DS',
    'physics.bio-ph',
    'physics.flu-dyn',
    'math.CO',
    'cs.SD',
    'physics.acc-ph',
    'math.LO',
    'cs.DM',
    'nucl-th',
    'cs.SI',
    'q-bio.PE',
    'math.PR',
    'cond-mat',
    'q-fin.PR',
    'cond-mat.other',
    'physics.app-ph',
    'math.DG',
    'math.GT',
    'q-bio.NC',
    'hep-th',
    'q-fin.ST',
    'physics.atom-ph',
    'physics.space-ph',
    'cs.GR',
    'q-fin.GN',
    'cs.CV',
    'physics.hist-ph',
    'math.GM',
    'gr-qc',
    'cond-mat.dis-nn',
    'cs.DL',
    'cs.OS',
    'physics.plasm-ph',
    'q-fin.EC',
    'astro-ph.IM',
    'econ.TH',
    'nlin.CD',
    'physics.ins-det',
    'eess.IV',
    'physics.soc-ph',
    'cs.IR',
    'cond-mat.mtrl-sci',
    'cs.CG',
    'math.OC',
    'nlin.SI',
    'physics.class-ph',
    'math.GR',
    'cs.AR',
    'math.AG',
    'q-fin.RM',
    'math.NT',
    'astro-ph.CO',
    'cs.SC',
    'astro-ph.HE',
    'nlin.PS',
    'math.SP',
    'math.CT',
    'math.MG',
    'astro-ph.SR',
    'stat.ML',
    'q-fin.MF',
    'cs.CR',
    'astro-ph.EP',
    'physics.data-an',
    'cs.HC',
    'q-bio.CB',
    'cs.PF',
    'cs.AI',
    'cs.NE',
    'cs.RO',
    'math.AC',
    'math.OA',
    'cs.CY',
    'nlin.AO',
    'cs.SE',
    'q-bio.SC',
    'cs.CE',
    'cond-mat.str-el',
    'cs.LG',
    'cs.OH',
    'physics.chem-ph',
    'cs.DC',
    'q-fin.CP',
    'q-bio.TO',
    'quant-ph',
    'math.SG',
    'astro-ph.GA',
    'stat.OT',
    'cond-mat.mes-hall',
    'cond-mat.stat-mech',
    'cs.GT',
    'physics.atm-clus',
    'q-bio.GN',
    'cs.MM',
    'q-bio.OT',
    'nucl-ex',
    'stat.ME',
    'cs.ET',
    'math.GN',
    'q-bio.QM',
    'math.KT',
    'cs.LO',
    'physics.ed-ph',
    'hep-lat',
    'astro-ph',
    'math.RT',
    'math.DS',
    'math.RA',
    'physics.ao-ph',
    'math.NA',
    'cs.FL',
    'stat.CO',
    'cs.SY',
    'cs.PL',
    'math.CA',
    'cs.DB',
    'cs.NI',
    'hep-ex',
    'cs.CC',
    'math.AP',
    'q-bio.BM',
    'physics.med-ph'
]
