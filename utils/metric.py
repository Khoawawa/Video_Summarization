from collections import defaultdict
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

def calculate_metrics(preds, tgts):
    '''
    '''
    assert len(preds) == len(tgts), "preds and targets must have same length"
    
    gts = defaultdict(list) 
    res = defaultdict(list)
    for i, (p, refs) in enumerate(zip(preds, tgts)):
        res[i] = [p]
        gts[i] = refs
    
    # BLEU (1-4)
    bleu_scorer = Bleu(4)
    bleu_score, _ = bleu_scorer.compute_score(gts, res)
    # CIDEr
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    
    return {
        "BLEU-1": bleu_score[0],
        "BLEU-2": bleu_score[1],
        "BLEU-3": bleu_score[2],
        "BLEU-4": bleu_score[3],
        "CIDEr": cider_score
    }
    