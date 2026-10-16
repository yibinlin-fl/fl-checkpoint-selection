import csv
import tempfile
from pathlib import Path
import pytest
from p1.selection import read_selection_scores, compute_selection
from p1.audit import audit_regret


def mk(scenario,rnd,cid,acc,ce=1.0,n=20):
    return dict(scenario=scenario,round=rnd,client_id=cid,n=n,noisy_acc=acc,noisy_ce=ce)


def test_selection_and_tie_break():
    scores = [mk('heterogeneous',1,0,0.5), mk('heterogeneous',1,1,0.5),
              mk('heterogeneous',5,0,0.5), mk('heterogeneous',5,1,0.5)]
    selected = compute_selection(scores)
    assert next(x for x in selected if x['selector']=='last')['selected_round'] == 5
    assert next(x for x in selected if x['selector']=='weighted_noisy_acc')['selected_round'] == 1


def test_selector_schema_blocks_oracle_leaks(tmp_path):
    path = tmp_path / 'scores.csv'
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=['round','client_id','scenario','n','noisy_acc','noisy_ce','test_acc'])
        w.writeheader()
        w.writerow(dict(round=1,client_id=0,scenario='clean',n=10,noisy_acc=0.5,noisy_ce=1.0,test_acc=0.9))
    with pytest.raises(ValueError,match='schema'):
        read_selection_scores(path)


def test_selector_refuses_clean_scenario(tmp_path):
    path = tmp_path / "scores.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=['round','client_id','scenario','n','noisy_acc','noisy_ce'])
        w.writeheader()
        w.writerow(mk('clean', 1, 0, .9))
    with pytest.raises(ValueError, match="forbidden"):
        read_selection_scores(path)


def test_oracle_regret_after_selection():
    chosen=[{'scenario':'heterogeneous','selector':'last','selected_round':5}]
    oracle=[{'round':1,'test_accuracy':0.9},{'round':5,'test_accuracy':0.7}]
    result=audit_regret(chosen, oracle)
    assert result[0]['selection_regret_pp'] == pytest.approx(20.0)
