"""Automatic-K software demo; no reference K or labels are supplied to the model."""
import argparse
from pathlib import Path
import numpy as np
from scmetabasin import scMetaBasin, save_representations_npz
from scmetabasin.result import save_scmetabasin_results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('demo_auto'))
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Output directory already exists')
    args.output.mkdir(parents=True)
    rng=np.random.default_rng(20261024)
    x=np.vstack([rng.normal(center,.4,size=(50,2)) for center in [[-6.,-3.],[6.,-3.],[0.,7.]]])
    x=x[rng.permutation(len(x))]
    bank=dict(view_a=x,view_b=x*1.7+.3,view_c=x+rng.normal(0,.025,x.shape))
    ids=np.array([f'demo_cell_{i}' for i in range(len(x))])
    input_path=args.output/'representations.npz'
    save_representations_npz(input_path,bank,cell_ids=ids)
    model=scMetaBasin()
    labels=model.fit_predict(bank)
    if labels is None:
        raise RuntimeError(f'No recommended split: {model.status_}')
    save_scmetabasin_results(model,args.output/'results',cell_ids=ids,input_path=input_path)
    print(f'Estimated K={model.estimated_K_}; actual K={model.realized_K_}; status={model.status_}')
    print('Synthetic software demonstration only; not a biological validation.')


if __name__=='__main__':
    main()
