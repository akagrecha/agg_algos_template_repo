from pathlib import Path

import pandas as pd

class NoContextData:
    def __init__(self, data_dir: Path, 
                 eps: float=1e-6) -> None:
        self.data_dir = data_dir
        self.eps = eps

    def get_data(self):
        df = pd.read_csv(self.data_dir/Path(f'data.csv'))
        outcomes = df['outcome'].values
        df = df.drop(columns=['outcome'])
        # clip model preds to 1-eps and eps
        df = df.clip(lower=self.eps, upper=1 - self.eps)
        return df, outcomes
