import pytest
import numpy as np
import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.INFO, filename='tests/logfile.log', filemode='w')

from editable_proj import NoContextData

# check size of ests and outcomes
# check if ests are clipped
def test_size_and_range():
    data_dir = 'data/no_context'
    eps = 1e-6
    loader = NoContextData(data_dir=data_dir, eps=eps)
    df, outcomes = loader.get_data()
    logging.info("ests shape %s", df.shape)
    assert df.shape[0] == outcomes.shape[0]
    logging.info("ests have correct shape")
    # mask nans
    mask = np.isnan(df)
    # check if non nan values are clipped
    assert np.all(np.logical_or(df >= eps, mask))
    assert np.all(np.logical_or(df <= 1 - eps, mask))
    logging.info("ests are clipped")
    assert np.all(np.isin(outcomes, [0, 1]))
    logging.info("outcomes are binary")