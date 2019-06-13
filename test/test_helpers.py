import os
import sys

sys.path.append(os.path.abspath('./'))
from src import helpers
from sklearn.linear_model import LogisticRegression
import pickle


def test_get_files():
    files = ['sample1.txt', 'sample2.txt', 'sample3.csv']
    assert sorted([os.path.basename(f) for f in helpers.get_files('test/assets')]) == files


def test_get_files_filtered():
    files = ['sample3.csv']
    assert [os.path.basename(f) for f in helpers.get_files('test/assets', '*.csv')] == files


def test_get_newest_model():
    # Create two dummy models
    mod1 = LogisticRegression(fit_intercept=True)
    mod2 = LogisticRegression(fit_intercept=False)

    # Write models to files for finding newest of two
    with open('test/assets/model1.pkl', 'wb') as f:
        pickle.dump(mod1, f)
    with open('test/assets/model2.pkl', 'wb') as f:
        pickle.dump(mod2, f)
    files = ['test/assets/model1.pkl', 'test/assets/model2.pkl']

    # model retrieved should be model2 as it was most recent saved
    model = helpers.get_newest_model(files)

    # Cleanup
    os.remove('test/assets/model1.pkl')
    os.remove('test/assets/model2.pkl')

    # Compare models by their params (mod2 has different params than mod1)
    assert model.get_params() == mod2.get_params()
