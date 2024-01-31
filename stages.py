import os
from pathlib import Path

import numpy as np
from astromodule.hp import HP, HyperParameterSet
from astromodule.io import read_table
from astromodule.pipeline import Pipeline, PipelineStage

from model import Model, get_best_trial, get_study

BROAD_BANDS = ['r', 'g', 'i', 'u', 'z']
NARROW_BANDS = ['J0378', 'J0395', 'J0410', 'J0430', 'J0515', 'J0660', 'J0861']
SPLUS_BANDS = BROAD_BANDS + NARROW_BANDS
RANDOM_STATE = 42
DATA_ROOT = Path(os.environ.get('DATA_ROOT', 'data'))
BASE_STUDY_PATH = Path('logs')
BASE_STUDY_PATH.mkdir(parents=True, exist_ok=True)



class DataPrepStage(PipelineStage):
  name = 'DataPrep'
  produces = ['features', 'targets', 'band']
  
  def __init__(self, band: str):
    self.band = band
  
  def run(self):
    df = read_table(DATA_ROOT / 'input' / f'input_{self.band}.csv')
    feature_names = [
      f'{self.band}_FWHM', f'{self.band}_3', f'{self.band}_5', 
      f'{self.band}_10', f'{self.band}_50', 'OAJ PRO REFAIRMASS', 
      'MJD-OBS', 'GAIN', #'REF_MOON_AZ', 'REF_MOON_ALT', 
      'REF_MOON_SEPARATION', 'REF_MOON_ILLUMINATION'
    ]
    features = df[feature_names].values
    target_name = f'ZP_{self.band}'
    targets = df[target_name].values
    if targets.shape[-1] == 1:
      targets = targets.ravel()
      
    self.set_output('features', features)
    self.set_output('targets', targets)
    self.set_output('band', self.band)
  
  
  
  
class BandwiseHPTunnigStage(PipelineStage):
  name = 'BandwiseHPTunning'
  requires = ['features', 'targets', 'band']
  
  def __init__(self, n_trials: float = 50, progress: bool = True):
    self.n_trials = n_trials
    self.progress = progress
  
  def run(self):
    hps = HyperParameterSet(
      # Scaler
      HP.cat('scaler', ['std', 'minmax']),
      # Meta Model
      HP.num('meta_n_estimators', low=5, high=50, step=1, dtype=int),
      HP.num('meta_subsample', low=0.2, high=0.7, log=True, dtype=float),
      HP.num('meta_min_samples_leaf', low=2, high=40, step=1, dtype=int),
      HP.num('meta_max_features', low=1, high=10, step=1, dtype=int),
      # Random Forest
      HP.cat('use_rf', [True, False]),
      HP.num('rf_n_estimators', low=10, high=500, step=1, dtype=int),
      HP.cat('rf_max_depth', [None, 5, 10, 15, 20, 30, 40, 50, 60, 100, 120]),
      HP.num('rf_min_samples_split', low=2, high=20, step=1, dtype=int),
      # Extremely Random Trees
      HP.cat('use_ert', [True, False]),
      HP.num('ert_n_estimators', low=10, high=500, step=1, dtype=int),
      HP.cat('ert_max_depth', [None, 5, 10, 15, 20, 30, 40, 50, 60, 100, 120]),
      HP.num('ert_min_samples_split', low=2, high=20, step=1, dtype=int),
      # KNN
      HP.cat('use_knn', [True, False]),
      HP.num('knn_n_neighbors', low=5, high=100, step=1, dtype=int),
      HP.cat('knn_weights', ['uniform', 'distance']),
      HP.num('knn_leaf_size', low=2, high=50, step=1, dtype=int),
      # SVR
      HP.cat('use_svr', [True, False]),
      HP.cat('svr_kernel', ['poly', 'rbf', 'sigmoid']),
      HP.num('svr_degree', low=3, high=6, step=1, dtype=int),
      HP.num('svr_C', low=0.5, high=2.0, dtype=float),
      # HP.cat('svr_shrinking', [True, False]),
      # HP.num('svr_epsilon', low=0.1, high=0.8, dtype=float),  
      # MLP
      HP.const('use_mlp', False),
      HP.num('mlp_layer_1', low=4, high=60, step=1, dtype=int),
      HP.num('mlp_layer_2', low=4, high=60, step=1, dtype=int),
      HP.cat('mlp_activation', ['relu', 'tanh']),
      HP.cat('mlp_solver', ['sgd', 'adam']),
      HP.num('mlp_alpha', low=1e-5, high=1e-3, log=True, dtype=float),
      HP.cat('mlp_learning_rate', ['invscaling', 'adaptive']),
      HP.num('mlp_learning_rate_init', low=1e-5, high=1e-2, log=True, dtype=float),
      HP.const('mlp_validation_fraction', 0.2),
      HP.const('mlp_early_stopping', True),
      HP.const('mlp_n_iter_no_change', 5),
      # Logist Regression
      HP.const('use_lr', False),
      HP.cat('lr_penality', ['l1', 'l2', 'elasticnet']),
      HP.cat('lr_dual', [True, False]),
      HP.num('lr_C', low=0.5, high=2, log=True, dtype=float),
      # General
      HP.const('n_jobs', 4),
      verbose=False,
    )
    model = Model(
      hps=hps, 
      band=self.get_output('band'),
      targets=self.get_output('targets'),
      features=self.get_output('features'), 
    )
    model.hyperparameter_search(n_trials=self.n_trials, progress=self.progress)
    
  
  
  
class LoadModelFromBestTrialStage(PipelineStage):
  name = 'LoadModelFromBestTrial'
  requires = ['band', 'features', 'targets']
  produces = ['model', 'study', 'best_trials']
  
  def run(self):
    band = self.get_output('band')
    features = self.get_output('features')
    targets = self.get_output('targets')
    model = Model.from_best_trial(band=band, features=features, targets=targets)
    study = get_study(f'{band}_ensemble')
    self.set_output('model', model)
    self.set_output('study', study)
    self.set_output('best_trial', study.best_trials[0])
  
  
class CVPredictionsStage(PipelineStage):
  def make_cv_preds(
    self, 
    model: Model, 
    features:np.ndarray, 
    targets: np.ndarray, 
    folds: int = 5
  ):
    pass
  
  def run(self):
    pass
    

class PlotsStage(PipelineStage):
  def __init__(self):
    pass
  
  def run(self):
    pass
