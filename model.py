from pathlib import Path
from typing import Sequence

import numpy as np
import optuna
from astromodule.hp import HP, HyperParameterSet
from astromodule.io import read_table
from sklearn import svm
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor, StackingRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import (cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

BROAD_BANDS = ['r', 'g', 'i', 'u', 'z']
NARROW_BANDS = ['J0378', 'J0395', 'J0410', 'J0430', 'J0515', 'J0660', 'J0861']
SPLUS_BANDS = BROAD_BANDS + NARROW_BANDS
RANDOM_STATE = 42
BASE_STUDY_PATH = Path('logs')
BASE_STUDY_PATH.mkdir(parents=True, exist_ok=True)


def get_features_by_band(band: str):
  if isinstance(band, str):
    band = list(band)
  df = read_table('data/zp+fwhm+mjd.csv')
  features_names = [f'{b}_FWHM' for b in band]
  features_names += [f'{b}_MJD' for b in band]
  features = df[features_names].values
  target_names = [f'ZP_{b}' for b in band]
  targets = df[target_names].values
  if targets.shape[-1] == 1:
    targets = targets.ravel()
  return features, targets


def remove_prefix(d: dict, prefix_size: int):
  return {k[prefix_size:]: v for k, v in d.items()}


def get_scaler(scaler: str):
  mapping = {
    'std': StandardScaler(),
    'minmax': MinMaxScaler(), 
  }
  return mapping.get(scaler)


def get_estimator(algorithm: str, hps: dict):
  mapping = {
    'RandomForestRegressor': RandomForestRegressor,
    'ExtraTreesRegressor': ExtraTreesRegressor,
    'GaussianMixture': GaussianMixture,
    'SVR': svm.SVR,
    'NuSVR': svm.NuSVR,
    'KNeighborsRegressor': KNeighborsRegressor,
    'MLPRegressor': MLPRegressor,
    'LogisticRegression': LogisticRegression,
    'GradientBoostingRegressor': GradientBoostingRegressor,
  }
  return mapping.get(algorithm)(**hps)


def get_study(study_name: str):
  study_path = str(BASE_STUDY_PATH / f'{study_name}.log')
  return optuna.load_study(
    study_name=study_name,
    storage=optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(study_path)
      )
  )


def get_best_trial(study_name: str):
  return get_study(study_name).best_trials[0]



class Model:
  def __init__(
    self, 
    band: str, 
    hps: HyperParameterSet, 
    features: np.ndarray, 
    targets: np.ndarray
  ):
    self.hps = hps
    self.features = features
    self.targets = targets
    self.band = band
    # self.features, self.targets = get_features_by_band(band)
    self._study = None
    
    
  @staticmethod
  def from_best_trial(band: str, features: np.ndarray, targets: np.ndarray):
    study_name = f'{band}_ensemble'
    best_trial = get_best_trial(study_name)
    hps = HyperParameterSet.from_dict(best_trial.params)
    return Model(band=band, hps=hps, features=features, targets=targets)
    
  
  @property
  def study(self):
    if self._study is None:
      study_name = f'{self.band}_ensemble'
      study_path = str(BASE_STUDY_PATH / f'{study_name}.log')
      self._study = optuna.create_study(
        load_if_exists=True,
        study_name=study_name,
        storage=optuna.storages.JournalStorage(
          optuna.storages.JournalFileStorage(study_path)
        ),
        directions=[
          optuna.study.StudyDirection.MAXIMIZE, # R^2
          optuna.study.StudyDirection.MINIMIZE  # MSE
        ],
        sampler=optuna.samplers.TPESampler(
          # multivariate=True, 
          n_startup_trials=15
        ),
      )
    return self._study
  
  
  def get_committee(self):
    estimators = []
    if self.hps.get('use_rf'):
      hps = remove_prefix(self.hps.get(r'rf_*', regex=True), 3)
      estimator = get_estimator('RandomForestRegressor', hps)
      estimators.append(('RandomForestRegressor', estimator))
    if self.hps.get('use_ert'):
      hps = remove_prefix(self.hps.get(r'ert_*', regex=True), 4)
      estimator = get_estimator('ExtraTreesRegressor', hps)
      estimators.append(('ExtraTreesRegressor', estimator))
    if self.hps.get('use_svr'):
      hps = remove_prefix(self.hps.get(r'svr_*', regex=True), 4)
      estimator = get_estimator('SVR', hps)
      estimators.append(('SVR', estimator))
    if self.hps.get('use_nusvr'):
      hps = remove_prefix(self.hps.get(r'nusvr_*', regex=True), 6)
      estimator = get_estimator('NuSVR', hps)
      estimators.append(('NuSVR', estimator))
    if self.hps.get('use_gm'):
      hps = remove_prefix(self.hps.get(r'gm_*', regex=True), 3)
      estimator = get_estimator('GaussianMixture', hps)
      estimators.append(('GaussianMixture', estimator))
    if self.hps.get('use_lr'):
      hps = remove_prefix(self.hps.get(r'lr_*', regex=True), 3)
      estimator = get_estimator('LogisticRegression', hps)
      estimators.append(('LogisticRegression', estimator))
    if self.hps.get('use_knn'):
      hps = remove_prefix(self.hps.get(r'knn_*', regex=True), 4)
      estimator = get_estimator('KNeighborsRegressor', hps)
      estimators.append(('KNeighborsRegressor', estimator))
    if self.hps.get('use_mlp'):
      hps = remove_prefix(self.hps.get(r'mlp_*', regex=True), 4)
      l1 = hps.pop('layer_1')
      l2 = hps.pop('layer_2')
      hps = {'hidden_layer_sizes': (l1, l2), **hps}
      estimator = get_estimator('MLPRegressor', hps)
      estimators.append(('MLPRegressor', estimator))
    return estimators
  
  
  def get_meta_estimator(self):
    meta_hps = remove_prefix(self.hps.get('meta_*', regex=True), 5)
    meta_estimator = get_estimator('GradientBoostingRegressor', meta_hps)
    return meta_estimator
  
  
  def build_ensemble(self):
    scaler = get_scaler(self.hps.get('scaler'))
    estimators = self.get_committee()
    if len(estimators) == 0:
      return None
    meta_estimator = self.get_meta_estimator()
    reg = StackingRegressor(
      estimators=estimators,
      final_estimator=meta_estimator,
      n_jobs=self.hps.get('n_jobs'),
    )
    pipe = make_pipeline(scaler, reg)
    return pipe
  
  
  def hyperparameter_search(self, n_trials: int = 10, progress: bool = True):
    generator = tqdm(range(n_trials), total=n_trials) if progress else range(n_trials)
    for _ in generator:
      trial = self.study.ask()
      self.hps.set_trial(trial)
      estimator = self.build_ensemble()
      if estimator:
        scores = cross_validate(
          estimator=estimator, 
          X=self.features, 
          y=self.targets, 
          scoring=['r2', 'neg_mean_squared_error'], 
          cv=5
        )
        r2_score = scores['test_r2'].mean()
        mse_score = -1 * scores['test_neg_mean_squared_error'].mean()
        self.study.tell(trial, [r2_score, mse_score])
      else:
        self.study.tell(trial, [0, 1000])
      
      
  def fit(self, test_size: float = 0.2, shuffle_dataset: bool = True):
    X_train, X_test, y_train, y_test = train_test_split(
      self.features, 
      self.targets, 
      test_size=test_size, 
      shuffle=shuffle_dataset,
      random_state=RANDOM_STATE
    )
    estimator = self.build()
    estimator.fit(X_train, y_train)
    return estimator




def model():
  features, targets = get_features_by_band(['r'])
  scaler = StandardScaler()
  scaler = MinMaxScaler()
  # scaler = Normalizer()
  
  reg1 = RandomForestRegressor(
    n_estimators=100, 
    max_depth=10,
    min_samples_split=2, 
    random_state=42
  )
  reg2 = ExtraTreesRegressor(
    n_estimators=10,
    max_depth=None,
    min_samples_split=2,
    random_state=42
  )
  # reg3 = GaussianMixture(
  #   n_components=1,
  #   random_state=42,
  # )
  reg3 = KNeighborsRegressor(
    n_neighbors=5,
    weights='distance',
  )
  # reg = svm.SVR()
  # reg = svm.NuSVR(nu=0.9)
  
  estimators = [('r1', reg1), ('r2', reg2), ('r3', reg3)]
  
  final_estimator = GradientBoostingRegressor(
    n_estimators=25,
    subsample=0.5,
    min_samples_leaf=25,
    max_features=1,
    random_state=42
  )
  reg = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator
  )
  
  pipe = make_pipeline(scaler, reg)
  scores = cross_validate(pipe, features, targets, scoring=['r2', 'neg_mean_squared_error'], cv=5)
  r2_score = scores['test_r2'].mean()
  mse_score = -1 * scores['test_neg_mean_squared_error'].mean()
  return r2_score, mse_score



if __name__ == '__main__':
  # print(read_table('seeing/zp+fwhm+mjd.csv').columns)
  print(model())
  # entrypoint()