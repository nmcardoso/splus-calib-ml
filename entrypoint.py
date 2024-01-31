from astromodule.pipeline import Pipeline

from stages import (BandwiseHPTunnigStage, CVPredictionsStage, DataPrepStage,
                    LoadModelFromBestTrialStage)

'''
VERIFIQUE SE O PACOTE ASTROMODULE ESTÁ NA ÚLTIMA VERSÃO!!!
pip install -U astromodule
'''

HYPEROPT_PIPELINE = Pipeline(
  DataPrepStage(band='r'),
  BandwiseHPTunnigStage(n_trials=1000, progress=True)
)

PREDS_PIPELINE = Pipeline(
  LoadModelFromBestTrialStage(),
  CVPredictionsStage()
)


if __name__ == '__main__':
  HYPEROPT_PIPELINE.run()