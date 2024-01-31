from io import BytesIO
from pathlib import Path

import pandas as pd
from astromodule.io import parallel_function_executor, read_table, write_table
from astromodule.splus import SplusService
from astromodule.table import concat_tables
from astropy.io import fits

HEADERS_FOLDER = Path('headers')
service = SplusService('natanael', 'natan')


def header_to_dataframe(header: fits.Header) -> pd.DataFrame:
  """
  Receives a header and returns a dataframe with the keys to the headers
  as Dataframe columns

  Parameters
  ----------
  header : fits.Header
    Header to be transformed into dataframe

  Returns
  -------
  pd.DataFrame
    Resulting dataframe
  """
  data = {k: v for k, v in header.items()}
  return pd.DataFrame([data])


def download_field_and_save_header(field: str, replace: bool = False):
  header_path = HEADERS_FOLDER / f'{field}.csv'
  
  if header_path.exists() and not replace: return
  
  field_file = BytesIO()
  service.download_field(
    field=field,
    output=field_file,
    band='R',
    weight_image=False,
  )
  field_file.seek(0)
  hdu_index = -1
  hdul = fits.open(field_file)
  # for i, hdu in enumerate(hdul):
  #   if hdu.shape:
  #     hdu_index = i
  #     break
  header = hdul[hdu_index].header
  header_df = header_to_dataframe(header)
  write_table(header_df, header_path)
  del field_file
  
  
def create_final_catalog():
  df = concat_tables(HEADERS_FOLDER.glob('*.csv'), comment='#')
  write_table(df, 'splus_headers+raw.csv')
  # write_table(df, 'splus_headers+raw.parquet')
  
  def column_name_cleaner(col: str):
    return col.replace('HIERARCH OAJ PRO ', '').replace('HIERARCH OAJ QC ', '').lower()
  
  df = df.rename(columns=column_name_cleaner)
  write_table(df, 'splus_headers+clean.csv')
  # write_table(df, 'splus_headers+clean.parquet')


def entrypoint():
  splus_fields = read_table('dr4_fields.csv')['Field'].values
  
  params = [
    {'field': field, 'replace': False}
    for field in splus_fields
  ]
  
  parallel_function_executor(
    func=download_field_and_save_header,
    params=params,
    workers=6,
    unit='field',
  )
  
  create_final_catalog()
  
  
if __name__ == '__main__':
  entrypoint()