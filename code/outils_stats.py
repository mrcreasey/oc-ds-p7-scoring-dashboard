"""
Fonctions statistiques
"""

import pandas as pd


def decrire_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Retourner un dataframe avec des informations sur les colonnes d'un dataframe"""
    print(f'{df.shape[0]} registres', f'{df.shape[1]} colonnes')
    print(f'dtypes={df.dtypes.value_counts()}')
    ret = df.columns.to_frame(name="column").set_index('column')
    ret['% manquantes'] = df.isna().mean()*100
    ret['unique'] = df.nunique()
    ret['dtype'] = df.dtypes
    ret['min'] = df.min()
    ret['max'] = df.max()
    ret['std'] = df.std()
    return ret


