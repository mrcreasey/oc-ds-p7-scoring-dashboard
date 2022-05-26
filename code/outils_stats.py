"""
Fonctions statistiques
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def list_colonnes_remplis(df: pd.DataFrame) -> list:
    """Retourne un liste de colonnes 100% remplis"""
    return df.columns[~df.isnull().any()].tolist()


def plot_bar_remplissage(df: pd.DataFrame, nb=200, titre=None, figsize=(10, 5),
                         vertical=False, filled_cols=False, thresh=50):
    """Visualise pourcent remplissage des colonnes"""
    remplissage = (round((1-df.isna().mean())*100)
                   .sort_values(ascending=False)
                   .to_frame(name='taux')
                   .rename_axis('colonne').reset_index())
    if filled_cols is False:
        remplissage = remplissage[remplissage['taux'] < 99.99]
    nb = min(nb, len(remplissage))
    remplissage = remplissage.head(nb)
    plt.figure(figsize=figsize)
    remplissage['hue'] = remplissage['taux']//7 * 7
    remplissage['hue'] = '>= ' + remplissage['hue'].astype(str)
    fontsize = 14

    if vertical:
        ax = sns.barplot(data=remplissage, y='colonne',
                         x='taux', hue='hue', dodge=False)
        ax.set_xlabel('Pourcent remplissage (%)', size=fontsize)
        ax.set_ylabel('Nom de la colonne', size=fontsize)
        ax.axvline(thresh, linestyle='--', color='red')
    else:
        ax = sns.barplot(data=remplissage, x='colonne',
                         y='taux', hue='hue', dodge=False)
        ax.set_xlabel('Nom de la colonne', size=fontsize)
        ax.set_ylabel('Pourcent remplissage (%)', size=fontsize)
        plt.xticks(rotation=90)
        ax.axhline(thresh, linestyle='--', color='red')
    if titre is None:
        titre = f'Taux de remplissage des {nb} colonnes plus remplis'
    if not filled_cols:
        titre = 'Taux de remplissage (sans colonnes 100% remplis)'
    ax.set_title(label=f'{titre} [nb={df.shape[0]}]', size=fontsize+2)
    plt.legend(title='remplissage (%)')
    sns.despine()
