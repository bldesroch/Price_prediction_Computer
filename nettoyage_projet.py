import pandas as pd
import numpy as np
import json
import regex as re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder



def encodage_prix(df):
    """Pour but de nettoyer la colonne prix supprimer les caractères spéciaux 
    et transfomer la colonne en type numérique"""
    
    df['prix'] = df['prix'].astype(str).str.replace('€', '.')
    df['prix'] = df['prix'].astype(float).astype(int) 
    return df

def supression_caractere_spe_colonnes(df):
    """Objectif est de supprimer les caractères spéciaux et de transformer les caractères 
    en minuscule les colonnes description et nom pour rendre la recherche de mot clé plus facile """
    
    df['desc'] = df['desc'].str.replace('[^A-Za-z0-9]+','').str.lower()
    df['nom'] = df['nom'].str.replace('[^A-Za-z0-9]+','').str.lower()
    return df


def encodage_processeur(df: pd.DataFrame) -> pd.DataFrame:
    """Recherche dans la colonne desc les caractéristiques recherchées pour notre modèle 
    et création d'une nouvelle colonne (proc) regroupant les informations sur les processeurs trouvées."""
    
    df['proc'] = df['desc'].str.find("i5")
    df.loc[df['proc'] != -1,'proc'] = 0
    df['proc1'] = df['desc'].str.find("i7")
    df.loc[df['proc1'] != -1,'proc1'] = 2
    df.loc[df['proc1'] == -1,'proc1'] = 0
    df["proc"] = df["proc"] + df["proc1"]
    del df["proc1"]
    df['proc1'] = df['desc'].str.find("i3")
    df.loc[df['proc1'] != -1,'proc1'] = 3
    df.loc[df['proc1'] == -1,'proc1'] = 0
    df["proc"] = df["proc"] + df["proc1"]
    del df["proc1"]
    df['proc1'] = df['desc'].str.find("amdryzen7")
    df.loc[df['proc1'] != -1,'proc1'] = 4
    df.loc[df['proc1'] == -1,'proc1'] = 0
    df["proc"] = df["proc"] + df["proc1"]
    del df["proc1"]
    df['proc1'] = df['desc'].str.find("amdryzen5")
    df.loc[df['proc1'] != -1,'proc1'] = 5
    df.loc[df['proc1'] == -1,'proc1'] = 0
    df["proc"] = df["proc"] + df["proc1"]
    del df["proc1"]
    df = df[df['proc'] >= 0]
    return df

def encodage_ram(df: pd.DataFrame) -> pd.DataFrame:
    """Recherche dans la colonne desc les caractéristiques recherchées pour notre modèle 
    et création d'une nouvelle colonne (ram) regroupant les informations sur la mémoire vive trouvées."""

    df['ram'] = df['desc'].str.find("ram2go")
    df.loc[df['ram'] != -1,'ram'] = 0
    df['ram1'] = df['desc'].str.find("2goram")
    df.loc[df['ram1'] != -1,'ram'] = 1
    df.loc[df['ram1'] == -1,'ram'] = 0
    df["ram"] = df["ram"] + df["ram1"]
    del df["ram1"]
    df['ram1'] = df['desc'].str.find("ram4go")
    df.loc[df['ram1'] != -1,'ram1'] = 2
    df.loc[df['ram1'] == -1,'ram1'] = 0
    df["ram"] = df["ram"] + df["ram1"]
    del df["ram1"]
    df['ram1'] = df['desc'].str.find("4goram")
    df.loc[df['ram1'] != -1,'ram1'] = 2
    df.loc[df['ram1'] == -1,'ram1'] = 0
    df["ram"] = df["ram"] + df["ram1"]
    del df["ram1"]
    df['ram1'] = df['desc'].str.find("ram8go")
    df.loc[df['ram1'] != -1,'ram1'] = 3
    df.loc[df['ram1'] == -1,'ram1'] = 0
    df["ram"] = df["ram"] + df["ram1"]
    del df["ram1"]
    df['ram1'] = df['desc'].str.find("8goram")
    df.loc[df['ram1'] != -1,'ram1'] = 3
    df.loc[df['ram1'] == -1,'ram1'] = 0
    df["ram"] = df["ram"] + df["ram1"]
    del df["ram1"]
    df['ram1'] = df['desc'].str.find("16goram")
    df.loc[df['ram1'] != -1,'ram1'] = 4
    df.loc[df['ram1'] == -1,'ram1'] = 0
    df["ram"] = df["ram"] + df["ram1"]
    del df["ram1"]
    df['ram1'] = df['desc'].str.find("ram16go")
    df.loc[df['ram1'] != -1,'ram1'] = 4
    df.loc[df['ram1'] == -1,'ram1'] = 0
    df["ram"] = df["ram"] + df["ram1"]
    del df["ram1"]
    df = df[df['ram'] <= 3]
    return df
    
def encodage_marque(df: pd.DataFrame) -> pd.DataFrame:
    """Recherche dans la colonne desc les caractéristiques recherchées pour notre modèle 
    et création d'une nouvelle colonne (marq) regroupant les informations sur les marques trouvées."""
        
    df['marq'] = df['nom'].str.find("asus")
    df.loc[df['marq'] != -1,'marq'] = 0
    df['marq1'] = df['nom'].str.find("apple")
    df.loc[df['marq1'] != -1,'marq1'] = 2
    df.loc[df['marq1'] == -1,'marq1'] = 0
    df["marq"] = df["marq"] + df["marq1"]
    del df["marq1"]
    df['marq1'] = df['nom'].str.find("lenovo")
    df.loc[df['marq1'] != -1,'marq1'] = 3
    df.loc[df['marq1'] == -1,'marq1'] = 0
    df["marq"] = df["marq"] + df["marq1"]
    del df["marq1"]
    df['marq1'] = df['nom'].str.find("acer")
    df.loc[df['marq1'] != -1,'marq1'] = 4
    df.loc[df['marq1'] == -1,'marq1'] = 0
    df["marq"] = df["marq"] + df["marq1"]
    del df["marq1"]
    df['marq1'] = df['nom'].str.find("hp")
    df.loc[df['marq1'] != -1,'marq1'] = 5
    df.loc[df['marq1'] == -1,'marq1'] = 0
    df["marq"] = df["marq"] + df["marq1"]
    del df["marq1"]
    df['marq1'] = df['nom'].str.find("huawei")
    df.loc[df['marq1'] != -1,'marq1'] = 6
    df.loc[df['marq1'] == -1,'marq1'] = 0
    df["marq"] = df["marq"] + df["marq1"]
    del df["marq1"]
    df['marq1'] = df['nom'].str.find("msi")
    df.loc[df['marq1'] != -1,'marq1'] = 7
    df.loc[df['marq1'] == -1,'marq1'] = 0
    df["marq"] = df["marq"] + df["marq1"]
    del df["marq1"]
    df['marq1'] = df['nom'].str.find("dell")
    df.loc[df['marq1'] != -1,'marq1'] = 8
    df.loc[df['marq1'] == -1,'marq1'] = 0
    df["marq"] = df["marq"] + df["marq1"]
    del df["marq1"]
    df = df[df['marq'] >= 0]
    return df

def suppression_colonnes_inutiles (df):
    """ Supression des colonnes désormais inutiles (desc, nom et lien)"""
    del df['desc']
    del df['nom']
    del df['lien']
    del df['Unnamed: 0']
    return df

def verification_valeur_manquante(df: pd.DataFrame) -> pd.DataFrame:
    """Permet de vérifier qu'il n'y a pas de valeur manquante dans la base de données nettoyée."""
    print(df.isna().any())

def conversion_types(df: pd.DataFrame) -> pd.DataFrame:
    """Permet de transformer les types des colonnes naturellement. """
    return df.convert_dtypes()

def _verifie_conversion_types() -> bool:
    """Vérifie la fonction précédente"""
    entree = __produit_df_test()
    resultat = conversion_types(entree).dtypes.apply(type)
    en_theorie = pd.Series(
        index=["prix", "proc", "ram", "marq"],
        data=[pd.Int32Dtype, pd.Int64Dtype, pd.Int64Dtype, pd.Int64Dtype]
    )
    return (resultat == en_theorie).all()


def numerise_les_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit la colonne en numérique"""
    def filtre(ram):
        if ram == -1:
            return 4
        return ram
     
    resultat = df.copy()
    resultat.ram = resultat.ram.apply(filtre)
    return resultat

def suppression_colonne(df: pd.DataFrame) -> pd.DataFrame:
    """Enleve la colonnee id, inutile une fois que les
    doublons ont été supprimés.
    """
    return df.drop(axis=1, labels="Unnamed: 0")

def _verifie_suppression_colonne() -> bool:
    """Vérifie la fonction précédente"""
    ...

def pie_chart(df):
    df["marq"].value_counts().plot(kind='pie')
    # Cette ligne assure que le pie chart est un cercle plutôt qu'une éllipse
    plt.axis('equal') 
    plt.show()
    df["ram"].value_counts().plot(kind='pie')
    plt.axis('equal') 
    plt.show()
    df["proc"].value_counts().plot(kind='pie')
    plt.axis('equal') 
    plt.show()

    
def densité_prix(df):
    sns.distplot(df['prix'], fit=norm)
    
def densité_log_prix(df):
    df['prix'] = np.log(df['prix'])
    sns.distplot(df['prix'], fit=norm)
    
def moyenne_par_groupe(df):
    print(df.groupby(['marq']).mean())
    print(df.groupby(['proc']).mean())
    print(df.groupby(['ram']).mean())
    
    
enc = OneHotEncoder(handle_unknown='ignore')

def onehot_colonne_marq(df):
    enc_df = pd.DataFrame(enc.fit_transform(df[['marq']]).toarray())
    df = df.join(enc_df)
    df = df.rename(columns = {0: 'Asus', 1: 'Apple', 2:'Lenovo', 3:'Acer', 4:'Hp', 5:'Huawei', 6:'MSI', 7:'Dell'})
    return df

def onehot_colonne_proc(df):
    enc_df = pd.DataFrame(enc.fit_transform(df[['proc']]).toarray())
    df = df.join(enc_df)
    df = df.rename(columns = {0: 'Intel Core i5', 1: 'Intel Core i7',2:'Intel Core i3',3:'AMD Ryzen 7',4:'AMD Ryzen 5'})
    return df

def onehot_colonne_ram(df):
    enc_df = pd.DataFrame(enc.fit_transform(df[['ram']]).toarray())
    df = df.join(enc_df)
    df = df.rename(columns = {0: '4 Go RAM', 1: '8 Go RAM',2:'16 Go RAM',3:'X Go RAM'})
    return df

def suppression_colonnes_inutiles_2(df):
    """On supprime les colonnes marq, ram et proc pour ne garder les variables transformées et utiles pour la modélisation"""
    del df['marq']
    del df['ram']
    del df['proc']
    return df

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


def cherche_le_meilleur_modele(X_tr, y_tr):
    modeles = list()
    modeles.append(LinearRegression())
    for val_alpha in (1e-3, 1e-2, 1e-1, 1):
        modeles.append(Lasso(alpha=val_alpha))
    for val_alpha in (1e-3, 1e-2, 1e-1, 1):
        modeles.append(Ridge(alpha=val_alpha))
    for val_alpha in (1e-3, 1e-2, 1e-1, 1):
        for val_l1 in (0.25, 0.5, 0.75):
            modeles.append(ElasticNet(alpha=val_alpha, l1_ratio=val_l1))
    for nb_voisins in range(3, 10):
        modeles.append(KNeighborsRegressor(n_neighbors=nb_voisins))
    modeles.append(GaussianProcessRegressor())
    for nb_estimateurs in (10, 20, 30, 50, 100, 150, 200):
        for maxx_depth in (80, 90, 100, 110):
            modeles.append(RandomForestRegressor(n_estimators=nb_estimateurs, max_depth=maxx_depth))
    resultats = dict()
    for modele in modeles:
        resultats[modele] = cross_val_score(modele, X_tr, y_tr, cv=5)
    df_result=pd.DataFrame.from_dict(resultats, orient='index').reset_index()
    df_result.columns = ['Modele', 'A', 'B', 'C','D','E']
    df_result['Moyenne'] = df_result.mean(axis = 1)
    df_result['Ecart Type'] = df_result.std(axis = 1)
    del df_result['A']
    del df_result['B']
    del df_result['C']
    del df_result['D']
    del df_result['E']
    df_result = df_result.sort_values(by=['Moyenne', 'Ecart Type'], ascending=[False, True])
    df_result.to_csv('df_result.csv', sep='\t')

    
"Random forest best estimators"    
def Trouver_meilleur_Random_Forest(X_tr, y_tr):
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [30,50,100, 200]
    }
    rf = RandomForestRegressor()
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                               cv = 3)
    grid_search.fit(X_tr, y_tr)
    grid_search.best_params_
    best_grid = grid_search.best_estimator_
    print(best_grid)
    

"Evaluation du modèle choisi"
    
from sklearn import metrics
from sklearn.model_selection import learning_curve


def evaluation(best_model, X, y, X_tr, y_tr, X_te, y_te):
    "Evaluation complète du modèle choisi"
    best_model.fit(X_tr, y_tr)
    y_pred = best_model.predict(X_te)  
    print('Model score',best_model.score(X_te, y_te))
    print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_te, y_pred))
    print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_te, y_pred))
    print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_te, y_pred)))
    mape = np.mean(np.abs((y_te - y_pred) / np.abs(y_te)))
    print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
    print('Accuracy:', round(100*(1 - mape), 2))
    
    N, train_score, val_score = learning_curve(best_model, X_tr, y_tr,
                                              cv=4, scoring='neg_mean_squared_error',
                                               train_sizes=np.linspace(0.1, 1, 10))
    
    
    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='training error')
    plt.plot(N, val_score.mean(axis=1), label='cross-validation error')
    plt.legend()
    plt.ylabel('Mean absolute error')
    plt.xlabel('Training Size')
    title = "Courbe d'apprentisage du meilleur modèle : RandomForestRegressor"
    plt.title(title, fontsize =18, y= 1.03)

    
if __name__ == "__main__":
    assert _verifie_conversion_types()
    assert _verifie_suppression_colonne()
    assert _verifie_numerise_les_colonnes()
    assert _verifie_supprime_partiellement_na()
    