# Importacion de Librerias
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime

data_cleaned = pd.read_excel(
    r"C:\Users\Daniel Martinez\Desktop\Tesis de Grado Maestria\Exploración\Analisis De Aceite Flotas Mineras.xlsx"
)

data_cleaned=data_cleaned.drop(columns=['SACODE_Salud','SACODE_Desgaste','SACODE_Contaminacion','SACODE_General'])

COLS_SALUD = [
    'Boro (ppm)', 'Calcio (ppm)', 'Cinc (ppm)', 'Fosforado (ppm)',
    'Magnesio (ppm)', 'Molibdeno (ppm)', 'Nitración JOAP (Abs/cm)',
    'Oxidación JOAP (Abs/cm)', 'Sulfatación JOAP (Abs/cm)', 'Sodio (ppm)'
]

COLS_DESGASTE = [
    'Aluminio (ppm)', 'Cobre (ppm)', 'Cromo (ppm)', 'Estaño (ppm)',
    'Hierro (ppm)', 'Plomo (ppm)', 'Oxidación JOAP (Abs/cm)'
]

COLS_CONTAMINACION = [
    'Agua (%)', 'Dilución por combustible (%)',
    'Hollín JOAP (Abs/cm)', 'Silicio (ppm)',
    'Sodio (ppm)', 'Aluminio (ppm)'
]


AJUSTES_CLIENTE = {
    'salud': {
        'Sodio (ppm)'              : {'lim_warn_sup': 40,  'lim_critico_sup': 100},
        'Oxidación JOAP (Abs/cm)'  : {'lim_warn_sup': 19,  'lim_critico_sup': 21},
    },
    'desgaste': {
        'Cobre (ppm)'              : {'lim_warn_sup': 6,   'lim_critico_sup': 8},
        'Plomo (ppm)'              : {'lim_warn_sup': 4,   'lim_critico_sup': 6},
    },
    'contaminacion': {}
}

# ------------------------------------------------------------------------------
# Funciones SACODE
# ------------------------------------------------------------------------------

def calcular_limites(df: pd.DataFrame,
                     factor_warning: float = 2.0,
                     factor_critico: float = 3.0) -> pd.DataFrame:
    """Calcula límites condenatorios por desviación estándar (Metodología Noria)."""
    stats = pd.DataFrame({
        'media'           : df.mean(),
        'sigma'           : df.std(),
        'lim_warn_sup'    : df.mean() + factor_warning * df.std(),
        'lim_warn_inf'    : (df.mean() - factor_warning * df.std()).clip(lower=0),
        'lim_critico_sup' : df.mean() + factor_critico * df.std(),
        'lim_critico_inf' : (df.mean() - factor_critico * df.std()).clip(lower=0),
    })
    return stats


def aplicar_ajustes_cliente(limites: pd.DataFrame,
                             ajustes: dict) -> pd.DataFrame:
    limites = limites.copy()
    for col, vals in ajustes.items():
        if col in limites.index:
            for campo, valor in vals.items():
                limites.loc[col, campo] = valor
    return limites


def clasificar_predictor(valor: float,
                          lim_warn_sup: float,
                          lim_critico_sup: float) -> str:
    if valor <= lim_warn_sup:
        return 'Normal'
    elif valor <= lim_critico_sup:
        return 'Advertencia'
    else:
        return 'Crítico'


def calificar_categoria(df: pd.DataFrame,
                         limites: pd.DataFrame) -> pd.Series:
    orden_severidad = {'Normal': 0, 'Advertencia': 1, 'Crítico': 2}
    inv_orden       = {v: k for k, v in orden_severidad.items()}

    calificaciones = pd.DataFrame(index=df.index)
    for col in df.columns:
        lim_w = limites.loc[col, 'lim_warn_sup']
        lim_c = limites.loc[col, 'lim_critico_sup']
        calificaciones[col] = df[col].apply(
            lambda x: clasificar_predictor(x, lim_w, lim_c)
        )

    peor_caso = calificaciones.map(lambda x: orden_severidad[x]).max(axis=1)
    return peor_caso.map(inv_orden)


# ------------------------------------------------------------------------------
# Cálculo SACODE por Flota
# ------------------------------------------------------------------------------

flotas         = data_cleaned['Flota'].unique()
limites_flota  = {}   # Almacena límites calculados por flota y categoría
resultados     = []   # Acumula filas clasificadas

print("=" * 65)
print("CALIFICACIÓN SACODE POR FLOTA")
print("=" * 65)

for flota in sorted(flotas):

    df_flota = data_cleaned[data_cleaned['Flota'] == flota].copy()
    n        = len(df_flota)

    if n < 30:
        print(f"\n⚠️  Flota '{flota}' omitida — solo {n} registros (mínimo 30)")
        continue

    print(f"\n{'─'*55}")
    print(f"  Flota : {flota}  ({n} registros)")
    print(f"{'─'*55}")

    # Subsets por categoría
    salud_f        = df_flota[COLS_SALUD]
    desgaste_f     = df_flota[COLS_DESGASTE]
    contaminacion_f = df_flota[COLS_CONTAMINACION]

    # Cálculo de límites estadísticos
    lim_salud        = calcular_limites(salud_f)
    lim_desgaste     = calcular_limites(desgaste_f)
    lim_contaminacion = calcular_limites(contaminacion_f)

    # Aplicar ajustes del cliente
    lim_salud        = aplicar_ajustes_cliente(lim_salud,        AJUSTES_CLIENTE['salud'])
    lim_desgaste     = aplicar_ajustes_cliente(lim_desgaste,     AJUSTES_CLIENTE['desgaste'])
    lim_contaminacion = aplicar_ajustes_cliente(lim_contaminacion, AJUSTES_CLIENTE['contaminacion'])

    # Guardar límites para trazabilidad
    limites_flota[flota] = {
        'salud'        : lim_salud,
        'desgaste'     : lim_desgaste,
        'contaminacion': lim_contaminacion
    }

    # Clasificación SACODE
    df_flota['SACODE_Salud']         = calificar_categoria(salud_f,         lim_salud)
    df_flota['SACODE_Desgaste']      = calificar_categoria(desgaste_f,      lim_desgaste)
    df_flota['SACODE_Contaminacion'] = calificar_categoria(contaminacion_f, lim_contaminacion)

    # Calificación general — peor caso entre las 3 categorías
    orden_sev  = {'Normal': 0, 'Advertencia': 1, 'Crítico': 2}
    inv_orden  = {v: k for k, v in orden_sev.items()}
    sev_num    = df_flota[['SACODE_Salud', 'SACODE_Desgaste', 'SACODE_Contaminacion']]\
                     .map(lambda x: orden_sev[x])
    df_flota['SACODE_General'] = sev_num.max(axis=1).map(inv_orden)

    # Resumen de distribución
    for cat in ['SACODE_Salud', 'SACODE_Desgaste', 'SACODE_Contaminacion', 'SACODE_General']:
        dist = df_flota[cat].value_counts()
        pct  = df_flota[cat].value_counts(normalize=True).mul(100).round(1)
        print(f"\n  {cat}:")
        for estado in ['Normal', 'Advertencia', 'Crítico']:
            cnt = dist.get(estado, 0)
            p   = pct.get(estado, 0.0)
            print(f"    {estado:12s}: {cnt:4d}  ({p:.1f}%)")

    resultados.append(df_flota)

df_test=pd.read_excel(
    r"C:\Users\Daniel Martinez\Desktop\Tesis de Grado Maestria\Exploración\Analisis De Aceite Flotas Mineras_test.xlsx"
)
df_flota1_test=df_test[df_test['Flota']=='Flota_1']
# Variables predictoras y variable objetivo
X_flota1 = df_flota1_test.drop(columns=['Assigned Condition Rating',
                             'ACR_Homologado',
                             'Assigned Condition Rating',
                             'Fault Effect',
                                'SACODE_Salud',
                                'SACODE_Desgaste',
                                'SACODE_Contaminacion',
                                'SACODE_General','Flota','Rule Based Rating'])

Salud_train = X_flota1[['Boro (ppm)', 'Calcio (ppm)', 'Cinc (ppm)', 'Fosforado (ppm)',
                       'Magnesio (ppm)', 'Molibdeno (ppm)', 'Nitración JOAP (Abs/cm)',
                       'Oxidación JOAP (Abs/cm)', 'Sulfatación JOAP (Abs/cm)', 'Sodio (ppm)']]
Desgaste_train = X_flota1[['Aluminio (ppm)', 'Cobre (ppm)', 'Cromo (ppm)', 'Estaño (ppm)',
                          'Hierro (ppm)', 'Plomo (ppm)',
                          'Oxidación JOAP (Abs/cm)']]
Contaminacion_train = X_flota1[['Agua (%)', 'Dilución por combustible (%)',
                               'Hollín JOAP (Abs/cm)', 'Silicio (ppm)',
                               'Sodio (ppm)', 'Aluminio (ppm)']]
X_flota1['SACODE_Salud']         = calificar_categoria(Salud_train,         limites_flota['Flota_1']['salud'])
X_flota1['SACODE_Desgaste']      = calificar_categoria(Desgaste_train,      limites_flota['Flota_1']['desgaste'])
X_flota1['SACODE_Contaminacion'] = calificar_categoria(Contaminacion_train, limites_flota['Flota_1']['contaminacion'])
X_flota1['SACODE_General'] = X_flota1[['SACODE_Salud', 'SACODE_Desgaste', 'SACODE_Contaminacion']].apply(
    lambda x: 'Crítico' if 'Crítico' in x.values else ('Advertencia' if 'Advertencia' in x.values else 'Normal'),
    axis=1
)

X_flota1["Flota"] = 'Flota_1'

df_flota2=df_test[df_test['Flota']=='Flota_2']
# Variables predictoras y variable objetivo
X_flota2 = df_flota2.drop(columns=['Assigned Condition Rating',
                             'ACR_Homologado',
                             'Assigned Condition Rating',
                             'Fault Effect',
                                'SACODE_Salud',
                                'SACODE_Desgaste',
                                'SACODE_Contaminacion',
                                'SACODE_General','Flota','Rule Based Rating'])

Salud_train = X_flota2[['Boro (ppm)', 'Calcio (ppm)', 'Cinc (ppm)', 'Fosforado (ppm)',
                       'Magnesio (ppm)', 'Molibdeno (ppm)', 'Nitración JOAP (Abs/cm)',
                       'Oxidación JOAP (Abs/cm)', 'Sulfatación JOAP (Abs/cm)', 'Sodio (ppm)']]
Desgaste_train = X_flota2[['Aluminio (ppm)', 'Cobre (ppm)', 'Cromo (ppm)', 'Estaño (ppm)',
                          'Hierro (ppm)', 'Plomo (ppm)',
                          'Oxidación JOAP (Abs/cm)']]
Contaminacion_train = X_flota2[['Agua (%)', 'Dilución por combustible (%)',
                               'Hollín JOAP (Abs/cm)', 'Silicio (ppm)',
                               'Sodio (ppm)', 'Aluminio (ppm)']]
X_flota2['SACODE_Salud']         = calificar_categoria(Salud_train,         limites_flota['Flota_2']['salud'])
X_flota2['SACODE_Desgaste']      = calificar_categoria(Desgaste_train,      limites_flota['Flota_2']['desgaste'])
X_flota2['SACODE_Contaminacion'] = calificar_categoria(Contaminacion_train, limites_flota['Flota_2']['contaminacion'])
X_flota2['SACODE_General'] = X_flota2[['SACODE_Salud', 'SACODE_Desgaste', 'SACODE_Contaminacion']].apply(
    lambda x: 'Crítico' if 'Crítico' in x.values else ('Advertencia' if 'Advertencia' in x.values else 'Normal'),
    axis=1
)

X_flota2["Flota"] = 'Flota_2'

df_flota3=df_test[df_test['Flota']=='Flota_3']


# Variables predictoras y variable objetivo
X_flota3 = df_flota3.drop(columns=['Assigned Condition Rating',
                             'ACR_Homologado',
                             'Assigned Condition Rating',
                             'Fault Effect',
                                'SACODE_Salud',
                                'SACODE_Desgaste',
                                'SACODE_Contaminacion',
                                'SACODE_General','Flota','Rule Based Rating'])

Salud_train = X_flota3[['Boro (ppm)', 'Calcio (ppm)', 'Cinc (ppm)', 'Fosforado (ppm)',
                       'Magnesio (ppm)', 'Molibdeno (ppm)', 'Nitración JOAP (Abs/cm)',
                       'Oxidación JOAP (Abs/cm)', 'Sulfatación JOAP (Abs/cm)', 'Sodio (ppm)']]
Desgaste_train = X_flota3[['Aluminio (ppm)', 'Cobre (ppm)', 'Cromo (ppm)', 'Estaño (ppm)',
                          'Hierro (ppm)', 'Plomo (ppm)',
                          'Oxidación JOAP (Abs/cm)']]
Contaminacion_train = X_flota3[['Agua (%)', 'Dilución por combustible (%)',
                               'Hollín JOAP (Abs/cm)', 'Silicio (ppm)',
                               'Sodio (ppm)', 'Aluminio (ppm)']]
X_flota3['SACODE_Salud']         = calificar_categoria(Salud_train,         limites_flota['Flota_3']['salud'])
X_flota3['SACODE_Desgaste']      = calificar_categoria(Desgaste_train,      limites_flota['Flota_3']['desgaste'])
X_flota3['SACODE_Contaminacion'] = calificar_categoria(Contaminacion_train, limites_flota['Flota_3']['contaminacion'])
X_flota3['SACODE_General'] = X_flota3[['SACODE_Salud', 'SACODE_Desgaste', 'SACODE_Contaminacion']].apply(
    lambda x: 'Crítico' if 'Crítico' in x.values else ('Advertencia' if 'Advertencia' in x.values else 'Normal'),
    axis=1
)

X_flota3["Flota"] = 'Flota_3'

X_final = pd.concat([X_flota1, X_flota2, X_flota3], ignore_index=True)


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Carpeta del modelo
MODEL_DIR ="C:/Users/Daniel Martinez/Desktop/Tesis de Grado Maestria/Modelos_ML/modelo_oilsense"

# Cargar archivos
pipeline_cargado = joblib.load(
    MODEL_DIR / "pipeline_lightgbm_calibrado.pkl"
)

encoder_cargado = joblib.load(
    MODEL_DIR / "label_encoder.pkl"
)

limites_cargados = joblib.load(
    MODEL_DIR / "limites_sacode_flota.pkl"
)

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = "C:/Users/Daniel Martinez/Desktop/Tesis de Grado Maestria/Modelos_ML/modelo_oilsense"

config_path = MODEL_DIR / "config_modelo.json"

with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# ==============================================================================
# PREDICCIÓN Y EXPORTACIÓN DEL DATASET FINAL
# ==============================================================================

N_MUESTRAS     = len(X_final)
y_proba_nuevas = pipeline_cargado.predict_proba(X_final)
y_pred_default = pipeline_cargado.predict(X_final)

# ==============================================================================
# Decodificar predicciones de numérico a texto
# ==============================================================================

predicciones_texto = encoder_cargado.inverse_transform(y_pred_default)

# ==============================================================================
# Construir dataset final
# ==============================================================================

final = X_final.copy()   # ← dataframe original con todas las columnas

# Predicción del modo de falla
final['Fault_Effect_Predicho'] = predicciones_texto


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Carpeta del modelo
MODEL_DIR_severidad = "C:/Users/Daniel Martinez/Desktop/Tesis de Grado Maestria/Modelos_ML/modelo_oilsense_severidad"

# Cargar archivos
pipeline_cargado_severidad = joblib.load(
    MODEL_DIR_severidad / "pipeline_randomforest_calibrado.pkl"
)

encoder_cargado_severidad = joblib.load(
    MODEL_DIR_severidad / "label_encoder.pkl"
)

limites_cargados_severidad = joblib.load(
    MODEL_DIR_severidad / "limites_sacode_flota.pkl"
)

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR_severidad = "C:/Users/Daniel Martinez/Desktop/Tesis de Grado Maestria/Modelos_ML/modelo_oilsense_severidad"

config_path = MODEL_DIR_severidad / "config_modelo.json"

with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)


final_sev=final.copy()
final_sev= final_sev[~((final_sev['Fault_Effect_Predicho'] == 'No Fault Identified'))]

N_MUESTRAS     = len(final_sev)
y_proba_nuevas_severidad = pipeline_cargado_severidad.predict_proba(final_sev)
y_pred_default_severidad = pipeline_cargado_severidad.predict(final_sev)

# ==============================================================================
# Decodificar predicciones de numérico a texto
# ==============================================================================

predicciones_severidad= encoder_cargado_severidad.inverse_transform(y_pred_default_severidad)
final_sev['Severidad_Predicha'] = predicciones_severidad
final_norm=final.copy()
final_norm= final_norm[~((final_norm['Fault_Effect_Predicho'] != 'No Fault Identified'))]
final_norm['Severidad_Predicha'] = 'Normal'
final = pd.concat([final_sev, final_norm], ignore_index=True)
final.to_excel("dataset_final_con_prediccion.xlsx", index=False)