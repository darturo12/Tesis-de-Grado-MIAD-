# ==============================================================================
# SHAP VALUES — Modelo de Severidad (Random Forest)
# OilSense ML — Tesis MIAD Universidad de los Andes
# ==============================================================================

import pandas as pd
import numpy as np
import joblib
import json
import warnings
import matplotlib.pyplot as plt
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline    import Pipeline as SKPipeline

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. COLUMNAS
# ==============================================================================

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
    'salud'        : {'Sodio (ppm)'           : {'lim_warn_sup': 40,  'lim_critico_sup': 100},
                      'Oxidación JOAP (Abs/cm)': {'lim_warn_sup': 19,  'lim_critico_sup': 21}},
    'desgaste'     : {'Cobre (ppm)'            : {'lim_warn_sup': 6,   'lim_critico_sup': 8},
                      'Plomo (ppm)'            : {'lim_warn_sup': 4,   'lim_critico_sup': 6}},
    'contaminacion': {}
}
COLS_EXCLUIR_TEST = [
    'Assigned Condition Rating', 'ACR_Homologado', 'Fault Effect',
    'SACODE_Salud', 'SACODE_Desgaste', 'SACODE_Contaminacion',
    'SACODE_General', 'Flota', 'Rule Based Rating'
]
ORDEN_MAP = {'Normal': 0, 'Advertencia': 1, 'Crítico': 2}

# ==============================================================================
# 2. FUNCIONES SACODE
# ==============================================================================

def calcular_limites(df, factor_warning=2.0, factor_critico=3.0):
    return pd.DataFrame({
        'media'          : df.mean(),
        'sigma'          : df.std(),
        'lim_warn_sup'   : df.mean() + factor_warning * df.std(),
        'lim_warn_inf'   : (df.mean() - factor_warning * df.std()).clip(lower=0),
        'lim_critico_sup': df.mean() + factor_critico * df.std(),
        'lim_critico_inf': (df.mean() - factor_critico * df.std()).clip(lower=0),
    })

def aplicar_ajustes_cliente(limites, ajustes):
    limites = limites.copy()
    for col, vals in ajustes.items():
        if col in limites.index:
            for campo, valor in vals.items():
                limites.loc[col, campo] = valor
    return limites

def clasificar_predictor(valor, lim_warn_sup, lim_critico_sup):
    if valor <= lim_warn_sup:      return 'Normal'
    elif valor <= lim_critico_sup: return 'Advertencia'
    else:                          return 'Crítico'

def calificar_categoria(df, limites):
    orden     = {'Normal': 0, 'Advertencia': 1, 'Crítico': 2}
    inv_orden = {v: k for k, v in orden.items()}
    cals      = pd.DataFrame(index=df.index)
    for col in df.columns:
        lw = limites.loc[col, 'lim_warn_sup']
        lc = limites.loc[col, 'lim_critico_sup']
        cals[col] = df[col].apply(lambda x: clasificar_predictor(x, lw, lc))
    return cals.map(lambda x: orden[x]).max(axis=1).map(inv_orden)

def preparar_flota(df_test, flota, limites_flota):
    df_f = df_test[df_test['Flota'] == flota].copy()
    X    = df_f.drop(columns=[c for c in COLS_EXCLUIR_TEST if c in df_f.columns],
                     errors='ignore')
    lims = limites_flota[flota]
    X['SACODE_Salud']         = calificar_categoria(
        X[[c for c in COLS_SALUD         if c in X.columns]], lims['salud'])
    X['SACODE_Desgaste']      = calificar_categoria(
        X[[c for c in COLS_DESGASTE      if c in X.columns]], lims['desgaste'])
    X['SACODE_Contaminacion'] = calificar_categoria(
        X[[c for c in COLS_CONTAMINACION if c in X.columns]], lims['contaminacion'])
    X['SACODE_General'] = X[['SACODE_Salud','SACODE_Desgaste','SACODE_Contaminacion']].apply(
        lambda r: 'Crítico' if 'Crítico' in r.values
                  else ('Advertencia' if 'Advertencia' in r.values else 'Normal'), axis=1
    )
    X['Flota'] = flota
    return X

# ==============================================================================
# 3. CALCULAR LÍMITES SACODE CON DATA DE ENTRENAMIENTO
# ==============================================================================

data_cleaned = pd.read_excel(
    "C:/Users/Daniel Martinez/Desktop/Tesis de Grado Maestria/Exploración/Analisis De Aceite Flotas Mineras.xlsx"
)
data_cleaned = data_cleaned.drop(
    columns=['SACODE_Salud','SACODE_Desgaste','SACODE_Contaminacion','SACODE_General'],
    errors='ignore'
)

limites_flota = {}
for flota in sorted(data_cleaned['Flota'].dropna().unique()):
    df_f = data_cleaned[data_cleaned['Flota'] == flota]
    if len(df_f) < 30:
        continue
    lim_s = aplicar_ajustes_cliente(calcular_limites(df_f[COLS_SALUD]),         AJUSTES_CLIENTE['salud'])
    lim_d = aplicar_ajustes_cliente(calcular_limites(df_f[COLS_DESGASTE]),      AJUSTES_CLIENTE['desgaste'])
    lim_c = aplicar_ajustes_cliente(calcular_limites(df_f[COLS_CONTAMINACION]), AJUSTES_CLIENTE['contaminacion'])
    limites_flota[flota] = {'salud': lim_s, 'desgaste': lim_d, 'contaminacion': lim_c}

print(f"✅ Límites calculados para: {list(limites_flota.keys())}")

# ==============================================================================
# 4. PREPARAR X_FINAL
# ==============================================================================

df_test = pd.read_csv(
    "C:/Users/Daniel Martinez/Desktop/Tesis de Grado Maestria/Exploración/Analisis De Aceite Flotas Mineras_test.csv"
)

df_test = df_test.dropna(axis=1, how='all')
df_test = df_test.drop(
	columns=[
		'Color del Refrigerante','Glycerin (%)','Nitrito (ppm)','PUNTO DE EBULLICION (°C)',
		'Sólidos Totales Disueltos (ppm)','Turbidez (NTU)','pH','Added','Lube Drained','Live',
		'Lube Age','days','Observation Interval','Component Age', 'All Time Meter Reading',
		'Meter Reading','Observation Code','Observation Date', 'Observation Type','Connection Code',
		'Lubricant','Component Profile','Location','Component ID','SILICATOS (ppm)','Asset ID'
	],
	errors='ignore'
)

df_test = df_test.drop(
	columns=[
		'Action Summary','Partículas Ferrosas (ppm)','Further Recommendations','Reviewed',
		'Reviewer','Condition Review Notes','Trakka Rating','Observation Rating'
	],
	errors='ignore'
)

X_final = pd.concat(
    [preparar_flota(df_test, f, limites_flota) for f in ['Flota_1','Flota_2','Flota_3']],
    ignore_index=True
)
print(f"✅ X_final shape: {X_final.shape}")

# ==============================================================================
# 5. CARGAR MODELOS Y PREDECIR
# ==============================================================================

MODEL_DIR     = "C:/Users/Daniel Martinez/Desktop/Tesis de Grado Maestria/Modelos_ML/modelo_oilsense"
MODEL_DIR_SEV = "C:/Users/Daniel Martinez/Desktop/Tesis de Grado Maestria/Modelos_ML/modelo_oilsense_severidad"

pipeline_fe  = joblib.load(MODEL_DIR     + "/pipeline_lightgbm_calibrado.pkl")
encoder_fe   = joblib.load(MODEL_DIR     + "/label_encoder.pkl")
pipeline_sev = joblib.load(MODEL_DIR_SEV + "/pipeline_randomforest_calibrado.pkl")
encoder_sev  = joblib.load(MODEL_DIR_SEV + "/label_encoder.pkl")

print(f"✅ Clases Fault Effect : {encoder_fe.classes_}")
print(f"✅ Clases Severidad    : {encoder_sev.classes_}")

# Predicción Fault Effect
y_pred_fe       = pipeline_fe.predict(X_final)
final           = X_final.copy()
final['Fault_Effect_Predicho'] = encoder_fe.inverse_transform(y_pred_fe)

# Predicción Severidad
final_sev  = final[final['Fault_Effect_Predicho'] != 'No Fault Identified'].copy()
final_norm = final[final['Fault_Effect_Predicho'] == 'No Fault Identified'].copy()

y_pred_sev = pipeline_sev.predict(final_sev)
final_sev['Severidad_Predicha']  = encoder_sev.inverse_transform(y_pred_sev)
final_norm['Severidad_Predicha'] = 'Normal'
final = pd.concat([final_sev, final_norm], ignore_index=True)

print(f"✅ Predicciones completadas: {len(final)} registros")
print(f"\nFault Effect:\n{final['Fault_Effect_Predicho'].value_counts()}")
print(f"\nSeveridad:\n{final['Severidad_Predicha'].value_counts()}")

# ==============================================================================
# 6. EXTRAER RANDOM FOREST BASE DEL PIPELINE CALIBRADO
# ==============================================================================

def extraer_rf_base(pipeline):
    ultimo = pipeline[-1]
    if isinstance(ultimo, CalibratedClassifierCV):
        estimador = ultimo.calibrated_classifiers_[0].estimator
        if isinstance(estimador, SKPipeline):
            return estimador[-1]
        return estimador
    if hasattr(pipeline, 'named_steps') and 'clf' in pipeline.named_steps:
        clf = pipeline.named_steps['clf']
        if isinstance(clf, CalibratedClassifierCV):
            return clf.calibrated_classifiers_[0].estimator
        return clf
    return ultimo

rf_base = extraer_rf_base(pipeline_sev)
print(f"\n✅ Modelo base extraído: {type(rf_base).__name__}")

# ==============================================================================
# 7. PREPARAR X_SHAP — solo columnas numéricas, sin object
# ==============================================================================

# Usar final_sev: muestras con falla identificada (las que tienen severidad)
X_shap = final_sev.copy()

# Convertir SACODE a numérico
for col in ['SACODE_Salud','SACODE_Desgaste','SACODE_Contaminacion','SACODE_General']:
    if col in X_shap.columns:
        X_shap[col] = X_shap[col].map(ORDEN_MAP).fillna(0).astype(int)

# Eliminar columnas no-features
COLS_ELIMINAR = [
    'Fault_Effect_Predicho', 'Severidad_Predicha',
    'Assigned Condition Rating', 'ACR_Homologado', 'Fault Effect',
    'Rule Based Rating', 'Unnamed: 0', 'Condition Review Notes',
    'Further Recommendations', 'Action Summary', 'Observation Code',
    'Observation Type', 'Observation Date', 'Observation Rating',
    'Component ID', 'Asset ID', 'Location', 'Component Profile',
    'Lubricant', 'Connection Code', 'Reviewer', 'Reviewed',
    'Color del Refrigerante'
]
X_shap = X_shap.drop(
    columns=[c for c in COLS_ELIMINAR if c in X_shap.columns],
    errors='ignore'
)

# Eliminar cualquier columna object restante
cols_obj = X_shap.select_dtypes(include='object').columns.tolist()
if cols_obj:
    print(f"⚠️  Eliminando columnas object: {cols_obj}")
    X_shap = X_shap.drop(columns=cols_obj)

# Rellenar NaN con la mediana
X_shap = X_shap.fillna(X_shap.median(numeric_only=True))
X_shap = X_shap.reset_index(drop=True)
feature_names = X_shap.columns.tolist()

print(f"✅ X_shap shape   : {X_shap.shape}")
print(f"✅ N features     : {len(feature_names)}")
print(f"✅ Tipos          : {X_shap.dtypes.value_counts().to_dict()}")

#── 8. CALCULAR SHAP VALUES ───────────────────────────────────────────────────

# Muestra representativa (max 300 para velocidad)
N_MAX      = min(300, len(X_shap))
np.random.seed(42)
idx_sample = np.random.choice(len(X_shap), N_MAX, replace=False)

# ✅ X_sample SIN Flota — para SHAP (solo numéricas)
X_sample   = X_shap.iloc[idx_sample].reset_index(drop=True)
X_np       = X_sample.values.astype(float)

# ✅ X_sample CON Flota — para predict_proba del pipeline
X_sample_con_flota = final_sev.iloc[idx_sample].copy().reset_index(drop=True)

# Convertir SACODE a numérico también en X_sample_con_flota
for col in ['SACODE_Salud','SACODE_Desgaste','SACODE_Contaminacion','SACODE_General']:
    if col in X_sample_con_flota.columns:
        X_sample_con_flota[col] = X_sample_con_flota[col].map(ORDEN_MAP).fillna(0).astype(int)

# Eliminar columnas no-features del pipeline (excepto Flota)
COLS_ELIMINAR_PIPE = [
    'Fault_Effect_Predicho', 'Severidad_Predicha',
    'Assigned Condition Rating', 'ACR_Homologado', 'Fault Effect',
    'Rule Based Rating', 'Unnamed: 0', 'Condition Review Notes',
    'Further Recommendations', 'Action Summary', 'Observation Code',
    'Observation Type', 'Observation Date', 'Observation Rating',
    'Component ID', 'Asset ID', 'Location', 'Component Profile',
    'Lubricant', 'Connection Code', 'Reviewer', 'Reviewed',
    'Color del Refrigerante'
]
X_sample_con_flota = X_sample_con_flota.drop(
    columns=[c for c in COLS_ELIMINAR_PIPE if c in X_sample_con_flota.columns],
    errors='ignore'
)
# Eliminar columnas object excepto Flota
cols_obj_pipe = [c for c in X_sample_con_flota.select_dtypes(include='object').columns
                 if c != 'Flota']
if cols_obj_pipe:
    X_sample_con_flota = X_sample_con_flota.drop(columns=cols_obj_pipe)

X_sample_con_flota = X_sample_con_flota.fillna(X_sample_con_flota.median(numeric_only=True))

print(f"✅ X_sample (SHAP)     : {X_sample.shape} — sin Flota")
print(f"✅ X_sample_con_flota  : {X_sample_con_flota.shape} — con Flota")

print(f"\nCalculando SHAP values sobre {N_MAX} muestras...")

explainer   = shap.TreeExplainer(rf_base)
shap_values = explainer.shap_values(X_np)

print(f"✅ shap_values shape : {np.array(shap_values).shape}")
print(f"✅ Clases            : {encoder_sev.classes_}")

IDX_CRITICAL = np.where(encoder_sev.classes_ == 'Critical')[0][0]
print(f"✅ IDX_CRITICAL = {IDX_CRITICAL} → '{encoder_sev.classes_[IDX_CRITICAL]}'")

if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_critical = shap_values[:, :, IDX_CRITICAL]
    expected_val  = (explainer.expected_value[IDX_CRITICAL]
                     if hasattr(explainer.expected_value, '__len__')
                     else explainer.expected_value)
elif isinstance(shap_values, list):
    shap_critical = shap_values[IDX_CRITICAL]
    expected_val  = (explainer.expected_value[IDX_CRITICAL]
                     if hasattr(explainer.expected_value, '__len__')
                     else explainer.expected_value)
else:
    shap_critical = shap_values
    expected_val  = (explainer.expected_value
                     if np.isscalar(explainer.expected_value)
                     else explainer.expected_value[0])

print(f"✅ shap_critical shape: {shap_critical.shape}")
print(f"✅ expected_val       : {float(expected_val):.4f}")

# ── 9. TABLA DE IMPORTANCIA ───────────────────────────────────────────────────

df_importancia = pd.DataFrame({
    'Feature'        : feature_names,
    'SHAP_abs_medio' : np.abs(shap_critical).mean(axis=0).tolist(),
    'SHAP_medio'     : shap_critical.mean(axis=0).tolist(),
}).sort_values('SHAP_abs_medio', ascending=False).reset_index(drop=True)

df_importancia['Dirección'] = df_importancia['SHAP_medio'].apply(
    lambda v: '▲ Aumenta P(Critical)' if v > 0 else '▼ Reduce P(Critical)'
)

print("\n=== Top 15 Features — Impacto en Clase Critical ===")
print(df_importancia.head(15).round(4).to_string(index=False))

# ── 10. VISUALIZACIONES SHAP ──────────────────────────────────────────────────

# Plot 1: Summary Dot Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values   = shap_critical,
    features      = X_np,
    feature_names = feature_names,
    max_display   = 15,
    show          = False,
    plot_type     = 'dot'
)
plt.title('SHAP Values — Clase Critical\nModelo de Severidad (Random Forest)',
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('shap_summary_dot_severidad.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ shap_summary_dot_severidad.png")

# Plot 2: Bar Plot
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values   = shap_critical,
    features      = X_np,
    feature_names = feature_names,
    max_display   = 15,
    show          = False,
    plot_type     = 'bar'
)
plt.title('Importancia de Features — |SHAP| medio\nModelo de Severidad — Clase Critical',
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('shap_barplot_severidad.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ shap_barplot_severidad.png")

# Plot 3: Dependence plots — Top 3
top3     = df_importancia['Feature'].head(3).tolist()
top3_idx = [feature_names.index(f) for f in top3 if f in feature_names]

fig, axes = plt.subplots(1, len(top3_idx), figsize=(6*len(top3_idx), 5))
if len(top3_idx) == 1:
    axes = [axes]

for ax, fidx in zip(axes, top3_idx):
    shap.dependence_plot(
        ind           = fidx,
        shap_values   = shap_critical,
        features      = X_np,
        feature_names = feature_names,
        ax            = ax,
        show          = False
    )
    ax.set_title(feature_names[fidx], fontsize=10, fontweight='bold')

plt.suptitle('Relación Feature → SHAP Value — Top 3 (Clase Critical)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_dependence_top3_severidad.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ shap_dependence_top3_severidad.png")

# ==============================================================================
# CORRECCIÓN — Agregar la feature faltante a X_np
# El RF fue entrenado con Flota codificada numéricamente
# ==============================================================================

# Verificar cuál feature falta
if hasattr(rf_base, 'feature_names_in_'):
    features_rf    = list(rf_base.feature_names_in_)
    features_shap  = feature_names

    faltantes = [f for f in features_rf if f not in features_shap]
    print(f"Features faltantes en X_np: {faltantes}")

    # Reconstruir X_shap_completo con las features que espera el RF
    X_shap_completo = final_sev.copy()

    # Convertir SACODE a numérico
    for col in ['SACODE_Salud','SACODE_Desgaste','SACODE_Contaminacion','SACODE_General']:
        if col in X_shap_completo.columns:
            X_shap_completo[col] = X_shap_completo[col].map(ORDEN_MAP).fillna(0).astype(int)

    # Codificar Flota a numérico si es la que falta
    if 'Flota' in faltantes and 'Flota' in X_shap_completo.columns:
        flota_map = {'Flota_1': 1, 'Flota_2': 2, 'Flota_3': 3}
        X_shap_completo['Flota'] = X_shap_completo['Flota'].map(flota_map).fillna(0).astype(int)
        print("✅ Flota codificada: Flota_1=1, Flota_2=2, Flota_3=3")

    # Seleccionar exactamente las columnas que espera el RF en el orden correcto
    cols_disponibles = [c for c in features_rf if c in X_shap_completo.columns]
    cols_faltantes   = [c for c in features_rf if c not in X_shap_completo.columns]

    if cols_faltantes:
        print(f"⚠️  Columnas no recuperables: {cols_faltantes} — rellenando con 0")
        for c in cols_faltantes:
            X_shap_completo[c] = 0

    X_shap_completo = X_shap_completo[features_rf].copy()
    X_shap_completo = X_shap_completo.fillna(X_shap_completo.median(numeric_only=True))

    print(f"✅ X_shap_completo shape: {X_shap_completo.shape}")
    print(f"✅ Columnas coinciden con RF: {list(X_shap_completo.columns) == features_rf}")

    # Recalcular muestra con las features completas
    X_sample_completo = X_shap_completo.iloc[idx_sample].reset_index(drop=True)
    X_np_completo     = X_sample_completo.values.astype(float)

    # Recalcular SHAP con las features completas
    print("\nRecalculando SHAP values con features completas...")
    shap_values_v2  = explainer.shap_values(X_np_completo)

    if isinstance(shap_values_v2, np.ndarray) and shap_values_v2.ndim == 3:
        shap_critical_v2 = shap_values_v2[:, :, IDX_CRITICAL]
    elif isinstance(shap_values_v2, list):
        shap_critical_v2 = shap_values_v2[IDX_CRITICAL]
    else:
        shap_critical_v2 = shap_values_v2

    feature_names_v2 = features_rf
    print(f"✅ shap_critical_v2 shape: {shap_critical_v2.shape}")

    # Tabla de importancia actualizada
    df_importancia = pd.DataFrame({
        'Feature'        : feature_names_v2,
        'SHAP_abs_medio' : np.abs(shap_critical_v2).mean(axis=0).tolist(),
        'SHAP_medio'     : shap_critical_v2.mean(axis=0).tolist(),
    }).sort_values('SHAP_abs_medio', ascending=False).reset_index(drop=True)

    df_importancia['Dirección'] = df_importancia['SHAP_medio'].apply(
        lambda v: '▲ Aumenta P(Critical)' if v > 0 else '▼ Reduce P(Critical)'
    )
    print("\n=== Top 15 Features (actualizado) ===")
    print(df_importancia.head(15).round(4).to_string(index=False))

    # Reasignar variables para los plots
    shap_critical = shap_critical_v2
    feature_names = feature_names_v2
    X_np          = X_np_completo

else:
    print("⚠️  rf_base no tiene feature_names_in_")
    print("   Agregando columna de ceros como feature extra")
    X_np_completo = np.hstack([X_np, np.zeros((X_np.shape[0], 1))])
    X_np          = X_np_completo

# ==============================================================================
# Plot 4: Waterfall — ahora con X_np correcto
# ==============================================================================

proba_sample_base = rf_base.predict_proba(X_np)
proba_critical    = proba_sample_base[:, IDX_CRITICAL]
idx_top           = int(np.argmax(proba_critical))

print(f"\n  Muestra con mayor P(Critical): idx={idx_top}, "
      f"P(Critical)={proba_critical[idx_top]:.4f}")

plt.figure(figsize=(12, 5))
shap.waterfall_plot(
    shap.Explanation(
        values        = shap_critical[idx_top],
        base_values   = float(expected_val),
        data          = X_np[idx_top],
        feature_names = feature_names
    ),
    max_display = 12,
    show        = False
)
plt.title(f'Explicación local — mayor P(Critical): {proba_critical[idx_top]:.4f}',
          fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_waterfall_critical.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ shap_waterfall_critical.png")

# ==============================================================================
# CORRECCIÓN DEFINITIVA — Recalcular todo desde cero con features alineadas
# ==============================================================================

# ── Paso 1: Obtener features exactas del RF ───────────────────────────────────
n_features_rf = rf_base.n_features_in_
print(f"RF espera          : {n_features_rf} features")

if hasattr(rf_base, 'feature_names_in_'):
    features_rf = list(rf_base.feature_names_in_)
    print(f"Nombres RF         : {features_rf}")
else:
    print("⚠️  RF sin feature_names_in_ — inferir por conteo")
    features_rf = None

# ── Paso 2: Construir X con EXACTAMENTE las features del RF ──────────────────
X_base = final_sev.copy()

# Codificar Flota
flota_map = {'Flota_1': 1, 'Flota_2': 2, 'Flota_3': 3}
if 'Flota' in X_base.columns:
    X_base['Flota'] = X_base['Flota'].map(flota_map).fillna(0).astype(int)

# Codificar SACODE
for col in ['SACODE_Salud','SACODE_Desgaste','SACODE_Contaminacion','SACODE_General']:
    if col in X_base.columns:
        X_base[col] = X_base[col].map(ORDEN_MAP).fillna(0).astype(int)

# Eliminar columnas no-features
COLS_ELIMINAR_TOTAL = [
    'Fault_Effect_Predicho', 'Severidad_Predicha',
    'Assigned Condition Rating', 'ACR_Homologado', 'Fault Effect',
    'Rule Based Rating', 'Unnamed: 0', 'Condition Review Notes',
    'Further Recommendations', 'Action Summary', 'Observation Code',
    'Observation Type', 'Observation Date', 'Observation Rating',
    'Component ID', 'Asset ID', 'Location', 'Component Profile',
    'Lubricant', 'Connection Code', 'Reviewer', 'Reviewed',
    'Color del Refrigerante'
]
X_base = X_base.drop(
    columns=[c for c in COLS_ELIMINAR_TOTAL if c in X_base.columns],
    errors='ignore'
)

# Eliminar columnas object restantes
cols_obj = X_base.select_dtypes(include='object').columns.tolist()
if cols_obj:
    print(f"⚠️  Eliminando object: {cols_obj}")
    X_base = X_base.drop(columns=cols_obj)

X_base = X_base.fillna(X_base.median(numeric_only=True)).reset_index(drop=True)

# ── Paso 3: Seleccionar/ordenar columnas según el RF ─────────────────────────
if features_rf is not None:
    # Agregar columnas faltantes con 0
    for c in features_rf:
        if c not in X_base.columns:
            print(f"  Agregando columna faltante con 0: {c}")
            X_base[c] = 0
    # Seleccionar en el orden exacto del RF
    X_base        = X_base[features_rf].copy()
    feature_names = features_rf
else:
    # Sin nombres — usar las primeras n_features_rf columnas
    X_base        = X_base.iloc[:, :n_features_rf].copy()
    feature_names = X_base.columns.tolist()

print(f"✅ X_base shape     : {X_base.shape}")
print(f"✅ feature_names    : {len(feature_names)}")
print(f"✅ RF espera        : {n_features_rf}")
assert X_base.shape[1] == n_features_rf, "❌ Sigue habiendo desajuste de features"

# ── Paso 4: Muestra para SHAP ─────────────────────────────────────────────────
N_MAX      = min(300, len(X_base))
np.random.seed(42)
idx_sample = np.random.choice(len(X_base), N_MAX, replace=False)
X_sample   = X_base.iloc[idx_sample].reset_index(drop=True)
X_np       = X_sample.values.astype(float)

print(f"✅ X_np shape       : {X_np.shape}")

# ── Paso 5: Calcular SHAP ─────────────────────────────────────────────────────
print("\nCalculando SHAP values...")
explainer   = shap.TreeExplainer(rf_base)
shap_values = explainer.shap_values(X_np)

print(f"shap_values shape  : {np.array(shap_values).shape}")

IDX_CRITICAL = np.where(encoder_sev.classes_ == 'Critical')[0][0]

if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_critical = shap_values[:, :, IDX_CRITICAL]
    expected_val  = (explainer.expected_value[IDX_CRITICAL]
                     if hasattr(explainer.expected_value, '__len__')
                     else explainer.expected_value)
elif isinstance(shap_values, list):
    shap_critical = shap_values[IDX_CRITICAL]
    expected_val  = (explainer.expected_value[IDX_CRITICAL]
                     if hasattr(explainer.expected_value, '__len__')
                     else explainer.expected_value)
else:
    shap_critical = shap_values
    expected_val  = (explainer.expected_value
                     if np.isscalar(explainer.expected_value)
                     else explainer.expected_value[0])

# ── Verificación final antes de los plots ─────────────────────────────────────
print(f"\n{'='*50}")
print(f"shap_critical : {shap_critical.shape}")
print(f"X_np          : {X_np.shape}")
print(f"feature_names : {len(feature_names)}")
assert shap_critical.shape[1] == X_np.shape[1] == len(feature_names), \
    f"❌ Desajuste: shap={shap_critical.shape[1]}, X={X_np.shape[1]}, f={len(feature_names)}"
print("✅ Shapes alineados — procediendo con los plots")

# ── Paso 6: Tabla de importancia ──────────────────────────────────────────────
df_importancia = pd.DataFrame({
    'Feature'        : feature_names,
    'SHAP_abs_medio' : np.abs(shap_critical).mean(axis=0).tolist(),
    'SHAP_medio'     : shap_critical.mean(axis=0).tolist(),
}).sort_values('SHAP_abs_medio', ascending=False).reset_index(drop=True)

df_importancia['Dirección'] = df_importancia['SHAP_medio'].apply(
    lambda v: '▲ Aumenta P(Critical)' if v > 0 else '▼ Reduce P(Critical)'
)
print("\n=== Top 15 Features — Impacto en Clase Critical ===")
print(df_importancia.head(15).round(4).to_string(index=False))

# ── Paso 7: Visualizaciones ───────────────────────────────────────────────────

# Plot 1 — Summary Dot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_critical, X_np, feature_names=feature_names,
                  max_display=15, show=False, plot_type='dot')
plt.title('SHAP Values — Clase Critical\nModelo de Severidad (Random Forest)',
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('shap_summary_dot_severidad.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ shap_summary_dot_severidad.png")

# Plot 2 — Bar
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_critical, X_np, feature_names=feature_names,
                  max_display=15, show=False, plot_type='bar')
plt.title('Importancia de Features — |SHAP| medio\nModelo de Severidad — Clase Critical',
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('shap_barplot_severidad.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ shap_barplot_severidad.png")

# Plot 3 — Dependence Top 3
top3     = df_importancia['Feature'].head(3).tolist()
top3_idx = [feature_names.index(f) for f in top3 if f in feature_names]

fig, axes = plt.subplots(1, len(top3_idx), figsize=(6*len(top3_idx), 5))
if len(top3_idx) == 1:
    axes = [axes]
for ax, fidx in zip(axes, top3_idx):
    shap.dependence_plot(fidx, shap_critical, X_np,
                         feature_names=feature_names, ax=ax, show=False)
    ax.set_title(feature_names[fidx], fontsize=10, fontweight='bold')
plt.suptitle('Relación Feature → SHAP Value — Top 3 (Clase Critical)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_dependence_top3_severidad.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ shap_dependence_top3_severidad.png")

# Plot 4 — Waterfall
proba_base     = rf_base.predict_proba(X_np)
proba_critical = proba_base[:, IDX_CRITICAL]
idx_top        = int(np.argmax(proba_critical))

print(f"\n  Muestra con mayor P(Critical): idx={idx_top}, "
      f"P(Critical)={proba_critical[idx_top]:.4f}")

plt.figure(figsize=(12, 5))
shap.waterfall_plot(
    shap.Explanation(
        values        = shap_critical[idx_top],
        base_values   = float(expected_val),
        data          = X_np[idx_top],
        feature_names = feature_names
    ),
    max_display=12, show=False
)
plt.title(f'Explicación local — mayor P(Critical): {proba_critical[idx_top]:.4f}',
          fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_waterfall_critical.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ shap_waterfall_critical.png")

# ── Paso 8: Exportar ──────────────────────────────────────────────────────────
final.to_excel("dataset_final_con_prediccion.xlsx", index=False)
df_importancia.to_excel("shap_importancia_severidad.xlsx", index=False)

print("\n✅ dataset_final_con_prediccion.xlsx")
print("✅ shap_importancia_severidad.xlsx")
print(f"\nTop 5 variables que más impactan en clase Critical:")
for i, row in df_importancia.head(5).iterrows():
    print(f"  {i+1}. {row['Feature']:30s} | "
          f"SHAP abs: {row['SHAP_abs_medio']:.4f} | {row['Dirección']}")