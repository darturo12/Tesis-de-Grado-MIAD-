# Proyecto Aplicado de Analítica de Datos - OilSense ML

Este repositorio contiene notebooks y código para análisis de datos y desarrollo de modelos de machine learning para la generación de un modelo de clasificación y analisis de resultados de muestras de aceite usado de motores de maquinaría pesada para minería.

---

## 🧩 Requisitos

- Python 3.11.3 (recomendado)
- Git

---

## ⚙️ Configuración del entorno

### 1. Clonar el repositorio
```bash
git clone https://github.com/darturo12/Tesis-de-Grado-MIAD-.git
cd repositorio
```
### 2. Crear entorno virtual
```bash
python -m venv venv
```
### 3. Activar entorno
Windows
```bash
venv\Scripts\activate
```
Mac/Linux
```bash
source venv/bin/activate
```
### 4. Instalar dependencias
```bash
pip install -r requirements.txt
```
### 5. Configurar kernel para Jupyter (recomendado)
```bash
pip install ipykernel
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```
Luego, en el notebook, seleccionar el kernel Python (venv).

---

## ▶️ Uso del proyecto

Los notebooks se encuentran en la carpeta correspondiente (por ejemplo: Modelos_ML/)

Ejecutar las celdas en orden

Verificar que el kernel activo sea el correcto (venv)

---

## 🔁 Actualización de dependencias

Si se agregan nuevas librerías se debe actualizar el archivo de requirements.txt, asi:
```bash
pip install nombre_libreria
pip freeze > requirements.txt
```
Luego actualizar el repositorio:
```bash
git add requirements.txt
git commit -m "Actualiza dependencias"
git push
```

---

## 🚫 Archivos ignorados

Este repositorio no incluye:

Entornos virtuales (venv/)

Archivos temporales (__pycache__/)

Checkpoints de Jupyter (.ipynb_checkpoints/)

---

## 📌 Buenas prácticas

No trabajar directamente sobre la rama principal (master)
Crear una rama para cada cambio o desarrollo
Hacer commits claros y frecuentes
Mantener actualizado el archivo requirements.txt

---

## 👥 Flujo de trabajo colaborativo

Crear una nueva rama:
```bash
git checkout -b mi-rama
```
Realizar cambios y guardar:
```bash
git add .
git commit -m "Descripción del cambio"
```
Subir la rama:
```bash
git push origin mi-rama
```
Crear un Pull Request en GitHub

---

## ⚠️ Solución de problemas comunes

Error: módulos no encontrados
Verificar que el entorno virtual esté activado
Confirmar que se instalaron las dependencias correctamente
Validar que el notebook esté usando el kernel correcto
Validar entorno activo

---

## Dentro de Python:
```bash
import sys
print(sys.executable)
```
Debe apuntar al entorno venv.

---

## 📁 Estructura del proyecto

```text
repo/
│
├── Data/
├── Modelos_ML/
├── Exploración/
├── Exploración Cerrejon/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📌 Notas finales

Este proyecto está diseñado para ser reproducible.
Cualquier usuario debería poder ejecutar el código siguiendo los pasos anteriores sin configuraciones adicionales.
