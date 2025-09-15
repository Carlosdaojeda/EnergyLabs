from flask import Flask, request, render_template
import pandas as pd
import os
from src.pipeline.predict_pipeline import PredictPipeline
from src.exception import CustomException
import sys
import matplotlib
matplotlib.use('Agg')  # para servidor sin interfaz gráfica
import matplotlib.pyplot as plt
import io
import base64

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files.get('file')
        if not file:
            return "No se subió ningún archivo"

        filename = file.filename
        ext = os.path.splitext(filename)[-1].lower()

        # Leer archivo
        if ext == ".csv":
            df = pd.read_csv(file)
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(file)
        else:
            return "Formato no soportado. Usa CSV o Excel."

        # Columnas requeridas
        model_required_cols = ['RHOB', 'GR', 'NPHI', 'PEF']
        plot_required_cols = ['DEPTH']

        missing_model_cols = [c for c in model_required_cols if c not in df.columns]
        missing_plot_cols = [c for c in plot_required_cols if c not in df.columns]

        if missing_model_cols:
            return f"Faltan columnas requeridas para el modelo: {', '.join(missing_model_cols)}"
        if missing_plot_cols:
            return f"Faltan columnas requeridas para graficar: {', '.join(missing_plot_cols)}"

        # Predicciones
        predict_pipeline = PredictPipeline()
        df['Predicted_DT'] = predict_pipeline.predict(df[model_required_cols])

        # Renombrar columna real si existe
        if 'DT' in df.columns:
            df = df.rename(columns={'DT': 'Real_DT'})

        # --- Graficar log panel estilo petrofísico ---
        df = df.sort_values(by='DEPTH')  # asegurar que DEPTH vaya de arriba hacia abajo

        petro_logs = ['GR', 'NPHI', 'RHOB', 'PEF']
        dt_logs = ['Real_DT', 'Predicted_DT']
        colors_petro = ['green', 'orange', 'brown', 'purple']
        colors_dt = ['blue', 'red']

        all_logs = petro_logs + dt_logs
        all_colors = colors_petro + colors_dt

        fig, axes = plt.subplots(nrows=1, ncols=len(all_logs), figsize=(18, 10), sharey=True)
        fig.subplots_adjust(wspace=0.05)

        for ax, log, color in zip(axes, all_logs, all_colors):
            if log in df.columns:
                ax.plot(df[log], df['DEPTH'], label=log, color=color)
                ax.set_xlabel(log)
                ax.grid(True)
                ax.legend(fontsize=8)

        axes[0].set_ylabel('DEPTH (m)')
        axes[0].invert_yaxis()  # invertir eje y una sola vez
        plt.tight_layout()

        # Convertir gráfico a base64 para HTML
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)

        return render_template(
            "results.html",
            graph_url=graph_url
        )

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
