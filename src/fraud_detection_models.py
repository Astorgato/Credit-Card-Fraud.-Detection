"""
Sistema avanzado de detección de fraude en tarjetas de crédito.

Este módulo implementa un pipeline completo de Machine Learning para la detección
de transacciones fraudulentas, optimizado para datasets con clases altamente 
desbalanceadas.

Características principales:
- Manejo experto de clases desbalanceadas (ratio 1:580)
- Comparación de múltiples algoritmos de ML
- Métricas especializadas para detección de fraude
- Análisis de ROI y impacto de negocio
- Código production-ready con logging y error handling

Autor: Data Science Professional
Fecha: Octubre 2025
Versión: 1.0.0
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score,
    precision_recall_curve, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetectionPipeline:
    """
    Pipeline completo para detección de fraude en tarjetas de crédito.

    Esta clase implementa un sistema end-to-end que incluye:
    - Carga y validación de datos
    - Preprocesamiento optimizado para clases desbalanceadas
    - Entrenamiento de múltiples algoritmos de ML
    - Evaluación con métricas especializadas
    - Análisis de impacto de negocio y ROI

    Attributes:
        random_state (int): Semilla para reproducibilidad
        scaler (StandardScaler): Escalador de características
        models (Dict): Diccionario de modelos entrenados
        results (Dict): Resultados de evaluación de modelos
        best_model (str): Nombre del mejor modelo

    Example:
        >>> pipeline = FraudDetectionPipeline(random_state=42)
        >>> X, y = pipeline.load_data('data/credit_card_fraud_dataset.csv')
        >>> pipeline.prepare_data(X, y)
        >>> pipeline.train_models()
        >>> best_model = pipeline.get_best_model()
        >>> report = pipeline.generate_report()
    """

    def __init__(self, random_state: int = 42):
        """
        Inicializar el pipeline de detección de fraude.

        Args:
            random_state: Semilla para garantizar reproducibilidad
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None

        logger.info(f"Pipeline inicializado con random_state={random_state}")

    def load_data(self, file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Cargar y validar dataset de transacciones de tarjetas de crédito.

        Args:
            file_path: Ruta al archivo CSV

        Returns:
            Tupla con características (X) y variable objetivo (y)

        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el dataset no tiene el formato esperado
        """
        try:
            logger.info(f"Cargando dataset desde {file_path}")
            df = pd.read_csv(file_path)

            # Validaciones básicas
            if 'Class' not in df.columns:
                raise ValueError("Dataset debe contener columna 'Class' como variable objetivo")

            if df.empty:
                raise ValueError("Dataset está vacío")

            # Separar características y variable objetivo
            X = df.drop('Class', axis=1)
            y = df['Class']

            # Estadísticas del dataset
            fraud_rate = y.mean()
            total_transactions = len(df)
            fraud_count = y.sum()

            logger.info(f"Dataset cargado exitosamente:")
            logger.info(f"  - Total transacciones: {total_transactions:,}")
            logger.info(f"  - Transacciones fraudulentas: {fraud_count:,}")
            logger.info(f"  - Tasa de fraude: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")
            logger.info(f"  - Ratio de desbalance: 1:{int(1/fraud_rate):,}")
            logger.info(f"  - Características: {X.shape[1]}")

            return X, y

        except Exception as e:
            logger.error(f"Error cargando dataset: {str(e)}")
            raise

    def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """
        Preparar datos para entrenamiento con validación estratificada.

        Args:
            X: Características
            y: Variable objetivo
            test_size: Proporción de datos para test

        Returns:
            Tupla con datos divididos y escalados
        """
        try:
            logger.info("Preparando datos para entrenamiento...")

            # División estratificada para preservar proporción de fraudes
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=self.random_state, 
                stratify=y
            )

            # Escalado de características
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)

            # Estadísticas de división
            train_fraud_rate = self.y_train.mean()
            test_fraud_rate = self.y_test.mean()

            logger.info(f"Datos preparados:")
            logger.info(f"  - Train: {len(self.X_train):,} muestras ({self.y_train.sum():,} fraudes)")
            logger.info(f"  - Test: {len(self.X_test):,} muestras ({self.y_test.sum():,} fraudes)")
            logger.info(f"  - Tasa fraude train: {train_fraud_rate:.4f}")
            logger.info(f"  - Tasa fraude test: {test_fraud_rate:.4f}")
            logger.info(f"  - Características escaladas correctamente")

            return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test

        except Exception as e:
            logger.error(f"Error preparando datos: {str(e)}")
            raise

    def train_models(self) -> None:
        """
        Entrenar múltiples algoritmos optimizados para detección de fraude.

        Los modelos están configurados específicamente para manejar clases
        desbalanceadas utilizando class_weight y otras técnicas.
        """
        try:
            logger.info("Iniciando entrenamiento de modelos...")

            # Definir modelos optimizados para fraude
            self.models = {
                'Logistic Regression': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight='balanced',
                    solver='liblinear'
                ),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    class_weight='balanced',
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    learning_rate=0.1,
                    max_depth=6
                ),
                'SVM': SVC(
                    kernel='rbf',
                    random_state=self.random_state,
                    probability=True,
                    class_weight='balanced',
                    gamma='scale'
                ),
                'Naive Bayes': GaussianNB()
            }

            # Entrenar cada modelo
            for name, model in self.models.items():
                logger.info(f"Entrenando {name}...")

                # Seleccionar datos apropiados para el modelo
                if name in ['Logistic Regression', 'SVM', 'Naive Bayes']:
                    X_train_use = self.X_train_scaled
                    X_test_use = self.X_test_scaled
                else:
                    X_train_use = self.X_train
                    X_test_use = self.X_test

                # Entrenar modelo
                model.fit(X_train_use, self.y_train)

                # Predicciones
                y_pred = model.predict(X_test_use)
                y_pred_proba = (model.predict_proba(X_test_use)[:, 1] 
                               if hasattr(model, 'predict_proba') else None)

                # Calcular métricas
                metrics = self._calculate_metrics(y_pred, y_pred_proba)

                # Validación cruzada
                cv_scores = cross_val_score(
                    model, X_train_use, self.y_train,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                    scoring='f1'
                )

                self.results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'cv_f1_mean': cv_scores.mean(),
                    'cv_f1_std': cv_scores.std()
                }

                logger.info(f"  {name} completado - F1: {metrics['f1_score']:.4f}")

            logger.info("Entrenamiento de modelos completado exitosamente")

        except Exception as e:
            logger.error(f"Error entrenando modelos: {str(e)}")
            raise

    def _calculate_metrics(self, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Calcular métricas de evaluación especializadas para detección de fraude.

        Args:
            y_pred: Predicciones binarias
            y_pred_proba: Probabilidades de la clase positiva

        Returns:
            Diccionario con métricas calculadas
        """
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1_score': f1_score(self.y_test, y_pred, zero_division=0)
        }

        # ROC-AUC solo si hay probabilidades
        if y_pred_proba is not None and len(np.unique(self.y_test)) > 1:
            metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)
        else:
            metrics['roc_auc'] = 0.5

        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Caso edge donde solo se predice una clase
            if y_pred.sum() == 0:  # Solo predice clase 0
                tn, fp, fn, tp = len(self.y_test) - sum(self.y_test), 0, sum(self.y_test), 0
            else:  # Solo predice clase 1
                tn, fp, fn, tp = 0, len(self.y_test) - sum(self.y_test), 0, sum(self.y_test)

        metrics.update({'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})

        return metrics

    def get_best_model(self) -> Tuple[str, float]:
        """
        Identificar el mejor modelo basado en F1-Score.

        Para detección de fraude, F1-Score es la métrica más apropiada
        porque balancea precision y recall, ambos críticos para minimizar
        tanto falsos positivos como falsos negativos.

        Returns:
            Tupla con nombre del mejor modelo y su F1-Score
        """
        if not self.results:
            raise ValueError("No hay modelos entrenados. Ejecuta train_models() primero.")

        best_f1 = 0
        best_name = None

        for name, result in self.results.items():
            f1 = result['metrics']['f1_score']
            if f1 > best_f1:
                best_f1 = f1
                best_name = name

        self.best_model = best_name
        logger.info(f"Mejor modelo identificado: {best_name} (F1-Score: {best_f1:.4f})")

        return best_name, best_f1

    def calculate_business_impact(self, avg_fraud_amount: float = 700.0, 
                                 fp_investigation_cost: float = 25.0) -> Dict:
        """
        Calcular el impacto de negocio y ROI del mejor modelo.

        Args:
            avg_fraud_amount: Monto promedio de transacción fraudulenta
            fp_investigation_cost: Costo de investigar un falso positivo

        Returns:
            Diccionario con métricas de negocio
        """
        if not self.best_model:
            self.get_best_model()

        best_metrics = self.results[self.best_model]['metrics']
        tp, fp, fn, tn = best_metrics['tp'], best_metrics['fp'], best_metrics['fn'], best_metrics['tn']

        # Cálculos financieros
        fraud_prevented = tp * avg_fraud_amount
        fraud_missed = fn * avg_fraud_amount
        investigation_costs = fp * fp_investigation_cost
        net_monthly_benefit = fraud_prevented - investigation_costs
        net_annual_benefit = net_monthly_benefit * 12

        # Métricas de negocio
        business_metrics = {
            'model_name': self.best_model,
            'detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_alarm_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'fraud_prevented_amount': fraud_prevented,
            'fraud_missed_amount': fraud_missed,
            'investigation_costs': investigation_costs,
            'net_monthly_benefit': net_monthly_benefit,
            'net_annual_benefit': net_annual_benefit,
            'precision': best_metrics['precision'],
            'recall': best_metrics['recall'],
            'f1_score': best_metrics['f1_score'],
            'roc_auc': best_metrics['roc_auc']
        }

        logger.info(f"Impacto de negocio calculado para {self.best_model}")
        logger.info(f"  - Fraude evitado: ${fraud_prevented:,.2f}")
        logger.info(f"  - Beneficio neto mensual: ${net_monthly_benefit:,.2f}")
        logger.info(f"  - Tasa de detección: {business_metrics['detection_rate']:.1%}")

        return business_metrics

    def generate_report(self) -> pd.DataFrame:
        """
        Generar reporte completo de resultados.

        Returns:
            DataFrame con comparación de todos los modelos
        """
        if not self.results:
            raise ValueError("No hay resultados disponibles. Ejecuta train_models() primero.")

        logger.info("Generando reporte de resultados...")

        # Crear tabla comparativa
        report_data = []
        for name, result in self.results.items():
            metrics = result['metrics']
            report_data.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc'],
                'True_Positives': metrics['tp'],
                'False_Positives': metrics['fp'],
                'False_Negatives': metrics['fn'],
                'CV_F1_Mean': result['cv_f1_mean'],
                'CV_F1_Std': result['cv_f1_std']
            })

        df_report = pd.DataFrame(report_data)
        df_report = df_report.sort_values('F1-Score', ascending=False)

        # Mostrar reporte
        print("\n" + "="*80)
        print("REPORTE COMPLETO DE DETECCIÓN DE FRAUDE")
        print("="*80)
        print(df_report.round(4).to_string(index=False))

        # Identificar mejor modelo
        best_name, best_f1 = self.get_best_model()
        print(f"\n🏆 MEJOR MODELO: {best_name}")
        print(f"📊 F1-Score: {best_f1:.4f}")
        print(f"🎯 Ideal para detección de fraude por balance precision-recall")

        logger.info("Reporte generado exitosamente")

        return df_report


def main():
    """
    Función principal para demostrar el uso del pipeline.
    """
    print("🚀 SISTEMA DE DETECCIÓN DE FRAUDE EN TARJETAS DE CRÉDITO")
    print("="*60)
    print("Pipeline de Machine Learning para clasificación avanzada")
    print("Optimizado para datasets con clases altamente desbalanceadas\n")

    try:
        # Inicializar pipeline
        pipeline = FraudDetectionPipeline(random_state=42)

        # Ejemplo de uso (requiere dataset)
        print("📋 INSTRUCCIONES DE USO:")
        print("1. Coloca el dataset en: data/credit_card_fraud_dataset.csv")
        print("2. Ejecuta: python src/fraud_detection_models.py")
        print("3. Revisa resultados en: results/")
        print("\n💡 Para análisis interactivo, usa el Jupyter notebook:")
        print("   jupyter notebook notebooks/fraud_detection_analysis.ipynb")

    except Exception as e:
        logger.error(f"Error en ejecución principal: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
