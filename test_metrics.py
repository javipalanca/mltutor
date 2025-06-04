#!/usr/bin/env python3
"""
Script para probar las métricas de regresión corregidas
"""

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np


def test_basic_metrics():
    """Prueba básica de métricas"""
    print("=== PRUEBA BÁSICA DE MÉTRICAS ===")

    # Datos de prueba simples
    y_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

    print(f"y_test: {y_test}")
    print(f"y_pred: {y_pred}")

    # Calcular métricas con sklearn
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return r2, mae, mse, rmse


def test_with_propinas_data():
    """Prueba con datos reales de propinas"""
    print("\n=== PRUEBA CON DATOS DE PROPINAS ===")

    try:
        from dataset_manager import load_data
        from model_training import train_linear_model
        from model_evaluation import evaluate_regression_model

        # Cargar datos de propinas
        X, y, feature_names, class_names, dataset_info, task_type = load_data(
            '💰 Propinas - Predicción de propinas')
        print(
            f"Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")

        # Entrenar modelo
        result = train_linear_model(
            X, y, model_type='Linear', test_size=0.2, random_state=42)
        print(f"Modelo entrenado exitosamente")

        # Métricas desde train_linear_model
        train_r2 = result["test_results"]["r2"]
        train_mae = result["test_results"]["mae"]
        train_rmse = result["test_results"]["rmse"]

        print(f"Métricas desde train_linear_model:")
        print(f"  R² Score: {train_r2:.4f}")
        print(f"  MAE: {train_mae:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")

        # Métricas desde evaluate_regression_model
        eval_result = evaluate_regression_model(
            result['y_test'], result['y_pred'])
        eval_r2 = eval_result["r2"]
        eval_mae = eval_result["mae"]
        eval_rmse = eval_result["rmse"]

        print(f"Métricas desde evaluate_regression_model:")
        print(f"  R² Score: {eval_r2:.4f}")
        print(f"  MAE: {eval_mae:.4f}")
        print(f"  RMSE: {eval_rmse:.4f}")

        # Verificar consistencia
        r2_diff = abs(train_r2 - eval_r2)
        mae_diff = abs(train_mae - eval_mae)
        rmse_diff = abs(train_rmse - eval_rmse)

        print(f"\nDiferencias entre métodos:")
        print(f"  R² diff: {r2_diff:.6f}")
        print(f"  MAE diff: {mae_diff:.6f}")
        print(f"  RMSE diff: {rmse_diff:.6f}")

        if r2_diff < 0.001 and mae_diff < 0.001 and rmse_diff < 0.001:
            print("✅ Las métricas son consistentes!")
        else:
            print("❌ Hay inconsistencias en las métricas")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Ejecutar pruebas
    test_basic_metrics()
    test_with_propinas_data()
