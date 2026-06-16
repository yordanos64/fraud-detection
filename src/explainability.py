# src/explainability.py
import shap
import matplotlib.pyplot as plt

def generate_global_summary(model, X_transformed, feature_names, save_path):
    """Generates and saves the SHAP global features tree summary."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_transformed)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Verifiable SHAP global plot saved to: {save_path}")
