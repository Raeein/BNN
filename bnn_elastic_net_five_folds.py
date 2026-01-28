"""
Elastic Net Feature Selection + BNN Cross-Validation
All-in-one script: Elastic Net feature selection followed by BNN CV
"""

import time
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from numpyro import sample, plate, deterministic
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.preprocessing import StandardScaler
from functools import partial
import matplotlib.pyplot as plt

import numpyro
import warnings
warnings.filterwarnings('ignore')

import sys
sys.stdout.reconfigure(line_buffering=True)


def run_elasticnet_feature_selection(phenotype_name, 
                                       alpha=None,
                                       l1_ratio=0.5,
                                       cv_folds=5,
                                       n_alphas=100,
                                       coef_threshold=1e-6,
                                       n_cv_splits=5):
    """Run Elastic Net feature selection and create CV splits"""
    
    print(f"\n{'='*70}")
    print(f"Elastic Net Feature Selection for {phenotype_name}")
    print(f"{'='*70}\n")
    
    print("Loading data")
    filename = f"Geno_merged_{phenotype_name}_df_prop.csv"
    df = pd.read_csv(filename)
    print(f"Data loaded: {df.shape}")
    
    feature_start_idx = 6
    feature_cols = df.columns[feature_start_idx:-1]
    
    df_final = df[list(feature_cols) + ['y']].copy()
    
    print(f"\nPreparing data")
    for col in feature_cols:
        df_final[col] = df_final[col].astype(str)
    
    df_final = df_final.sample(frac=1, random_state=100).reset_index(drop=True)
    n = df_final.shape[0]
    print(f"Samples: {n}, Features: {len(feature_cols)}")
    
    print(f"\nEncoding SNPs for Elastic Net")
    t_encode = time.time()
    
    X_encoded = np.zeros((n, len(feature_cols)))
    for i, col in enumerate(feature_cols):
        x = df_final[col].values
        x_enc = x.copy()
        x_enc[x_enc == 'A'] = 0
        x_enc[x_enc == 'T'] = 1
        x_enc[x_enc == 'G'] = 2
        x_enc[x_enc == 'C'] = 3
        X_encoded[:, i] = x_enc.astype(float)
        
        if (i + 1) % 500 == 0:
            print(f"  Encoded {i+1}/{len(feature_cols)} SNPs")
    
    y = df_final['y'].values
    encode_time = time.time() - t_encode
    print(f"Encoding completed in {encode_time:.2f}s")
    
    print(f"\nStandardizing features")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    print(f"\n{'='*70}")
    print("Elastic Net Feature Selection")
    print(f"{'='*70}\n")
    
    t0 = time.time()
    
    if alpha is None:
        print(f"Running {cv_folds}-fold CV to select alpha")
        print(f"Testing {n_alphas} alpha values")
        print(f"L1 ratio: {l1_ratio} (L1={l1_ratio}, L2={1-l1_ratio})")
        
        elasticnet_cv = ElasticNetCV(
            l1_ratio=l1_ratio,
            cv=cv_folds,
            n_alphas=n_alphas,
            alphas=None,
            max_iter=10000,
            random_state=100,
            n_jobs=-1,
            verbose=0
        )
        
        elasticnet_cv.fit(X_scaled, y)
        alpha_selected = elasticnet_cv.alpha_
        
        print(f"\nCV completed")
        print(f"  Best alpha: {alpha_selected:.6e}")
        print(f"  CV R²: {elasticnet_cv.score(X_scaled, y):.6f}")
        
        coefs = elasticnet_cv.coef_
        
    else:
        print(f"Using fixed alpha: {alpha:.6e}")
        print(f"L1 ratio: {l1_ratio} (L1={l1_ratio}, L2={1-l1_ratio})")
        elasticnet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=100)
        elasticnet.fit(X_scaled, y)
        alpha_selected = alpha
        coefs = elasticnet.coef_
        print(f"  Training R²: {elasticnet.score(X_scaled, y):.6f}")
    
    elapsed = time.time() - t0
    print(f"\nElastic Net completed in {elapsed:.2f}s")
    
    print(f"\n{'='*70}")
    print("Coefficient Analysis")
    print(f"{'='*70}\n")
    
    abs_coefs = np.abs(coefs)
    
    print("Coefficient magnitude distribution:")
    quantiles = np.quantile(abs_coefs, [0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    print(f"  Min:  {quantiles[0]:.6e}")
    print(f"  25%:  {quantiles[1]:.6e}")
    print(f"  50%:  {quantiles[2]:.6e}")
    print(f"  75%:  {quantiles[3]:.6e}")
    print(f"  95%:  {quantiles[4]:.6e}")
    print(f"  99%:  {quantiles[5]:.6e}")
    print(f"  Max:  {quantiles[6]:.6e}")
    
    print(f"\nNon-zero coefficients at different thresholds:")
    thresholds = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    for thresh in thresholds:
        count = np.sum(abs_coefs > thresh)
        print(f"  |coef| > {thresh:.0e}: {count} SNPs")
    
    print(f"\nSelecting SNPs with |coefficient| > {coef_threshold:.0e}")
    selected_mask = abs_coefs > coef_threshold
    selected_indices = np.where(selected_mask)[0]
    n_selected = len(selected_indices)
    
    print(f"Selected {n_selected} SNPs ({n_selected/len(feature_cols)*100:.2f}% of total)")
    
    if n_selected == 0:
        print("\nWARNING: No SNPs selected")
        print("  Consider lowering coef_threshold or decreasing alpha")
        return None
    
    print(f"\nTop 10 most important SNPs:")
    top_indices = np.argsort(abs_coefs)[-10:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        snp_name = feature_cols[idx]
        coef_val = coefs[idx]
        print(f"  {rank:2d}. {snp_name:30s} coef = {coef_val:+.6e}")
    
    print(f"\n{'='*70}")
    print("Creating Filtered Dataset")
    print(f"{'='*70}\n")
    
    selected_cols = [feature_cols[i] for i in selected_indices]
    df_new = df_final[selected_cols + ['y']].copy()
    
    print(f"Filtered dataset dimensions: {df_new.shape}")
    
    print(f"\nEncoding filtered dataset")
    
    t_encode2 = time.time()
    for col in selected_cols:
        df_new[col] = df_new[col].replace('A', 0)
    print(f"A completed")
    
    for col in selected_cols:
        df_new[col] = df_new[col].replace('T', 1)
    print(f"T completed")
    
    for col in selected_cols:
        df_new[col] = df_new[col].replace('G', 2)
    print(f"G completed")
    
    for col in selected_cols:
        df_new[col] = df_new[col].replace('C', 3)
    print(f"C completed")
    
    for col in selected_cols:
        df_new[col] = pd.to_numeric(df_new[col])
    
    encode_time2 = time.time() - t_encode2
    print(f"Encoding completed in {encode_time2:.2f}s")
    
    output_file = f"{phenotype_name}_filtered_prop.csv"
    df_new.to_csv(output_file, index=False)
    print(f"\nSaved filtered data: {output_file}")
    
    print(f"\n{'='*70}")
    print(f"Creating {n_cv_splits}-Fold CV Splits")
    print(f"{'='*70}\n")
    
    nobs = df_new.shape[0]
    d1 = df_new.shape[1]
    
    kfold = KFold(n_splits=n_cv_splits, shuffle=True, random_state=100)
    
    fold_info = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(df_new), start=1):
        print(f"\nProcessing Fold {fold_idx}/{n_cv_splits}")
        
        trainData = df_new.iloc[train_idx].copy()
        testData = df_new.iloc[test_idx].copy()
        
        print(f"  Train: {len(trainData)} samples")
        print(f"  Test:  {len(testData)} samples")
        
        x_test = testData.iloc[:, 0:(d1-1)]
        x_test.to_csv(f"XTest_{phenotype_name}_fold{fold_idx}_prop.csv", index=True)
        
        y_test = testData.iloc[:, d1-1:d1]
        y_test.to_csv(f"ytest_{phenotype_name}_fold{fold_idx}_prop.csv", index=True)
        
        x_train = trainData.iloc[:, 0:(d1-1)]
        x_train.to_csv(f"XTrain_{phenotype_name}_fold{fold_idx}_prop.csv", index=True)
        
        y_train = trainData.iloc[:, d1-1:d1]
        y_train.to_csv(f"ytrain_{phenotype_name}_fold{fold_idx}_prop.csv", index=True)
        
        fold_info.append({
            'fold': fold_idx,
            'train_samples': len(trainData),
            'test_samples': len(testData)
        })
    
    print(f"\n{n_cv_splits}-fold split completed")
    
    summary_file = f"{phenotype_name}_elasticnet_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Elastic Net Feature Selection Summary - {phenotype_name}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total SNPs in original data: {len(feature_cols)}\n")
        f.write(f"Selected SNPs: {n_selected} ({n_selected/len(feature_cols)*100:.2f}%)\n")
        f.write(f"Alpha used: {alpha_selected:.6e}\n")
        f.write(f"L1 ratio: {l1_ratio}\n")
        f.write(f"Coefficient threshold: {coef_threshold:.6e}\n")
        f.write(f"Training time: {elapsed:.2f}s\n\n")
        
        f.write(f"Cross-Validation Splits: {n_cv_splits} folds\n")
        f.write("-"*70 + "\n")
        for info in fold_info:
            f.write(f"Fold {info['fold']}: Train={info['train_samples']}, Test={info['test_samples']}\n")
        f.write("\n")
        
        f.write("Top 20 most important SNPs:\n")
        top_20_indices = np.argsort(abs_coefs)[-20:][::-1]
        for rank, idx in enumerate(top_20_indices, 1):
            snp_name = feature_cols[idx]
            coef_val = coefs[idx]
            f.write(f"  {rank:2d}. {snp_name:30s} coef = {coef_val:+.6e}\n")
    
    print(f"\nSaved Elastic Net summary: {summary_file}")
    
    print(f"\n{'='*70}")
    print(f"Elastic Net Feature Selection Complete")
    print(f"{'='*70}")
    print(f"\nFiles saved:")
    print(f"  - {phenotype_name}_filtered_prop.csv")
    print(f"  - {phenotype_name}_elasticnet_summary.txt")
    
    print(f"\n{n_cv_splits}-Fold CV files created:")
    for fold_idx in range(1, n_cv_splits + 1):
        print(f"\n  Fold {fold_idx}:")
        print(f"    - XTrain_{phenotype_name}_fold{fold_idx}_prop.csv")
        print(f"    - XTest_{phenotype_name}_fold{fold_idx}_prop.csv")
        print(f"    - ytrain_{phenotype_name}_fold{fold_idx}_prop.csv")
        print(f"    - ytest_{phenotype_name}_fold{fold_idx}_prop.csv")
    
    print(f"\nSelected {n_selected} SNPs using Elastic Net")
    
    return {
        'alpha': alpha_selected,
        'l1_ratio': l1_ratio,
        'n_selected': n_selected,
        'coefs': coefs,
        'selected_indices': selected_indices,
        'selected_snps': selected_cols,
        'fold_info': fold_info
    }


def bnn_model(x, y=None, H=2, n_H=32, activation='selu'):
    """Bayesian Neural Network model"""
    N, P = x.shape

    d_t_h = sample("data_to_hidden_weights",
                   dist.Normal(0, 0.1).expand([P, n_H]).to_event(2))
    h_t_h = [sample(f"hidden_to_hidden_weights_{i}",
                    dist.Normal(0, 0.5).expand([n_H, n_H]).to_event(2))
             for i in range(H - 1)]
    h_t_d = sample("hidden_to_data_weights",
                   dist.Normal(0, 0.5).expand([n_H]))
    hidden_bias = [sample(f"hidden_bias_{i}",
                          dist.Normal(0, 1).expand([n_H]))
                   for i in range(H)]
    y_bias = sample("y_bias", dist.Normal(0, 1))
    sigma = sample("sigma", dist.HalfNormal(1.0))

    if activation == 'selu':
        act_fn = jax.nn.selu
    elif activation == 'elu':
        act_fn = jax.nn.elu
    elif activation == 'relu':
        act_fn = jax.nn.relu
    elif activation == 'tanh':
        act_fn = jnp.tanh
    else:
        act_fn = jax.nn.selu

    def forward(x_row):
        h = x_row @ d_t_h + hidden_bias[0]
        h = jnp.clip(h, -10.0, 10.0)
        h = act_fn(h)

        for i in range(1, H):
            h = h @ h_t_h[i - 1] + hidden_bias[i]
            h = jnp.clip(h, -10.0, 10.0)
            h = act_fn(h)

        out = h @ h_t_d + y_bias
        return jnp.clip(out, -1000.0, 1000.0)

    with plate("data", N):
        mu = jax.vmap(forward)(x)
        deterministic("mu", mu)
        if y is not None:
            sample("obs", dist.Normal(mu, sigma), obs=y)


def train_bnn(X_train, y_train, H=2, n_H=32, activation='selu',
              num_samples=1000, num_warmup=500, num_chains=2, seed=42,
              verbose=False):
    """Train BNN and return MCMC object and scaling parameters"""

    mu = X_train.mean(0)
    sd = X_train.std(0) + 1e-8
    X_scaled = (X_train - mu) / sd

    kernel = NUTS(
        partial(bnn_model, H=H, n_H=n_H, activation=activation),
        target_accept_prob=0.8,
        max_tree_depth=10,
        dense_mass=False
    )

    mcmc = MCMC(
        kernel,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=verbose
    )

    mcmc.run(jax.random.PRNGKey(seed), X_scaled, y_train)

    return mcmc, mu, sd


def predict_bnn(mcmc, X_test, mu, sd, H=2, n_H=32, activation='selu', seed=123):
    """Make predictions using trained BNN"""
    X_scaled = (X_test - mu) / sd

    predictive = Predictive(
        partial(bnn_model, H=H, n_H=n_H, activation=activation),
        posterior_samples=mcmc.get_samples(),
        return_sites=["mu"]
    )

    mu_samples = predictive(jax.random.PRNGKey(seed), X_scaled)["mu"]
    pred_mean = np.asarray(mu_samples.mean(axis=0))
    pred_std = np.asarray(mu_samples.std(axis=0))

    return pred_mean, pred_std


def run_cv_with_preselected_features(
    phenotype_name: str,
    n_folds: int = 5,
    H: int = 2,
    n_H: int = 32,
    activation: str = 'selu',
    num_samples: int = 1000,
    num_warmup: int = 500,
    num_chains: int = 2,
    seed: int = 42
):
    """Run cross-validation using pre-generated fold files"""

    print(f"\n{'='*70}")
    print(f"BNN Cross-Validation")
    print(f"Phenotype: {phenotype_name}")
    print(f"{'='*70}\n")

    results = {
        'fold_r2': [],
        'fold_rmse': [],
        'fold_mae': [],
        'fold_train_size': [],
        'fold_test_size': [],
        'predictions': [],
        'true_values': [],
        'pred_std': []
    }

    for fold_idx in range(1, n_folds + 1):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx}/{n_folds}")
        print(f"{'='*70}")

        try:
            X_train = pd.read_csv(f"XTrain_{phenotype_name}_fold{fold_idx}_prop.csv", index_col=0).values
            X_test = pd.read_csv(f"XTest_{phenotype_name}_fold{fold_idx}_prop.csv", index_col=0).values
            y_train = pd.read_csv(f"ytrain_{phenotype_name}_fold{fold_idx}_prop.csv", index_col=0).values.ravel()
            y_test = pd.read_csv(f"ytest_{phenotype_name}_fold{fold_idx}_prop.csv", index_col=0).values.ravel()

            print(f"Loaded fold {fold_idx} data")
            print(f"  Train: {len(y_train)} samples, {X_train.shape[1]} features")
            print(f"  Test: {len(y_test)} samples")

        except FileNotFoundError as e:
            print(f"Error loading fold {fold_idx} files: {e}")
            continue

        print(f"\nTraining BNN")
        t0 = time.time()
        
        try:
            mcmc, mu, sd = train_bnn(
                X_train, y_train,
                H=H, n_H=n_H, activation=activation,
                num_samples=num_samples, num_warmup=num_warmup,
                num_chains=num_chains, seed=seed+fold_idx,
                verbose=False
            )
            train_time = time.time() - t0
            print(f"Training completed in {train_time:.1f}s")

            print(f"\nPredicting on test set")
            y_pred, y_std = predict_bnn(
                mcmc, X_test, mu, sd,
                H=H, n_H=n_H, activation=activation,
                seed=seed+fold_idx+100
            )

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            print(f"Results:")
            print(f"  R² = {r2:.4f}")
            print(f"  RMSE = {rmse:.4f}")
            print(f"  MAE = {mae:.4f}")

            results['fold_r2'].append(r2)
            results['fold_rmse'].append(rmse)
            results['fold_mae'].append(mae)
            results['fold_train_size'].append(len(y_train))
            results['fold_test_size'].append(len(y_test))
            results['predictions'].extend(y_pred)
            results['true_values'].extend(y_test)
            results['pred_std'].extend(y_std)

        except Exception as e:
            print(f"BNN training failed in fold {fold_idx}: {e}")
            continue

    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}\n")

    if len(results['fold_r2']) == 0:
        print("ERROR: All folds failed")
        return results

    print(f"Completed {len(results['fold_r2'])}/{n_folds} folds\n")

    r2_mean = np.mean(results['fold_r2'])
    r2_std = np.std(results['fold_r2'])
    r2_min = np.min(results['fold_r2'])
    r2_max = np.max(results['fold_r2'])
    
    rmse_mean = np.mean(results['fold_rmse'])
    rmse_std = np.std(results['fold_rmse'])
    
    mae_mean = np.mean(results['fold_mae'])
    mae_std = np.std(results['fold_mae'])

    print(f"R²:   {r2_mean:.4f} ± {r2_std:.4f} (min: {r2_min:.4f}, max: {r2_max:.4f})")
    print(f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
    print(f"MAE:  {mae_mean:.4f} ± {mae_std:.4f}")

    overall_r2 = r2_score(results['true_values'], results['predictions'])
    print(f"\nOverall R² across all predictions: {overall_r2:.4f}")

    r2_ci_lower = np.percentile(results['fold_r2'], 2.5)
    r2_ci_upper = np.percentile(results['fold_r2'], 97.5)
    print(f"95% CI for R²: [{r2_ci_lower:.4f}, {r2_ci_upper:.4f}]")

    return results


def plot_cv_results(results, phenotype_name, save_path=None):
    """Plot cross-validation results"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(results['fold_r2'], bins=min(10, len(results['fold_r2'])),
            alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(results['fold_r2']), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(results["fold_r2"]):.4f}')
    ax.set_xlabel('R²', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('R² Distribution Across Folds', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    scatter = ax.scatter(results['true_values'], results['predictions'],
              c=results['pred_std'], alpha=0.6, s=30, cmap='viridis')
    lim = [min(min(results['true_values']), min(results['predictions'])),
           max(max(results['true_values']), max(results['predictions']))]
    ax.plot(lim, lim, 'r--', lw=2, label='Perfect prediction')
    overall_r2 = r2_score(results['true_values'], results['predictions'])
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(f'Overall Predictions (R² = {overall_r2:.4f})',
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Pred. Std Dev')

    ax = axes[1, 0]
    folds = range(1, len(results['fold_r2']) + 1)
    ax.plot(folds, results['fold_r2'], 'o-', linewidth=2,
            markersize=8, color='purple', label='R²')
    ax.axhline(np.mean(results['fold_r2']), color='red',
               linestyle='--', linewidth=2, label='Mean')
    ax.fill_between(folds,
                     np.mean(results['fold_r2']) - np.std(results['fold_r2']),
                     np.mean(results['fold_r2']) + np.std(results['fold_r2']),
                     alpha=0.3, color='red', label='±1 Std')
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('R² Per Fold', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    residuals = np.array(results['true_values']) - np.array(results['predictions'])
    ax.scatter(results['predictions'], residuals, alpha=0.6, s=30, color='darkgreen')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Values', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title('Residual Plot', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.suptitle(f'Cross-Validation Results - {phenotype_name}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved plot: {save_path}")

    plt.show()


if __name__ == "__main__":
    phenotype = "EarHT"

    print("\n" + "="*70)
    print("Elastic Net Feature Selection + BNN Cross-Validation")
    print("="*70)

    total_start = time.time()

    print("\n" + "="*70)
    print("STEP 1: ELASTIC NET FEATURE SELECTION")
    print("="*70)
    
    elasticnet_results = run_elasticnet_feature_selection(
        phenotype_name=phenotype,
        alpha=None,
        l1_ratio=0.5,
        cv_folds=5,
        n_alphas=100,
        coef_threshold=1e-6,
        n_cv_splits=5
    )

    if elasticnet_results is None:
        print("\nElastic Net feature selection failed. Exiting.")
        exit(1)

    print("\n" + "="*70)
    print("STEP 2: BNN CROSS-VALIDATION")
    print("="*70)

    bnn_start = time.time()

    bnn_results = run_cv_with_preselected_features(
        phenotype_name=phenotype,
        n_folds=5,
        H=2,
        n_H=32,
        activation='selu',
        num_samples=1000,
        num_warmup=500,
        num_chains=2,
        seed=42
    )

    bnn_elapsed = time.time() - bnn_start
    total_elapsed = time.time() - total_start

    print(f"\n{'='*70}")
    print(f"COMPLETED")
    print(f"{'='*70}")
    print(f"Elastic Net time: {(total_elapsed - bnn_elapsed)/60:.1f} minutes")
    print(f"BNN time: {bnn_elapsed/60:.1f} minutes")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'='*70}")

    if len(bnn_results['fold_r2']) > 0:
        plot_cv_results(bnn_results, phenotype,
                       save_path=f"{phenotype}_bnn_cv_results.png")

        results_file = f"{phenotype}_bnn_cv_summary.txt"
        with open(results_file, 'w') as f:
            f.write(f"BNN Cross-Validation Summary - {phenotype}\n")
            f.write("="*70 + "\n\n")
            f.write(f"Elastic Net selected {elasticnet_results['n_selected']} features\n")
            f.write(f"L1 ratio: {elasticnet_results['l1_ratio']}\n\n")
            f.write(f"Mean R²: {np.mean(bnn_results['fold_r2']):.4f} ± {np.std(bnn_results['fold_r2']):.4f}\n")
            f.write(f"Mean RMSE: {np.mean(bnn_results['fold_rmse']):.4f} ± {np.std(bnn_results['fold_rmse']):.4f}\n")
            f.write(f"Mean MAE: {np.mean(bnn_results['fold_mae']):.4f} ± {np.std(bnn_results['fold_mae']):.4f}\n\n")
            
            f.write(f"Per-Fold Results:\n")
            for i in range(len(bnn_results['fold_r2'])):
                f.write(f"  Fold {i+1}: R²={bnn_results['fold_r2'][i]:.4f}, ")
                f.write(f"RMSE={bnn_results['fold_rmse'][i]:.4f}, ")
                f.write(f"MAE={bnn_results['fold_mae'][i]:.4f}\n")

        print(f"\nSaved results: {results_file}")
        print(f"\nFinal R²: {np.mean(bnn_results['fold_r2']):.4f} ± {np.std(bnn_results['fold_r2']):.4f}")
