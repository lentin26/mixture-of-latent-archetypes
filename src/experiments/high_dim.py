import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
from src.train import train_mola_on_book, train_cdm_on_book
from src.preprocess import (
    create_tables, 
    split_data_on_chapt_sec, 
    create_section,
    create_global_mask
)
import math
import sys
import warnings

def run_learning_curve_analysis(
        df, n_comps=5, data_sizes=None, n_repeats=10,
        models = ["MoLA", "DINA", "HO-DINA", "GDINA"],
        max_skills=8,
        split_type="chronological"
    ):
    """
    Run learning curve analysis for a book.
    """   
    if data_sizes is None:
        data_sizes = [len(df)] 
    # Break up sections with too many skills
    df = create_section(df, MAX_SKILLS=max_skills)
    df = create_global_mask(df, split_type=split_type)
    max_skills = df.groupby(['chapter_num', 'section_num'])['outcome_code'].nunique().max()
    min_skills = df.groupby(['chapter_num', 'section_num'])['outcome_code'].nunique().min()
    print(f'Max skills in section: {max_skills}.', file=sys.__stdout__)
    print(f'Min skills in section: {min_skills}.', file=sys.__stdout__)
    assert min_skills > 1, f"min_skills = {min_skills}"

    base_seed = 42 # 41ƒ
    max_attempts = 50  # safety cap

    pairs = df[["userId", "itemId", "chapter_num", "section_num", "is_test"]]\
        .drop_duplicates()

    aucs = {}
    for model in models:
        aucs[model] = []

    info = {
        "mola_comp_sizes": [],
        "min_data_size": math.inf,   # start very high
        "max_data_size": -math.inf   # start very low
    }

    for n_users in data_sizes:
        print(f'Running size={n_users}...', file=sys.__stdout__)
        collected = 0
        attempts = 0

        while collected < n_repeats and attempts < max_attempts:
            failed = False
            trial_results = {}
            # Increment seed until desired repeats
            seed = base_seed + attempts
            attempts += 1

            pairs_sampled = (
                pairs
                .groupby(['chapter_num', 'section_num'], as_index=False)
                .apply(lambda x: x.sample(n=min(len(x), n_users), random_state=seed))
            ) 

            df_sub = df.merge(pairs_sampled[["userId", "itemId"]], 
                              on=["userId", "itemId"])
            
            user_item_fact_df, asset_skill_fact_df = create_tables(df_sub)
            train_df, test_df = split_data_on_chapt_sec(user_item_fact_df, seed)

            # ---- Train MoLA ----
            # n_comps = 5 # max(5, int(len(df_sub) / 1000)) #1000
            if "MoLA" in models:
                mola, result \
                    = train_mola_on_book(
                        train_df, test_df, asset_skill_fact_df, 
                        n_comps=n_comps, use_attempts=False
                    )
                
                print(mola.nll_trace, file=sys.__stdout__)
                
                if np.isnan(result["train_auc"]) or np.isnan(result["test_auc"]):
                    failed = True
                else:
                    trial_results['MoLA'] = result

            # ---- Train DINA ----
            # df_dina = df.sample(n=n, random_state=seed)
            if "DINA"in models:
                result = train_cdm_on_book(
                    "DINA", train_df, test_df, asset_skill_fact_df
                )

                if np.isnan(result["train_auc"]) or np.isnan(result["test_auc"]):
                    failed = True
                else:
                    trial_results['DINA'] = result

            if "HO-DINA"in models:
                result = train_cdm_on_book(
                    "HO-DINA", train_df, test_df, asset_skill_fact_df
                )

                if np.isnan(result["train_auc"]) or np.isnan(result["test_auc"]):
                    failed = True
                else:
                    trial_results['HO-DINA'] = result

            if "GDINA"in models:
                result = train_cdm_on_book(
                    "GDINA", train_df, test_df, asset_skill_fact_df
                )

                if np.isnan(result["train_auc"]) or np.isnan(result["test_auc"]):
                    failed = True
                else:
                    trial_results['GDINA'] = result

            # ---- Reject if either AUC is NaN ----
            if failed:
                print(f'❌ Trial failed.', file=sys.__stdout__)
                continue
            else:
                print(f'✅ Trial succeeded.', file=sys.__stdout__)

            # ---- Accept sample ----
            for model_name, result in trial_results.items():
                aucs[model_name].append(result)

            collected += 1

            # ---- Update info table ----
            info['mola_comp_sizes'].append(n_comps)

            n_pairs = len(pairs_sampled)
            if n_pairs > info['max_data_size']:
                info['max_data_size'] = n_pairs
            if n_pairs < info['min_data_size']:
                info['min_data_size'] = n_pairs

        if collected < n_repeats:
            warnings.warn(
                f"Only collected {collected} out of {n_repeats} repeats.",
                RuntimeWarning
            )
    
    return aucs, info
