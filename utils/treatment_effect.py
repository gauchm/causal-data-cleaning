import numpy as np
import pandas as pd
import scipy
#from pymatch.Matcher import Matcher
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics


def propensity_score(data, TREATMENT, TREATMENT_COL, treatment_subdimensions, OUTCOME, CAT_COVARIATES, n_models = 50, additional_excludes=[]):
    """ Augment data frame with propensity scores """
    #m = Matcher(data[data[TREATMENT] == 1], data[data[TREATMENT] == 0], yvar=TREATMENT, 
    #            exclude=[OUTCOME, TREATMENT_COL] + treatment_subdimensions + additional_excludes)
    #m.fit_scores(balance=True, nmodels=n_models)
    #m.predict_scores()
    #print(m.data['scores'].value_counts())
    #return m.data.copy()
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoder.fit(data[CAT_COVARIATES])

    train_X = pd.concat([data.drop(CAT_COVARIATES, axis=1).reset_index(drop=True), 
                         pd.DataFrame(encoder.transform(data[CAT_COVARIATES]))], axis=1)\
                .drop([TREATMENT, TREATMENT_COL, OUTCOME] + treatment_subdimensions + additional_excludes, axis=1)

    data['scores'] = 0.0
    for i in range(n_models):
        print('Training model {}/{}'.format(i+1, n_models))
        m = LogisticRegression(solver='liblinear')
        m.fit(train_X, data[TREATMENT])

        prediction = m.predict(train_X)    
        data['scores'] += m.predict_proba(train_X)[:,1]
        
    data['scores'] = data['scores'] / n_models
    print('Train Accuracy: ', metrics.accuracy_score(data[TREATMENT], data['scores'] > 0.5))
    return data


def propensity_stratification(scored_data, continuous_covariates, categorical_covariates, 
                              TREATMENT, OUTCOME, num_strata=50, min_count_per_stratum=10, n_iter=50):
    """ Calculates ATT through propensity stratification """
    scored_data['stratum'] = pd.qcut(scored_data['scores'], num_strata, labels=False)
    
    scored_data['Tbar'] = (1 - scored_data[TREATMENT])
    scored_data['T_y'] = scored_data[TREATMENT] * scored_data[OUTCOME]
    scored_data['Tbar_y'] = scored_data['Tbar'] * scored_data[OUTCOME]    
    stratified = scored_data.groupby('stratum').filter(lambda stratum: min(stratum.loc[stratum[TREATMENT] == 1].shape[0],
                                                                           stratum.loc[stratum[TREATMENT] == 0].shape[0]) > min_count_per_stratum)

    lost = list(filter(lambda x: x not in stratified['stratum'].unique(), range(num_strata)))
    if len(lost) > 0:
        print('Lost strata: {}. They were too small.'.format(lost))
    
    for col in continuous_covariates + categorical_covariates:
        treated_contingencies = scored_data[scored_data[TREATMENT] == 1].groupby('stratum')[col].value_counts()
        control_contingencies = scored_data[scored_data[TREATMENT] == 0].groupby('stratum')[col].value_counts()
        contingencies = pd.concat([treated_contingencies, control_contingencies], axis=1).fillna(0)
        contingencies.columns = ['treatment_contingency', 'control_contingency']
        
        alpha = .05
        for stratum in filter(lambda s: s not in lost, contingencies.index.get_level_values(0).unique()):
            if col in categorical_covariates:
                contingency = contingencies.loc[stratum]
                # remove all-zero rows to avoid scipy throwing an error in chi2 test
                contingency =  contingency[(contingency['treatment_contingency'] != 0) | (contingency['control_contingency'] != 0)]
                chi2 = scipy.stats.chi2_contingency(contingency.T.values)
                critical_value = scipy.stats.chi2.ppf(q=1-alpha, df=chi2[2])
                if chi2[0] > critical_value and chi2[1] < alpha:
                    print('Covariate {} not balanced enough in stratum {}. chi2-statistic: {} > {}, p-value {}'.format(col, stratum, chi2[0], 
                                                                                                                       critical_value, chi2[1]))
            else:
                ks_test = scipy.stats.ks_2samp(scored_data[scored_data[TREATMENT] == 1][col], scored_data[scored_data[TREATMENT] == 0][col])
                critical_value = np.sqrt(-np.log(alpha) / 2) * np.sqrt(2 * n_iter/(n_iter * n_iter))
                if ks_test[0] > critical_value and ks_test[1] < alpha:
                    print('Covariate {} not balanced enough in stratum {}. ks-statistic: {}'.format(col, stratum, ks_test))
    
    agg_outcomes = stratified.groupby('stratum').agg({TREATMENT: ['sum'], 'Tbar': ['sum'], 'T_y': ['sum'], 'Tbar_y': ['sum']})
    agg_outcomes.columns = ["_".join(x) for x in agg_outcomes.columns.ravel()]
    treatment_sum_name = TREATMENT + "_sum"

    agg_outcomes['T_y_mean'] = agg_outcomes['T_y_sum'] / agg_outcomes[treatment_sum_name]
    agg_outcomes['Tbar_y_mean'] = agg_outcomes['Tbar_y_sum'] / agg_outcomes['Tbar_sum']
    agg_outcomes['effect'] = agg_outcomes['T_y_mean'] - agg_outcomes['Tbar_y_mean']
    treatment_population = agg_outcomes[treatment_sum_name].sum()

    treatment_effect_among_treated = (agg_outcomes['effect'] * agg_outcomes[treatment_sum_name]).sum()
    return treatment_effect_among_treated, treatment_population


def match_then_stratify(dataset, exact_match_vars, TREATMENT, TREATMENT_COL, treatment_subdimensions, OUTCOME, CONT_COVARIATES, 
                        CAT_COVARIATES, n_models=50, num_strata=5, min_count_per_stratum=10, n_iter=50, additional_excludes=[]):
    """
    Explicitly control for highly correlated covariate, then do propensity stratification.
    Explicit control is needed when covariates would otherwise not be balanced within the strata.
    """
    cum_tt = 0
    cum_count = 0
    if len(exact_match_vars) > 0:
        groups = dataset.groupby(exact_match_vars)
    else:
        groups = [('', dataset)]
    for key, data in groups:
        scored = propensity_score(data.drop(exact_match_vars, axis=1), TREATMENT, TREATMENT_COL, treatment_subdimensions, 
                                  OUTCOME, [c for c in CAT_COVARIATES if c not in exact_match_vars], n_models=n_models, 
                                  additional_excludes=additional_excludes)
        tt, count = propensity_stratification(scored, list(c for c in CONT_COVARIATES if c not in exact_match_vars), 
                                                      list(c for c in CAT_COVARIATES if c not in exact_match_vars), 
                                              TREATMENT, OUTCOME, num_strata, min_count_per_stratum, n_iter=n_iter)
        cum_tt += tt
        cum_count += count
    
    # Average treatment effect among treated. 
    # "How would the outcome for the treated subpopulation change had they not been treated."
    att = cum_tt / cum_count
    return att


def match_exactly(dataset, TREATMENT, TREATMENT_COL, treatment_subdimensions, OUTCOME, exclude_vars=[]):
    match_vars = [v for v in dataset.columns.values if v not in [TREATMENT, OUTCOME, TREATMENT_COL] + treatment_subdimensions + exclude_vars]
    cum_tt, cum_count = 0, 0
    dataset['Tbar'] = (1 - dataset[TREATMENT])
    dataset['T_y'] = dataset[TREATMENT] * dataset[OUTCOME]
    dataset['Tbar_y'] = dataset['Tbar'] * dataset[OUTCOME]

    agg_outcomes = dataset.groupby(match_vars).agg({TREATMENT: ['sum'], 'Tbar': ['sum'], 'T_y': ['sum'], 'Tbar_y': ['sum']})
    agg_outcomes.columns = ["_".join(x) for x in agg_outcomes.columns.ravel()]
    treatment_sum_name = TREATMENT + "_sum"

    agg_outcomes['T_y_mean'] = agg_outcomes['T_y_sum'] / agg_outcomes[treatment_sum_name]
    agg_outcomes['Tbar_y_mean'] = agg_outcomes['Tbar_y_sum'] / agg_outcomes['Tbar_sum']
    agg_outcomes['effect'] = agg_outcomes['T_y_mean'] - agg_outcomes['Tbar_y_mean']
    treatment_population = agg_outcomes[treatment_sum_name].sum()
    
    treatment_effect_among_treated = (agg_outcomes['effect'] * agg_outcomes[treatment_sum_name]).sum()
    return treatment_effect_among_treated / treatment_population
