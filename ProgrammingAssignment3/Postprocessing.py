#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: Financial Cost
#######################################################################################################################
""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""

from utils import *


def enforce_demographic_parity(categorical_results, epsilon):
    demographic_parity_data,test_data = {}, {}
    thresholds = {"African-American": 0, "Caucasian": 0, "Hispanic": 0, "Other": 0}
    accuracy = 0
    print("PRinting categories",categorical_results.keys())
    afr_pred, cauc_pred, hisp_pred, oth_pred = [], [], [], []
    thresh_list, test_thresholds = [], []
    loop_var =  [float(j) / 100 for j in range(100)]
    for i in loop_var:
        thresh_list.append(i)
        test_data["Hispanic"] = apply_threshold(categorical_results["Hispanic"], i)
        hisp_pred.append(
            get_num_predicted_positives(test_data["Hispanic"])
            / len(test_data["Hispanic"])
        )

        test_data["African-American"] = apply_threshold(
            categorical_results["African-American"], i
        )
        afr_pred.append(
            get_num_predicted_positives(test_data["African-American"])
            / len(test_data["African-American"])
        )

        test_data["Caucasian"] = apply_threshold(categorical_results["Caucasian"], i)
        cauc_pred.append(
            get_num_predicted_positives(test_data["Caucasian"])
            / len(test_data["Caucasian"])
        )
        
        test_data["Other"] = apply_threshold(categorical_results["Other"], i)
        oth_pred.append(
            get_num_predicted_positives(test_data["Other"]) / len(test_data["Other"])
        )

    for afr_probab in afr_pred:
        for cauc_probab in cauc_pred:
            if not(compare_probabs(cauc_probab, afr_probab, epsilon)):
                continue
            for hisp_probab in hisp_pred:
                if (
                    not(compare_probabs(hisp_probab, afr_probab, epsilon))
                    or not(compare_probabs(hisp_probab, cauc_probab, epsilon))
                ):
                    continue
                for oth_probab in oth_pred:
                    if (
                        not(compare_probabs(oth_probab, afr_probab, epsilon))
                        or not(compare_probabs(oth_probab, cauc_probab, epsilon)) 
                        or not(compare_probabs(oth_probab, hisp_probab, epsilon)) 
                    ):
                        continue
                    else:
                        poss_threshold = [
                            thresh_list[afr_pred.index(afr_probab)],
                            thresh_list[cauc_pred.index(cauc_probab)],
                            thresh_list[hisp_pred.index(hisp_probab)],
                            thresh_list[oth_pred.index(oth_probab)],
                        ]
                        if poss_threshold not in test_thresholds:
                            test_thresholds.append(poss_threshold)

    for thresh in test_thresholds:
        test_data["Hispanic"] = apply_threshold(
            categorical_results["Hispanic"], thresh[2]
        )
        test_data["African-American"] = apply_threshold(
            categorical_results["African-American"], thresh[0]
        )
        test_data["Caucasian"] = apply_threshold(
            categorical_results["Caucasian"], thresh[1]
        )
        
        test_data["Other"] = apply_threshold(categorical_results["Other"], thresh[3])
        total_accuracy = get_total_accuracy(test_data)
        if total_accuracy > accuracy:
            accuracy = total_accuracy
            thresholds = {
                "African-American": thresh[0],
                "Caucasian": thresh[1],
                "Hispanic": thresh[2],
                "Other": thresh[3],
            }

    for key in categorical_results.keys():
        threshold = thresholds[key]
        demographic_parity_data[key] = apply_threshold(
            categorical_results[key], threshold
        )
    return demographic_parity_data, thresholds



#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""


def enforce_equal_opportunity(categorical_results, epsilon):


    equal_opportunity_data,test_data = {},{}
    thresholds = {"African-American": 0, "Caucasian": 0, "Hispanic": 0, "Other": 0}
    accuracy = 0

    afr_tpr, cauc_tpr, hisp_tpr, oth_tpr = [], [], [], []
    test_thresholds,thresh_list = [],[]
    loop_var = [float(j) / 100 for j in range(100)]
    for i in loop_var:
        thresh_list.append(i)
        
        test_data["Hispanic"] = apply_threshold(categorical_results["Hispanic"], i)
        hisp_tpr.append(get_true_positive_rate(test_data["Hispanic"]))

        test_data["African-American"] = apply_threshold(
            categorical_results["African-American"], i
        )
        afr_tpr.append(get_true_positive_rate(test_data["African-American"]))

        test_data["Caucasian"] = apply_threshold(categorical_results["Caucasian"], i)
        cauc_tpr.append(get_true_positive_rate(test_data["Caucasian"]))

        test_data["Other"] = apply_threshold(categorical_results["Other"], i)
        oth_tpr.append(get_true_positive_rate(test_data["Other"]))

    for afr_probab in afr_tpr:
        for cauc_probab in cauc_tpr:
            if not(compare_probabs(cauc_probab, afr_probab, epsilon)):
                continue
            for hisp_probab in hisp_tpr:
                if (
                    not(compare_probabs(hisp_probab, afr_probab, epsilon))
                    or not(compare_probabs(hisp_probab, cauc_probab, epsilon))
                ):
                    continue
                for oth_probab in oth_tpr:
                    if (
                        not(compare_probabs(oth_probab, afr_probab, epsilon))
                        or not(compare_probabs(oth_probab, cauc_probab, epsilon))
                        or not(compare_probabs(oth_probab, hisp_probab, epsilon))
                    ):
                        continue
                    else:
                        poss_threshold = [
                            thresh_list[afr_tpr.index(afr_probab)],
                            thresh_list[cauc_tpr.index(cauc_probab)],
                            thresh_list[hisp_tpr.index(hisp_probab)],
                            thresh_list[oth_tpr.index(oth_probab)],
                        ]
                        if poss_threshold not in test_thresholds:
                            test_thresholds.append(poss_threshold)

    for thresh in test_thresholds:
        test_data["Hispanic"] = apply_threshold(
            categorical_results["Hispanic"], thresh[2]
        )
        test_data["African-American"] = apply_threshold(
            categorical_results["African-American"], thresh[0]
        )
        test_data["Caucasian"] = apply_threshold(
            categorical_results["Caucasian"], thresh[1]
        )
        
        test_data["Other"] = apply_threshold(categorical_results["Other"], thresh[3])
        total_accuracy = get_total_accuracy(test_data)
        if total_accuracy > accuracy:
            accuracy = total_accuracy
            thresholds = {
                "African-American": thresh[0],
                "Caucasian": thresh[1],
                "Hispanic": thresh[2],
                "Other": thresh[3],
            }

    for key in categorical_results.keys():
        threshold = thresholds[key]
        equal_opportunity_data[key] = apply_threshold(
            categorical_results[key], threshold
        )
    return equal_opportunity_data, thresholds




#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""


def enforce_maximum_profit(categorical_results):
    mp_data,thresholds = {},{}
    test_data = {}
    afr_max, cauc_max, hisp_max, oth_max = 0, 0, 0, 0
    for i in [float(j) / 100 for j in range(0, 100, 1)]:
        test_data["Hispanic"] = apply_threshold(categorical_results["Hispanic"], i)
        hisp_acc = get_num_correct(test_data["Hispanic"]) / len(test_data["Hispanic"])
        if hisp_acc > hisp_max:
            hisp_max = hisp_acc
            thresholds["Hispanic"] = i
            
        test_data["African-American"] = apply_threshold(
            categorical_results["African-American"], i
        )
        afr_acc = get_num_correct(test_data["African-American"]) / len(
            test_data["African-American"]
        )
        if afr_acc > afr_max:
            afr_max = afr_acc
            thresholds["African-American"] = i

        test_data["Caucasian"] = apply_threshold(categorical_results["Caucasian"], i)
        cauc_acc = get_num_correct(test_data["Caucasian"]) / len(test_data["Caucasian"])
        if cauc_acc > cauc_max:
            cauc_max = cauc_acc
            thresholds["Caucasian"] = i

        test_data["Other"] = apply_threshold(categorical_results["Other"], i)
        oth_acc = get_num_correct(test_data["Other"]) / len(test_data["Other"])
        if oth_acc > oth_max:
            oth_max = oth_acc
            thresholds["Other"] = i

    for key in categorical_results.keys():
        threshold = thresholds[key]
        mp_data[key] = apply_threshold(categorical_results[key], threshold)
    return mp_data, thresholds



#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""


def enforce_predictive_parity(categorical_results, epsilon):
    predictive_parity_data,test_data = {}, {} 
   
    thresholds = {"African-American": 0, "Caucasian": 0, "Hispanic": 0, "Other": 0}
    accuracy = 0

    afr_pp, cauc_pp, hisp_pp, oth_pp = [], [], [], []
    test_thresholds,thresh_list = [],[]
    loop_var = [float(j) / 100 for j in range(100)];
    for i in loop_var:
        thresh_list.append(i)
        test_data["Hispanic"] = apply_threshold(categorical_results["Hispanic"], i)
        hisp_pp.append(get_positive_predictive_value(test_data["Hispanic"]))
        
        
        test_data["African-American"] = apply_threshold(
            categorical_results["African-American"], i
        )
        afr_pp.append(get_positive_predictive_value(test_data["African-American"]))

        test_data["Caucasian"] = apply_threshold(categorical_results["Caucasian"], i)
        cauc_pp.append(get_positive_predictive_value(test_data["Caucasian"]))


        test_data["Other"] = apply_threshold(categorical_results["Other"], i)
        oth_pp.append(get_positive_predictive_value(test_data["Other"]))

    for afr_probab in afr_pp:
        for cauc_probab in cauc_pp:
            if not(compare_probabs(cauc_probab, afr_probab, epsilon)):
                continue
            for hisp_probab in hisp_pp:
                if (
                    not(compare_probabs(hisp_probab, afr_probab, epsilon))
                    or not(compare_probabs(hisp_probab, cauc_probab, epsilon))
                ):
                    continue
                for oth_probab in oth_pp:
                    if (
                        not(compare_probabs(oth_probab, afr_probab, epsilon))
                        or not(compare_probabs(oth_probab, cauc_probab, epsilon))
                        or not(compare_probabs(oth_probab, hisp_probab, epsilon))
                    ):
                        continue
                    else:
                        poss_threshold = [
                            thresh_list[afr_pp.index(afr_probab)],
                            thresh_list[cauc_pp.index(cauc_probab)],
                            thresh_list[hisp_pp.index(hisp_probab)],
                            thresh_list[oth_pp.index(oth_probab)],
                        ]
                        if poss_threshold not in test_thresholds:
                            test_thresholds.append(poss_threshold)

    for thresh in test_thresholds:
        test_data["Hispanic"] = apply_threshold(
            categorical_results["Hispanic"], thresh[2]
        )
        test_data["African-American"] = apply_threshold(
            categorical_results["African-American"], thresh[0]
        )
        test_data["Caucasian"] = apply_threshold(
            categorical_results["Caucasian"], thresh[1]
        )

        test_data["Other"] = apply_threshold(categorical_results["Other"], thresh[3])
        total_accuracy = get_total_accuracy(test_data)
        if total_accuracy > accuracy:
            accuracy = total_accuracy
            thresholds = {
                "African-American": thresh[0],
                "Caucasian": thresh[1],
                "Hispanic": thresh[2],
                "Other": thresh[3],
            }

    for key in categorical_results.keys():
        threshold = thresholds[key]
        predictive_parity_data[key] = apply_threshold(
            categorical_results[key], threshold
        )
    return predictive_parity_data, thresholds



    ###################################################################################################################


""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""


def enforce_single_threshold(categorical_results):
    single_threshold_data,thresholds = {},{}
    test_data = {}
    accuracy = 0
    loop_var = [float(j) / 100 for j in range(0, 100, 1)]
    for i in loop_var:
        for key in categorical_results.keys():
            test_data[key] = apply_threshold(categorical_results[key], i)
        total_accuracy = get_total_accuracy(test_data)
        if total_accuracy > accuracy:
            accuracy = total_accuracy
            thresholds = {
                "African-American": i,
                "Caucasian": i,
                "Hispanic": i,
                "Other": i,
            }

    single_threshold_data = {}
    for key in categorical_results.keys():
        threshold = thresholds[key]
        single_threshold_data[key] = apply_threshold(
            categorical_results[key], threshold
        )

    return single_threshold_data, thresholds



def compare_probabs(p1, p2, epsilon):

    probab = abs(p1 - p2)
    if probab <= epsilon:
        return probab
