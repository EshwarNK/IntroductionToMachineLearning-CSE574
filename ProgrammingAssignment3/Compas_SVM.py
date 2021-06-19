from sklearn import svm
from Preprocessing import preprocess
from Report_Results import report_results
import numpy as np
from utils import *


def SVM_classification(metrics):

    training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics, recalculate=False, causal=False)

    np.random.seed(42)
    SVR = svm.LinearSVR(C=1.0/float(len(test_data)), max_iter=5000)
    SVR.fit(training_data, training_labels)

    data = np.concatenate((training_data, test_data))
    labels = np.concatenate((training_labels, test_labels))

    predictions = SVR.predict(data)
    return data, predictions, labels, categories, mappings

#######################################################################################################################

metrics = ["sex", "age_cat", 'race', 'c_charge_degree', 'priors_count']

data, predictions, labels, categories, mappings = SVM_classification(metrics)
race_cases = get_cases_by_metric(data, categories, "race", mappings, predictions, labels)

report_results(race_cases)

'''
Loaded training data
Attempting to enforce demographic parity...
--------------------DEMOGRAPHIC PARITY RESULTS--------------------

Probability of positive prediction for African-American: 0.40569395017793597
Probability of positive prediction for Caucasian: 0.4252400548696845
Probability of positive prediction for Hispanic: 0.41563055062166965
Probability of positive prediction for Other: 0.41297935103244837

Accuracy for African-American: 0.5744365361803084
Accuracy for Caucasian: 0.6085962505715592
Accuracy for Hispanic: 0.6447602131438721
Accuracy for Other: 0.6430678466076696

FPR for African-American: 0.30659415363698167
FPR for Caucasian: 0.3357903357903358
FPR for Hispanic: 0.30513595166163143
FPR for Other: 0.3106796116504854

FNR for African-American: 0.5176223040504997
FNR for Caucasian: 0.4616977225672878
FNR for Hispanic: 0.4267241379310345
FNR for Other: 0.42857142857142855

TPR for African-American: 0.4823776959495003
TPR for Caucasian: 0.5383022774327122
TPR for Hispanic: 0.5732758620689655
TPR for Other: 0.5714285714285714

TNR for African-American: 0.6934058463630184
TNR for Caucasian: 0.6642096642096642
TNR for Hispanic: 0.6948640483383686
TNR for Other: 0.6893203883495146

Threshold for African-American: 0.38
Threshold for Caucasian: 0.18
Threshold for Hispanic: 0.06
Threshold for Other: 0.04

Total cost: 
$-481,855,588
Total accuracy: 0.5957282154465253
-----------------------------------------------------------------

Attempting to enforce equal opportunity...
--------------------EQUAL OPPORTUNITY RESULTS--------------------

Accuracy for African-American: 0.5797746144721234
Accuracy for Caucasian: 0.6181984453589392
Accuracy for Hispanic: 0.6447602131438721
Accuracy for Other: 0.6430678466076696

FPR for African-American: 0.41536369816451396
FPR for Caucasian: 0.3521703521703522
FPR for Hispanic: 0.30513595166163143
FPR for Other: 0.3106796116504854

FNR for African-American: 0.42398737506575485
FNR for Caucasian: 0.4192546583850932
FNR for Hispanic: 0.4267241379310345
FNR for Other: 0.42857142857142855

TPR for African-American: 0.5760126249342452
TPR for Caucasian: 0.5807453416149069
TPR for Hispanic: 0.5732758620689655
TPR for Other: 0.5714285714285714

TNR for African-American: 0.5846363018354861
TNR for Caucasian: 0.6478296478296478
TNR for Hispanic: 0.6948640483383686
TNR for Other: 0.6893203883495146

Threshold for African-American: 0.24
Threshold for Caucasian: 0.08
Threshold for Hispanic: 0.06
Threshold for Other: 0.04

Total cost: 
$-474,671,482
Total accuracy: 0.6017644327503482
-----------------------------------------------------------------

Attempting to enforce maximum profit...
--------------------MAXIMUM PROFIT RESULTS--------------------

Accuracy for African-American: 0.6411625148279952
Accuracy for Caucasian: 0.6241426611796982
Accuracy for Hispanic: 0.6447602131438721
Accuracy for Other: 0.6814159292035398

FPR for African-American: 0.5030591434398368
FPR for Caucasian: 0.3759213759213759
FPR for Hispanic: 0.30513595166163143
FPR for Other: 0.09223300970873786

FNR for African-American: 0.24723829563387692
FNR for Caucasian: 0.37577639751552794
FNR for Hispanic: 0.4267241379310345
FNR for Other: 0.6691729323308271

TPR for African-American: 0.7527617043661231
TPR for Caucasian: 0.6242236024844721
TPR for Hispanic: 0.5732758620689655
TPR for Other: 0.3308270676691729

TNR for African-American: 0.4969408565601632
TNR for Caucasian: 0.624078624078624
TNR for Hispanic: 0.6948640483383686
TNR for Other: 0.9077669902912622

Threshold for Hispanic: 0.06
Threshold for African-American: 0.08
Threshold for Caucasian: 0.07
Threshold for Other: 0.66

Total cost: 
$-440,499,130
Total accuracy: 0.6378269617706237
-----------------------------------------------------------------

Attempting to enforce predictive parity...
--------------------PREDICTIVE PARITY RESULTS--------------------

Accuracy for African-American: 0.6408659549228944
Accuracy for Caucasian: 0.6108824874256973
Accuracy for Hispanic: 0.6305506216696269
Accuracy for Other: 0.6371681415929203

PPV for African-American: 0.6387771520514883
PPV for Caucasian: 0.6334106728538283
PPV for Hispanic: 0.6333333333333333
PPV for Other: 0.631578947368421

FPR for African-American: 0.610469068660775
FPR for Caucasian: 0.1294021294021294
FPR for Hispanic: 0.09969788519637462
FPR for Other: 0.06796116504854369

FNR for African-American: 0.16465018411362442
FNR for Caucasian: 0.717391304347826
FNR for Hispanic: 0.7543103448275862
FNR for Other: 0.8195488721804511

TPR for African-American: 0.8353498158863756
TPR for Caucasian: 0.28260869565217395
TPR for Hispanic: 0.2456896551724138
TPR for Other: 0.18045112781954886

TNR for African-American: 0.389530931339225
TNR for Caucasian: 0.8705978705978706
TNR for Hispanic: 0.9003021148036254
TNR for Other: 0.9320388349514563

Threshold for African-American: 0.06
Threshold for Caucasian: 0.69
Threshold for Hispanic: 0.67
Threshold for Other: 0.9

Total cost: 
$-450,002,032
Total accuracy: 0.629623897229531
-----------------------------------------------------------------

Attempting to enforce single threshold...
--------------------SINGLE THRESHOLD RESULTS--------------------

Accuracy for African-American: 0.6399762752075919
Accuracy for Caucasian: 0.6241426611796982
Accuracy for Hispanic: 0.6412078152753108
Accuracy for Other: 0.6548672566371682

FPR for African-American: 0.5445275322909585
FPR for Caucasian: 0.3759213759213759
FPR for Hispanic: 0.30513595166163143
FPR for Other: 0.2621359223300971

FNR for African-American: 0.21725407680168332
FNR for Caucasian: 0.37577639751552794
FNR for Hispanic: 0.4353448275862069
FNR for Other: 0.47368421052631576

TPR for African-American: 0.7827459231983167
TPR for Caucasian: 0.6242236024844721
TPR for Hispanic: 0.5646551724137931
TPR for Other: 0.5263157894736843

TNR for African-American: 0.4554724677090415
TNR for Caucasian: 0.624078624078624
TNR for Hispanic: 0.6948640483383686
TNR for Other: 0.7378640776699029

Threshold for African-American: 0.07
Threshold for Caucasian: 0.07
Threshold for Hispanic: 0.07
Threshold for Other: 0.07

Total cost: 
$-441,760,300
Total accuracy: 0.6355053397306919
-----------------------------------------------------------------
Postprocessing took approximately: 0:00:08.076376 seconds

'''

