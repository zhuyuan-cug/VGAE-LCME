# VGAE-LCME
VGAE-LCME is a VGAE algorithm-based method for predicting LncRNA and cancer metastasis events

# Requirements
python = 3.6

tensorflow==1.15.0

numpy==1.19.5

pandas ==0.25.3

scikit-learn==0.24.0

matplotlib == 3.2.1

# Usage
In this GitHub project, we give a demo to show how VGAE-LCME works. 

1.LncLnc.xlsx is the LncRNA similarity matrix file .

2.Lnc_type.xlsx is the LncRNA-cancer type association matrix file.

3.LncCancer.xlsx is the LncRNA-cancer metastatic event associations matrix file.

4.LncR2metasta.xls is raw data download from http://lncr2metasta.wchoda.com.

Run Lnc-Event.py can obtain Lnc-cancer metastatic event associations result and ROC、PR curve.

Run Lnc-CancerType.py can obtain Lnc-cancer type associations result and ROC、PR curve.
