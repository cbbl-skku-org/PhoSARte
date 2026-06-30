The prediction results will be saved in this folder, with the format 'PhoSARte_[model_type]_Results.csv'.
For example, if you run `python predict.py --input data/Combined/Combined_Test.csv --model_type Generic --batch_size 128`, the results will be saved to `results/PhoSARte_Generic_Results.csv`.

The output CSV file contains the following columns:
[Sample, Real_Label, Predict_Probability, Predict_Class]

Where:
- **Sample**: The input protein sequence.
- **Real_Label**: The true label of the protein (1 for positive, 0 for negative).
- **Predict_Probability**: The predicted probability of the protein being positive.
- **Predict_Class**: The predicted class of the protein (1 for positive, 0 for negative).

**Example:**
```csv
Sample,Real_Label,Predicted_Probability,Predicted_Class
EASLNKSKSATTTPSGSPRTSQQNVYNPSEGST,1,0.976992,1
SNDSRSSLIRKRSTRRSVRGSQAQDRKLSTKEA,1,0.23797959,0
PGSQYGTMTRQISRHNSTTSSTSSGGYRRTPSV,1,0.7804198,1
AKYVERKFVDKYSISLSPPEQQKKFVSKSSEEK,1,0.9791765,1
EQQKKFVSKSSEEKRLSISKFGPGDQVRASAQS,1,0.90572673,1
```