In **Allenamento_*algoritmo*.ipynb** è presente tutto ciò che serve per allenare i modelli, dove l'input è nella cartella *Data*, mentre l'output è nella cartella *modelli*.
In **confronto_modelli.ipynb** è presente il confronto delle metriche dei modelli allenati, le metriche utilizzate per il confronto sono:
- *Accuracy*, *F1-score* e *confusion matrix* per gli algoritmi di classificazione;
- *R2-score*, *MSE* e *MAE* per gli algoritmi di regressione.

Sia per dimensione dei file, sia per la maggior parte delle metriche, XGBoost risulta il migliore complessivamente, il quale viene utilizzato [in questo branch](https://github.com/EmmanuelLaPorta/TesiLocalizationIndoor/tree/dashboard-v1-tesi-v2).
