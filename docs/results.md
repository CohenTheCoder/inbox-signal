# Model comparison (test = 30 hand-written messages)

| model                 |   accuracy |   precision |   recall |   f1 |
|:----------------------|-----------:|------------:|---------:|-----:|
| zero-shot + rules     |       0.87 |        0.81 |     0.93 | 0.87 |
| tf-idf + logreg       |       0.93 |        0.88 |     1    | 0.93 |
| fine-tuned distilbert |       0.9  |        0.82 |     1    | 0.9  |
