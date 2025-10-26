The following table shows the end-to-end deduplication performance on the pinecone/core-2020-05-10-deduplication dataset, including precision, recall, and accuracy metrics.

| Algorithm   |   Precision (Duplicates) |   Recall (Duplicates) |   Precision (Non Duplicates) |   Recall (Non Duplicates) |   Macro F1 score |   Accuracy |
|:------------|-------------------------:|----------------------:|-----------------------------:|--------------------------:|-----------------:|-----------:|
| Datasketch  |                   0.4846 |                0.0467 |                       0.5301 |                    0.9558 |           0.5073 |     0.528  |
| FastSketch  |                   0.4858 |                0.0474 |                       0.5301 |                    0.9554 |           0.508  |     0.5281 |
| Rensa       |                   0.4785 |                0.0475 |                       0.5305 |                    0.9542 |           0.5045 |     0.5281 |
