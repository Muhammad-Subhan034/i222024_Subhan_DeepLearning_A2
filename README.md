# Legal Clause Similarity Detection System – Enhanced Report

**Course:** CS452 – Natural Language Processing  
**Student:** Subhan
**File Name:** `CS452_LegalClauseSimilarity_Report.pdf`

---

## 1. Network Details and Baseline Rationale (10 pts)

### Dataset and Settings
- **Data Source:** 395 CSV files containing labeled legal clause categories  
- **Number of Clauses:** 395  
- **Generated Pairs:** 10,000 (5,000 similar + 5,000 dissimilar)  
- **Max Vocabulary Size:** 10,000 (actual: 383)  
- **Max Sequence Length:** 200 tokens  
- **Training Epochs:** 30  
- **Batch Size:** 32  
- **Train/Val/Test Split:** 70% / 10% / 20%  
- **Optimizer:** Adam  
- **Early Stopping & LR Reduction:** Enabled  
- **Environment:** TensorFlow 2.19.0 (GPU-enabled)

### Baseline Architectures

#### (a) BiLSTM-Based Siamese Network

Encoder: Shared BiLSTM (64 units)
Combination: Concatenation + Difference + Element-wise Multiplication
Dense Layers: [128, 64] with Dropout (0.5)
Output: Sigmoid for similarity score
Trainable Parameters: ~234K

#### (b) Multi-Head Attention-Based Siamese Network

Encoder: Shared Multi-Head Attention (64-dim output)
Combination: Concatenation + Difference + Multiplication
Dense Layers: [128, 64] with Dropout
Trainable Parameters: ~231K

### Rationale for Baselines
- **BiLSTM:** captures sequential dependencies and context within clauses.  
- **Attention:** captures long-range global context efficiently.  
Both compare sequence-based vs context-based representations for textual similarity.

---

## 2. Baseline Comparison (10 pts)

| Metric     | BiLSTM (Test) | Attention (Test) |
|------------|:-------------:|:----------------:|
| Accuracy   | 0.4985        | 0.5000           |
| Precision  | 0.4992        | 0.5000           |
| Recall     | 0.9870        | 0.0260           |
| F1-Score   | 0.6631        | 0.0494           |
| ROC-AUC    | 0.4735        | 0.5009           |

### Observations
- Both models perform close to random (~50% accuracy).  
- **BiLSTM:** high recall but poor precision — tends to predict “similar” often.  
- **Attention:** conservative predictions — low recall.  
- Both generalize similarly (test ≈ validation), indicating limited learning signal.

---

## 3. Training Graphs (10 pts)

### BiLSTM Model
- Train Loss: 0.693 → 0.679 (decreasing)  
- Val Loss: 0.692 → 0.703 (increasing after epoch 3)  
- Early stopping triggered at epoch 6  
- Train Accuracy ≈ 0.57, Val Accuracy ≈ 0.50

### Attention Model
- Train Loss: 0.715 → 0.686 (decreasing)  
- Val Loss: plateau near 0.704  
- Early stopping triggered at epoch 8  
- Accuracy stabilized near 0.51

> To display plots in README, add images in `plots/` and reference them:
> `![BiLSTM Loss](plots/bilstm_loss.png)`  
> `![Attention Loss](plots/attention_loss.png)`

---

## 4. Performance Measures and Domain Discussion (15 pts)

### Metrics Used

| Metric   | Description                                  | Relevance |
|----------|----------------------------------------------|-----------|
| Accuracy | Proportion of correct predictions            | Overall indicator |
| Precision| True Positives / Predicted Positives         | Avoid false positives (important in legal) |
| Recall   | True Positives / Actual Positives            | Measures completeness |
| F1-Score | Harmonic mean of Precision & Recall          | Balanced single-number metric |
| ROC-AUC  | Area under ROC curve (separability)         | Class separability |
| PR-AUC   | Area under Precision-Recall curve            | Useful for imbalanced data |

### Most Relevant Metric
For a deployed legal clause similarity system, **Precision** (and ROC-AUC) is most important because false positives (incorrectly marking clauses as similar) can cause legal misinterpretations. Prioritize precision and then tune thresholds for required recall.

---

## 5. Correctly and Incorrectly Matched Clauses (4 pts)

**Correctly Matched (Predicted Similar = 1):**
Clause 1: "independent_contractor..."
Clause 2: "independent-contractor..."
Ground Truth: Similar (1)
Model Output: Similar (Correct)

**Incorrectly Matched (Predicted Similar = 1):**

Clause 1: "liabilities..."
Clause 2: "time..."
Ground Truth: Not Similar (0)
Model Output: Similar (Incorrect)


**Incorrectly Rejected (Predicted Not Similar = 0):**

Clause 1: "organization..."
Clause 2: "organization..."
Ground Truth: Similar (1)
Model Output: Not Similar (Incorrect)


> Note: dataset pairs were randomly generated due to single-clause-per-category; mismatches are expected.

---

## 6. Submission Info (1 pt)

**File Naming Convention:**  
`CS452_LegalClauseSimilarity_Report.pdf`

---

## 7. Summary

| Aspect            | BiLSTM      | Attention    | Remarks                          |
|-------------------|-------------|--------------|----------------------------------|
| Generalization    | Yes         | Yes          | Stable but weak                  |
| Training Behavior | Slight overfitting | Underfitting | Opposite learning dynamics |
| Vocabulary        | 383         | 383          | Too small for robust semantics   |
| Recommendation    | Add real clause pairs | Use pretrained embeddings | Improves semantic capture |

---

## 8. Conclusion

Both Siamese architectures (BiLSTM and Attention) were implemented and trained. Results are near-random due to limited and synthetic training pairs, but the pipeline demonstrates correct architecture, monitoring, and evaluation. The BiLSTM favored recall; the Attention model was conservative. For deployment, collect labelled clause-pair data and use pretrained legal embeddings (e.g., LegalBERT) and contrastive training.

### Future Work
- Integrate pretrained embeddings such as **LegalBERT**.  
- Annotate real clause similarity pairs rather than random pairing.  
- Experiment with **contrastive/triplet loss** and threshold tuning for precision.

---


