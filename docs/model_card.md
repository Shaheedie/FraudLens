# FraudLens model card

## Intended use
Educational and portfolio demonstration of rare-event classification, model comparison, threshold selection, and risk scoring.

## Not intended for
Real payment authorization, customer blocking, law-enforcement decisions, or any consequential automated decision.

## Data limitations
The ULB/Worldline benchmark covers two days of European card transactions from September 2013. V1–V28 are anonymized PCA components, limiting semantic explanations. Performance should not be assumed to transfer to current payment systems, other countries, merchants, or fraud types.

## Evaluation
Model selection uses validation PR-AUC. A separate validation threshold minimizes a configurable cost function. The held-out chronological test split is evaluated once after selection.
