# DUAL-X: Evidence-gated training data strategy (v0.1)

## Purpose
Train product understanding and cold-start ranking before real Thai affiliate transaction data exists. Never treat foreign-market reviews as Thai affiliate conversions or commissions.

## Candidate datasets (verify current licenses and redistribution terms before download)
| Dataset | URL | Intended use | Limitation |
|---|---|---|---|
| Amazon Reviews 2023 | https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 | Product text, category, co-purchase and preference pretraining | US/English, review selection bias; no Shopee/Lazada commission labels |
| Amazon-C4 | https://huggingface.co/datasets/McAuley-Lab/Amazon-C4 | Search relevance evaluation, retrieval fine-tuning only if split is permitted | Published test split must remain held out |
| Mercari MerRec | https://huggingface.co/datasets/mercari-us/merrec | Behavioral ranking exploration after license/schema inspection | Marketplace/domain shift; validate session semantics |
| Retailrocket events | https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset | Funnel/sequence model experiments | Historical, not Thailand; check usage terms |

## Provenance manifest required per source
source_id, canonical_url, revision_or_sha, retrieved_utc, license, license_evidence_url,
permitted_use, data_fields, market, date_range, split_policy, sha256, pii_review, ingestion_status.
Reject unknown licenses and missing provenance from training.

## Train in three stages
1. Text/category retrieval: multilingual embedding model + nearest-neighbor baseline; freeze Amazon-C4 test.
2. Ranking: LightGBM or CatBoost vs simple commission and popularity baselines. Use chronological split and group by product/seller; exclude future interactions.
3. Calibrate Thai-market conversion, approved EPC, refund, repeat purchases only on authorized Shopee/Lazada reports with matching click attribution and observation windows.

## Evidence gates
- No platform API session or credential scraping.
- No synthetic conversions as ground truth.
- Never train on held-out test data or future purchase outcomes.
- Report Recall@20, NDCG@10, log loss, Brier, ECE, observed approved EPC/1000 and net contribution with confidence intervals.
- Slice by platform, category, new product, campaign, time, price, and missing data.
- Compare fixed seeds, identical splits, identical compute budgets and a simple rule-based baseline.
- Deploy ranking only after prospective holdout or randomized A/B evidence, with kill switch and spend caps.

## Core business objective
Expected contribution = approved commission - advertising cost - platform-independent operating costs.
Long-term value requires repeat-purchase cohorts with explicit observation windows; avoid extrapolating beyond observed support.

## Status
Research plan only. Dataset downloads, training runs, model weights and live affiliate validation have NOT yet been executed.
