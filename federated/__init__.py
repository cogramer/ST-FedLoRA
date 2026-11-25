"""
federated

Package Phase 1 (Warm-up Head) + Phase 2 (Federated LoRA chính, 5 nhánh
tổng hợp độc lập) của ST-FedLoRA (mục 4.3.1, 7.1 đề cương).

    federated.dataset        -- Dataset/DataLoader tokenize cho ModernBERT
    federated.client          -- Local training: Client lành tính + Byzantine
    federated.server          -- Server: run_head_warmup, FederatedLoRAExperiment
    federated.run_experiment  -- Script orchestration cấp cao nhất (CLI)
    federated.aggregation     -- 5 cơ chế tổng hợp (đã hoàn thành trước đó)
"""

from __future__ import annotations

from . import aggregation
from . import client
from . import dataset
from . import server
from . import run_experiment

__all__ = ["aggregation", "client", "dataset", "server", "run_experiment"]
