#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dış-kaynak karşılaştırmaları için ortak bootstrap istatistik yardımcıları.

Önceki MODEL_CARD.md bootstrap rakamları (v7↔v8, eşik taraması) ad-hoc bir scriptle
üretilmiş ve repoya kaydedilmemişti — bu modül aynı yöntemi (yeniden örnekleme, %95 GA)
tekrar kullanılabilir hâle getiriyor.
"""
from __future__ import annotations

import random


def proportion_ci(hits: list[bool], n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Tek bir oranın (örn. doğru-ayırt) nokta tahmini + %95 bootstrap GA'sı."""
    rng = random.Random(seed)
    n = len(hits)
    point = sum(hits) / n
    boots = []
    idx = list(range(n))
    for _ in range(n_boot):
        sample = [hits[rng.choice(idx)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]
    return point, lo, hi


def paired_diff_ci(hits_a: list[bool], hits_b: list[bool], n_boot: int = 5000,
                    seed: int = 0) -> tuple[float, float, float]:
    """AYNI birimler üzerinde ölçülmüş iki koşulun (örn. stage2 açık/kapalı) farkı için
    eşleştirilmiş bootstrap GA'sı (b - a)."""
    assert len(hits_a) == len(hits_b)
    rng = random.Random(seed)
    n = len(hits_a)
    point = sum(hits_b) / n - sum(hits_a) / n
    idx = list(range(n))
    boots = []
    for _ in range(n_boot):
        sample = [rng.choice(idx) for _ in range(n)]
        a = sum(hits_a[i] for i in sample) / n
        b = sum(hits_b[i] for i in sample) / n
        boots.append(b - a)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]
    return point, lo, hi


def cluster_proportion_ci(hits: list[bool], cluster_ids: list, n_boot: int = 5000,
                           seed: int = 0) -> tuple[float, float, float]:
    """Küme-düzeyi (örn. deyim kimliği) bootstrap — kümeler (deyimler) yeniden örneklenir,
    her kümenin İÇİNDEKİ tüm satırlar birlikte gelir. Etkin örneklem küme sayısıdır, satır
    sayısı değil (Dodiom: 36 deyim, 6861 satır — asıl n=36)."""
    rng = random.Random(seed)
    by_cluster: dict = {}
    for h, c in zip(hits, cluster_ids):
        by_cluster.setdefault(c, []).append(h)
    clusters = list(by_cluster.values())
    k = len(clusters)
    point = sum(hits) / len(hits)
    boots = []
    for _ in range(n_boot):
        sampled = [rng.choice(clusters) for _ in range(k)]
        flat = [h for cl in sampled for h in cl]
        boots.append(sum(flat) / len(flat))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]
    return point, lo, hi


def per_cluster_rates(hits: list[bool], cluster_ids: list) -> dict:
    by_cluster: dict = {}
    for h, c in zip(hits, cluster_ids):
        by_cluster.setdefault(c, []).append(h)
    return {c: sum(v) / len(v) for c, v in by_cluster.items()}
