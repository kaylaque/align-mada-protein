#!/usr/bin/env python3
import argparse
from wt_mapping import build_cluster_to_wt_map

'''
python build_wt_mapping.py \
  --test dataset/predictive-pet-zero-shot-test-2025.csv \
  --wt   dataset/pet-2025-wildtype-cds.csv \
  --clusters output/param/backbone_clusters.csv \
  --out_cluster_map cluster_to_wt.csv \
  --out_sequence_map sequence_to_wt.csv

'''

def main():
    ap = argparse.ArgumentParser(description="Build WT origin mapping per cluster & per sequence")
    ap.add_argument("--test", required=True, help="predictive-pet-zero-shot-test-2025.csv")
    ap.add_argument("--wt", required=True, help="pet-2025-wildtype-cds.csv")
    ap.add_argument("--clusters", required=True, help="backbone_clusters.csv")
    ap.add_argument("--out_cluster_map", default="cluster_to_wt.csv")
    ap.add_argument("--out_sequence_map", default="sequence_to_wt.csv")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--dim", type=int, default=4096)
    args = ap.parse_args()

    cluster_map, seq_map = build_cluster_to_wt_map(
        test_csv=args.test,
        wt_csv=args.wt,
        backbone_clusters_csv=args.clusters,
        out_cluster_map_csv=args.out_cluster_map,
        out_sequence_map_csv=args.out_sequence_map,
        k=args.k,
        dim=args.dim,
    )

    n_clusters = len(cluster_map)
    n_unmapped = (cluster_map["wt_id"] < 0).sum()
    print(f"Clusters: {n_clusters}")
    print(f"Clusters without WT mapped: {n_unmapped}")
    print(f"Wrote: {args.out_cluster_map}")
    print(f"Wrote: {args.out_sequence_map}")

if __name__ == "__main__":
    main()