To reproduce:

First run split_lanl to chop up the dataset and build the maps/edge lists

Then run argus_to_csr to build the tgraphs from the chopped up files

nontemporal_split will break it into the reported tr/va/te partitions randomly, holding out all anomalous edges for the test set

../data/lanl14argus_tgraph_tr.pt: 2271328 edges
../data/lanl14argus_tgraph_va.pt: 283916 edges
../data/lanl14argus_tgraph_te.pt: 284346 edges
../data/lanl14argus_tgraph_raw_tr.pt: 2271328 edges
../data/lanl14argus_tgraph_raw_va.pt: 283916 edges
../data/lanl14argus_tgraph_raw_te.pt: 284346 edges

../temporal_split will build the dirty, and ts variants.

data/lanl14argus-dirtyts_tgraph_tr.pt: 1093189 edges
data/lanl14argus-dirtyts_tgraph_va.pt: 158378 edges
data/lanl14argus-dirtyts_tgraph_te.pt: 1588023 edges
data/lanl14argus-ts_tgraph_tr.pt: 276732 edges
data/lanl14argus-ts_tgraph_va.pt: 30748 edges
data/lanl14argus-ts_tgraph_te.pt: 2532110 edges

scrub_lanl will build the clean variant

inferred window: train/val < 604800 (7.00 days), train < 544320 (6.30 days)
reconstructed the dirty split exactly; scrubbing now
removed 29 malicious edges from train and 3 from validation (32 of 439 total)
wrote ../data/lanl14argus-cleants_tgraph_tr.pt: 1,093,160 edges
wrote ../data/lanl14argus-cleants_tgraph_va.pt: 158,375 edges
wrote ../data/lanl14argus-cleants_tgraph_te.pt: 1,588,023 edges, 407 malicious

test sets are identical to lanl14argus-dirtyts; train differs by 29 edges (0.0027% of training data)