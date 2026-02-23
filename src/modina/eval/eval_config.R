# Global params for evaluation
library(patchwork)
library(ggplot2)
library(dplyr)
library(readr)
library(stringr)
library(data.table)
library(colorspace)
library(magrittr)

# Metrics and Algorithms
edge_metrics_subset = c('pre-P', 'post-P', 'pre-E', 'post-E', 'pre-CS', 'post-CS', 'int-IS', 'pre-LS', 'post-LS')
node_metrics_subset = c('DC-P', 'DC-E', 'STC', 'PRC-P', 'PRC-E', 'WDC-P', 'WDC-E')
ranking_alg_subset = c('absDimontRank', 'DimontRank', 'PageRank', 'PageRank+', 'direct')

# Set colors for multiple ground truth types
ground_truth_palette <- c(
  "diff. corr."                = "#fdbf6f",
  "mean shift"                  = "#C195C4",
  "mean shift + diff. corr."    = "#b2df8a",
  "non-ground truth"            = "lightgray"
)

ground_truth_palette_edges <- c(
  "diff. corr."                = "#fdbf6f",
  "mean shift + diff. corr."    = "#b2df8a",
  "non-ground truth"            = "lightgray"
)

# Create a darker ground truth palette
ground_truth_palette_dark <- darken(ground_truth_palette, amount = 0.4)
ground_truth_palette_edges_dark <- darken(ground_truth_palette_edges, amount = 0.4)

# Set color for binary ground truth types
ground_truth_palette_boolean <- c("False" = "snow2", "True" = "#C03830")

filter_metric_palette <- c('resc. E' = '#A6CEE3', 'resc. P' = '#1C4E80')

