#!/usr/bin/env Rscript

######## ------------- Libraries ------------- ########
library(patchwork)
library(ggplot2)
library(dplyr)
library(stringr)
library(data.table)
library(colorspace)
library(argparse)
library(purrr)
library(tidyr)

######## ------------- Utils ------------- ########

# Colors
ground_truth_palette <- c(
  "diff. corr."                = "#fdbf6f",
  "mean shift"                  = "#C195C4",
  "mean shift + diff. corr."    = "#b2df8a",
  "non-ground truth"            = "lightgray"
)

# Create a darker ground truth palette
ground_truth_palette_dark <- darken(ground_truth_palette, amount = 0.4)

# Valid focus values
edge_metrics_subset = c('pre-P', 'post-P', 'pre-E', 'post-E', 'pre-CS', 'post-CS', 'int-IS', 'pre-LS', 'post-LS', 'pre-PE', 'post-PE')
node_metrics_subset = c('DC-P', 'DC-E', 'STC', 'PRC-P', 'PRC-E', 'WDC-P', 'WDC-E')
algorithms_subset = c('direct_node', 'PageRank', 'PageRank+', 'DimontRank', 'absDimontRank')

# Spearman correlation heatmap
corr_heatmap <- function(node_rankings, focus){
  # Filter for metric
  data <- node_rankings[edge_metric==metric, ]
  
  # Read in rankings
  ranking_list <- lapply(data$ranking_file, fread)
  
  # Create config column and rename rankings
  if (focus %in% edge_metrics_subset){
    data[, config := paste(node_metric, algorithm, sep = ", ")]
  } else if (focus %in% node_metrics_subset){
    data[, config := paste(edge_metric, algorithm, sep = ", ")]
  } else if (focus %in% algorithms_subset){
    data[, config := paste(node_metric, edge_metric, sep = ", ")]
  } else{
    stop(paste0('Invalid focus parameter: ', focus, '.'))
  }
  names(ranking_list) <- data$config
  
  # Merge ranking data
  merged_data <- reduce(ranking_list, full_join, by = "node")
  
  # Rename columns
  colnames(merged_data)[-1] <- data$config
  
  # Compute correlation matrix
  cor_mat <- cor(merged_data[,-1], method = "spearman", use = "pairwise.complete.obs")
  dist_mat <- as.dist(1 - cor_mat)
  hc <- hclust(dist_mat)
  
  # Cluster
  cor_mat_ordered <- cor_mat[hc$order, hc$order]
  
  cor_df <- as.data.frame(cor_mat_ordered) %>%
    tibble::rownames_to_column(var="Method1") %>%
    pivot_longer(
      cols = -Method1,
      names_to = "Method2",
      values_to = "Similarity"
    )
  
  # Heatmap
  cor_heatmap <- ggplot(cor_df, aes(x = Method2, y = Method1, fill = Similarity)) +
    geom_tile(color="white") +
    geom_text(aes(label = sprintf("%.2f", Similarity)), size = 3) +
    scale_fill_gradient(low = "white", high = "#C03830", name = "Spearman Correlation", limits = c(0, 1)) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle=45, hjust=1),
      panel.grid = element_blank(),
      axis.title = element_blank()
    ) +
    coord_fixed()
  
  height = 1 + 0.5 * ncol(merged_data)
  ggsave(paste0('spearman_corr_heatmap_', focus, '.png'), cor_heatmap, width = height+2, height = height)
}

######## ------------- Argument parser ------------- ########

parser <- ArgumentParser(description='Ranking Similarity')
parser$add_argument('summary_file', 
                    help='Input summary data file storing all generated configurations and their results.')
parser$add_argument('data_type', help = 'Type of data: simulation or real')
args <- parser$parse_args()

summary_file <- args$summary_file
data_type <- args$data_type

######## ------------- Process data ------------- ########

summary_dt <- fread(summary_file)
simulations <- max(summary_dt$id, na.rm = TRUE)

summary_dt <- unique(summary_dt[, c("id", "edge_metric", "node_metric", "algorithm", "ranking_file", "ground_truth_nodes")])
node_rankings <- summary_dt[algorithm!='direct_edge', ]
edge_rankings <- summary_dt[algorithm=='direct_edge', ]

######## ------------- Plotting ------------- ########

# Edge metrics
for (edge_metric in edge_metrics_subset){
  corr_heatmap(node_rankings = node_rankings, focus = edge_metric)
}

# Node metrics
for (node_metric in node_metrics_subset){
  corr_heatmap(node_rankings = node_rankings, focus = node_metric)
}

# Ranking algorithms
for (ranking_alg in ranking_alg_subset){
  corr_heatmap(node_rankings = node_rankings, focus = ranking_alg)
}


