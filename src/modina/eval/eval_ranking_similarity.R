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
library(pROC)

######## ------------- Utils ------------- ########

# Colors
ground_truth_palette <- c(
  "diff. corr."                = "#fdbf6f",
  "mean shift"                  = "#C195C4",
  "mean shift + diff. corr."    = "#b2df8a",
  "non-ground truth"            = "lightgray"
)

ground_truth_palette_boolean <- c("False" = "snow2", "True" = "#C03830")

# Valid focus values
edge_metrics_subset = c('pre-P', 'post-P', 'pre-E', 'post-E', 'pre-CS', 'post-CS', 'int-IS', 'pre-LS', 'post-LS', 'pre-PE', 'post-PE')
node_metrics_subset = c('DC-P', 'DC-E', 'STC', 'PRC-P', 'PRC-E', 'WDC-P', 'WDC-E')
algorithms_subset = c('direct_node', 'PageRank', 'PageRank+', 'DimontRank', 'absDimontRank')

# Spearman correlation heatmap
corr_heatmap <- function(data){
  # Compute correlation matrix
  cor_mat <- cor(data[,-1], method = "spearman", use = "pairwise.complete.obs")
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
  
  return(cor_heatmap)
}


# Create rank heatmaps for ground truth nodes
rank_heatmap <- function(data, gt_table){
  # Create dictionary containing all ground truth nodes
  gt_dict <- setNames(gt_table$description, gt_table$node)
  gt_nodes <- names(gt_dict)
  
  # Create ground truth annotation dataframe for heatmap
  gt_info <- as.data.table(data.frame(node = gt_nodes))
  gt_info <- gt_info[, groundtruth := gt_dict[match(node, names(gt_dict))]]
  
  # Prepare heatmap matrix
  m <- as.matrix(data[, -1])
  rownames(m) <- data$node
  
  #m <- matrix(NA, nrow = length(gt_nodes), ncol = ncol(data),
  #            dimnames = list(gt_nodes, data$config))
  
  if (ncol(m) < 2){
    return(FALSE)
  }
  
  # Sort according to ground truth annotation
  sorted_nodes <- gt_info$node[order(gt_info$groundtruth, decreasing = TRUE)]
  gt_info$node <- factor(gt_info$node, levels = rev(sorted_nodes))
  
  # Cluster configs
  dist_cols <- dist(t(m), method = "euclidean")
  clust_cols <- hclust(dist_cols, method = "ward.D2")
  sorted_configs <- clust_cols$labels[clust_cols$order]
  
  # Set gt palette
  gt_palette <- ground_truth_palette
  
  # Annotation column
  gt_info$groundtruth <- factor(gt_info$groundtruth, levels = c('mean shift + diff. corr.', 'mean shift', 'diff. corr.'))
  annotation <- ggplot(gt_info, aes(x = "Annotation", y = node, fill = groundtruth)) +
    geom_tile(color='white') +
    scale_fill_manual(values = gt_palette,
                      name = "Ground Truth",
                      na.value = 'snow2') +
    theme_void() +
    theme(legend.position = 'right')
  
  # Rank columns
  df <- as.data.table(m, keep.rownames = "node")
  df <- melt(df, id.vars = "node", variable.name = "config", value.name = "rank")
  
  # Change order
  df$config <- factor(df$config, levels = sorted_configs)
  df$node <- factor(df$node, levels = rev(sorted_nodes))
  
  # Plot
  heatmap <- ggplot(df, aes(x = config, y = node, fill = rank)) + 
    geom_tile(color = "white") +
    scale_fill_gradient(low='#1A4D91',
                        high='white',
                        na.value='white',
                        name='Rank') +
    theme_minimal() +
    theme(legend.position = 'right',
          axis.text.x = element_text(angle = 45, hjust = 1),
          axis.text.y = element_blank(),
          plot.margin = margin(t = 5, r = 5, b = 5, l = 50),
    ) +
    labs(x = "",
         y = "")
  
  annotated_heatmap <- heatmap + annotation + 
    plot_layout(widths = c(ncol(m), 1), guides = "collect") &
    theme(legend.position = "none")
  
  return(annotated_heatmap)
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

######## ------------- Plotting ------------- ########

for (sim in 1:simulations){
  sim_summary <- summary_dt[id == sim, ]
  gt <- fread(unique(sim_summary[, ground_truth_nodes]))
  node_rankings <- sim_summary[algorithm!='direct_edge', ]
  
  # Edge metrics
  for (edge_metric in edge_metrics_subset){
    # Filter for metric
    data <- node_rankings[edge_metric==metric, ]
    
    # Read in rankings
    ranking_list <- lapply(data$ranking_file, fread)
    
    # Create config column and rename rankings
    data[, config := paste(node_metric, algorithm, sep = ", ")]
    names(ranking_list) <- data$config
    
    # Merge ranking data
    merged_data <- reduce(ranking_list, full_join, by = "node")
    colnames(merged_data)[-1] <- data$config
    
    # Correlation heatmap
    corr_heatmap <- corr_heatmap(data = merged_data)
    height = 1 + 0.5 * ncol(merged_data)
    ggsave(paste0(sim, '_spearman_corr_heatmap_', edge_metric, '.png'), corr_heatmap, width = height+2, height = height)
    
    # Rank heatmap
    if (data_type == 'simulation'){
      rank_heatmap <- rank_heatmap(data = merged_data, gt_table = gt)
      width = 5.5
      height = 0.25 * nrow(gt)
      ggsave(paste0(sim, '_rank_heatmap_', edge_metric, '.png'), rank_heatmap, width = width, height = height)
    }
  }
  
  # Node metrics
  for (node_metric in node_metrics_subset){
    # Filter for metric
    data <- node_rankings[edge_metric==metric, ]
    
    # Read in rankings
    ranking_list <- lapply(data$ranking_file, fread)
    
    # Create config column and rename rankings
    data[, config := paste(edge_metric, algorithm, sep = ", ")]
    names(ranking_list) <- data$config
    
    # Merge ranking data
    merged_data <- reduce(ranking_list, full_join, by = "node")
    colnames(merged_data)[-1] <- data$config
    
    # Correlation heatmap
    corr_heatmap <- corr_heatmap(data = merged_data)
    height = 1 + 0.5 * ncol(merged_data)
    ggsave(paste0(sim, '_spearman_corr_heatmap_', node_metric, '.png'), corr_heatmap, width = height+2, height = height)
    
    # Rank heatmap
    if (data_type == 'simulation'){
      rank_heatmap <- rank_heatmap(data = merged_data, gt_table = gt)
      width = 5.5
      height = 0.25 * nrow(gt)
      ggsave(paste0(sim, '_rank_heatmap_', node_metric, '.png'), rank_heatmap, width = width, height = height)
    }
  }
  
  # Ranking algorithms
  for (ranking_alg in ranking_alg_subset){
    # Filter for metric
    data <- node_rankings[edge_metric==metric, ]
    
    # Read in rankings
    ranking_list <- lapply(data$ranking_file, fread)
    
    # Create config column and rename rankings
    data[, config := paste(node_metric, edge_metric, sep = ", ")]
    names(ranking_list) <- data$config
    
    # Merge ranking data
    merged_data <- reduce(ranking_list, full_join, by = "node")
    colnames(merged_data)[-1] <- data$config
    
    # Correlation heatmap
    corr_heatmap <- corr_heatmap(data = merged_data)
    height = 1 + 0.5 * ncol(merged_data)
    ggsave(paste0(sim, '_spearman_corr_heatmap_', ranking_alg, '.png'), corr_heatmap, width = height+2, height = height)
    
    # Rank heatmap
    if (data_type == 'simulation'){
      rank_heatmap <- rank_heatmap(data = merged_data, gt_table = gt)
      width = 5.5
      height = 0.25 * nrow(gt)
      ggsave(paste0(sim, '_rank_heatmap_', ranking_alg, '.png'), rank_heatmap, width = width, height = height)
    }
  }
  
}




