#!/usr/bin/env Rscript

######## ------------- Libraries ------------- ########
.libPaths("/nfs/home/students/a.raithel/miniconda3/envs/modina_eval_env/lib/R/library")
library(GGally)
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

# Valid focus values
edge_metrics_subset = c('pre-P', 'post-P', 'pre-E', 'post-E', 'pre-CS', 'post-CS', 'int-IS', 'pre-LS', 'post-LS', 'pre-PE', 'post-PE')
node_metrics_subset = c('DC-P', 'DC-E', 'STC', 'PRC-P', 'PRC-E', 'WDC-P', 'WDC-E')
algorithms_subset = c('direct_node', 'PageRank', 'PageRank+', 'DimontRank', 'absDimontRank')


# Extract ground truth information from a row
get_gt_info <- function(row, mode, gt_dict) {
  if (mode == 'edges'){
    edge <- paste(sort(c(row['label1'], row['label2'])), collapse = "_")
    
    description <- ifelse(edge %in% names(gt_dict), gt_dict[[edge]], NA)
    gt <- !is.na(description)
    description <- ifelse(is.na(description), 'non-ground truth', description)
  }
  else{
    node <- row[['node']]
    description <- ifelse(node %in% names(gt_dict), gt_dict[[node]], NA)
    gt <- !is.na(description)
    description <- ifelse(is.na(description), 'non-ground truth', description)
  }
  
  return(c(groundtruth = gt, description = description))    
}


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
rank_heatmap <- function(data, gt_dict){
  # Extract ground truth nodes
  gt_nodes <- names(gt_dict)
  
  # Create ground truth annotation dataframe for heatmap
  gt_info <- as.data.table(data.frame(node = gt_nodes))
  gt_info <- gt_info[, groundtruth := gt_dict[match(node, names(gt_dict))]]
  
  # Prepare heatmap matrix
  m <- as.matrix(data[, -1])
  rownames(m) <- data$node
  
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
    theme(legend.position = "right",
          panel.grid = element_blank(),
          axis.ticks = element_blank()
  )
  
  return(annotated_heatmap)
}


# Create a parallel coordinates plot to compare the rankings
par_coord <- function(data, metric, gt_dict){
  data <- as.data.table(data)
  
  # Replace NAs by last rank
  last_rank <- nrow(data)
  ranking_cols <- setdiff(colnames(data), "node")
  for (col in ranking_cols) {
    data[is.na(get(col)), (col) := last_rank]
  }
  
  # Add ground truth information
  gt_data <- t(apply(data, 1, get_gt_info, 
                     mode='nodes', gt_dict=gt_dict))
  data <- cbind(data, as.data.table(gt_data))
  
  # Get ranking columns
  ranking_cols <- setdiff(colnames(data),
                          c("node", "groundtruth", "description"))
  
  columns = which(colnames(data) %in% ranking_cols)
  
  # Plot
  p <- ggparcoord(
    data = data,
    columns = columns,
    groupColumn = 'description',
    alphaLines = 1.0,
    scale = "globalminmax"
  ) +
    labs(x='', y='Rank', color='Ground Truth') +
    scale_color_manual(values = ground_truth_palette) +
    theme_minimal() +
    theme(
      axis.text.y  = element_text(size = 18),
      axis.title.y = element_text(size = 18),
      legend.text  = element_text(size = 18),
      legend.title = element_text(size = 20),
      axis.text.x  = element_text(size = 18, angle = 45, hjust = 1)
    )
  
  return(p)
}

######## ------------- Argument parser ------------- ########

#parser <- ArgumentParser(description='Ranking Similarity')
#parser$add_argument('summary_file', 
#                    help='Input summary data file storing all generated configurations and their results.')
#parser$add_argument('data_type', help = 'Type of data: simulation or real')
#args <- parser$parse_args()

#summary_file <- args$summary_file
#data_type <- args$data_type

summary_file <- '/nfs/proj/a.raithel/thesis/data/nf_pipeline/out_file/summary.csv'
data_type <- 'simulation'

######## ------------- Process data ------------- ########

summary_dt <- fread(summary_file)
simulations <- max(summary_dt$id, na.rm = TRUE)

summary_dt <- unique(summary_dt[, c("id", "edge_metric", "node_metric", "algorithm", "ranking_file", "ground_truth_nodes")])

######## ------------- Plotting ------------- ########

for (sim in 1:simulations){
  sim_summary <- summary_dt[id == sim, ]
  gt_table <- fread(unique(sim_summary[, ground_truth_nodes]))
  gt_dict <- setNames(gt_table$description, gt_table$node)
  node_rankings <- sim_summary[algorithm!='direct_edge', ]
  
  # Edge metrics
  for (metric in edge_metrics_subset){
    # Filter for metric
    data <- node_rankings[edge_metric==metric, ]
    if (nrow(data) < 2){
      next
    }
    
    # Read in rankings
    ranking_list <- lapply(data$ranking_file, fread)
    
    # Create config column and rename rankings
    data[, config := paste(node_metric, algorithm, sep = ", ")]
    names(ranking_list) <- data$config
    
    # Merge ranking data
    merged_data <- reduce(ranking_list, full_join, by = "node")
    colnames(merged_data)[-1] <- data$config
    
    # Correlation heatmap
    heatmap <- corr_heatmap(data = merged_data)
    height = 1 + 0.5 * ncol(merged_data)
    ggsave(paste0(sim, '_spearman_corr_heatmap_', metric, '.png'), heatmap, width = height+2, height = height)
    
    # Plots only useful for simulated data
    if (data_type == 'simulation'){
      # Rank heatmap
      heatmap <- rank_heatmap(data = merged_data, gt_dict = gt_dict)
      width = 5.5
      height = 0.25 * nrow(gt_table)
      ggsave(paste0(sim, '_rank_heatmap_', metric, '.png'), heatmap, width = width, height = height)
      
      # Parallel coordinates plot
      parallel_coordinates <- par_coord(data = merged_data, metric = metric, gt_dict = gt_dict)
      width = 1.5 * ncol(merged_data)
      ggsave(paste0(sim, '_parallel_coordinates_', metric, '.png'), parallel_coordinates, width = width)
    }
  }
  
  # Node metrics
  for (metric in node_metrics_subset){
    # Filter for metric
    data <- node_rankings[node_metric==metric, ]
    if (nrow(data) < 2){
      next
    }
    
    # Read in rankings
    ranking_list <- lapply(data$ranking_file, fread)
    
    # Create config column and rename rankings
    data[, config := paste(edge_metric, algorithm, sep = ", ")]
    names(ranking_list) <- data$config
    
    # Merge ranking data
    merged_data <- reduce(ranking_list, full_join, by = "node")
    colnames(merged_data)[-1] <- data$config
    
    # Correlation heatmap
    heatmap <- corr_heatmap(data = merged_data)
    height = 1 + 0.5 * ncol(merged_data)
    ggsave(paste0(sim, '_spearman_corr_heatmap_', metric, '.png'), heatmap, width = height+2, height = height)
    
    # Plots only useful for simulated data
    if (data_type == 'simulation'){
      # Rank heatmap
      heatmap <- rank_heatmap(data = merged_data, gt_dict = gt_dict)
      width = 5.5
      height = 0.25 * nrow(gt_table)
      ggsave(paste0(sim, '_rank_heatmap_', metric, '.png'), heatmap, width = width, height = height)
      
      # Parallel coordinates plot
      parallel_coordinates <- par_coord(data = merged_data, metric = metric, gt_dict = gt_dict)
      width = 1.5 * ncol(merged_data)
      ggsave(paste0(sim, '_parallel_coordinates_', metric, '.png'), parallel_coordinates, width = width)
    }
  }
  
  # Ranking algorithms
  for (ranking_alg in algorithms_subset){
    # Filter for metric
    data <- node_rankings[algorithm==ranking_alg, ]
    if (nrow(data) < 2){
      next
    }
    
    # Read in rankings
    ranking_list <- lapply(data$ranking_file, fread)
    
    # Create config column and rename rankings
    data[, config := paste(node_metric, edge_metric, sep = ", ")]
    names(ranking_list) <- data$config
    
    # Merge ranking data
    merged_data <- reduce(ranking_list, full_join, by = "node")
    colnames(merged_data)[-1] <- data$config
    
    # Correlation heatmap
    heatmap <- corr_heatmap(data = merged_data)
    height = 1 + 0.5 * ncol(merged_data)
    ggsave(paste0(sim, '_spearman_corr_heatmap_', ranking_alg, '.png'), heatmap, width = height+2, height = height)
    
    # Plots only useful for simulated data
    if (data_type == 'simulation'){
      # Rank heatmap
      heatmap <- rank_heatmap(data = merged_data, gt_dict = gt_dict)
      width = 5.5
      height = 0.25 * nrow(gt_table)
      ggsave(paste0(sim, '_rank_heatmap_', ranking_alg, '.png'), heatmap, width = width, height = height)
      
      # Parallel coordinates plot
      parallel_coordinates <- par_coord(data = merged_data, metric = ranking_alg, gt_dict = gt_dict)
      width = 1.5 * ncol(merged_data)
      ggsave(paste0(sim, '_parallel_coordinates_', ranking_alg, '.png'), parallel_coordinates, width = width)
    }
  }
  
}




