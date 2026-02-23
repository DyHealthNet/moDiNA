#!/usr/bin/env Rscript

######## ------------- Libraries ------------- ########
.libPaths("/nfs/home/students/a.raithel/miniconda3/envs/modina_eval_env/lib/R/library")
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
library(stringr)
library(tibble)

######## ------------- Utils ------------- ########

# Node and edge metrics, algorithms
node_metrics <- c("WDC-P", "WDC-E", "DC-P", "DC-E", "PRC-P", "PRC-E", "STC", "None")
node_metrics_colors <- c("#8DD3C7", "#41B6C4", "#F1B6DA", "#DD1C77","#CCCCCC", "#636363", "#FFD700","#FF6B6B")
names(node_metrics_colors) <- node_metrics

edge_metrics <- c("pre-CS", "post-CS", "pre-LS", "post-LS", "pre-P", "post-P", "pre-E", "post-E", "pre-PE", "post-PE", "int-IS", "None")
edge_metrics_colors <- c("#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C","#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00", "#CAB2D6","#6A3D9A" , "#FFFF99", "#B15928")
names(edge_metrics_colors) <- edge_metrics

algorithms <- c('absDimontRank', 'DimontRank', 'PageRank', 'PageRank+', 'direct_node', 'direct_edge')
algorithm_colors <- c("#4B0082", "#9370DB", "#004225", "#228B22", "#8B0000", "#FF7F50")
names(algorithm_colors) <- algorithms

# Calculate AUC, TPR, and FPR
calculate_ROC_statistics <- function(ground_truth_file, ranking_file, node_ranking = TRUE) {
  # Read files
  ground_truth <- fread(ground_truth_file)
  ranking <- fread(ranking_file)
  
  # Sort ranking
  ranking <- ranking[order(rank)]
  
  # Vector of labels (1 for GT nodes, 0 for non-GT nodes)
  if(node_ranking){
    ranking[, is_ground_truth := ifelse(node %in% ground_truth$node, 1, 0)]
  } else {
    ranking[, is_ground_truth := ifelse(node %in% ground_truth$edge, 1, 0)]
  }
  
  # Calculate ROC
  roc_obj <- roc(ranking$is_ground_truth, ranking$rank, quiet = TRUE)
  
  return(roc_obj)
}


# Create averaged ROC curves
roc_curve <- function(data, variable_param, variable_colors){
  data_long <- data %>%
    select(id, {{variable_param}}, tpr, fpr) %>%
    unnest(cols = c(tpr, fpr))
  
  # Set fpr grid and interpolate
  fpr_grid <- seq(0, 1, length.out = 200)
  data_interp <- data_long %>%
    group_by({{variable_param}}, id) %>%
    summarise(
      interp = list(
        approx(
          x = fpr,
          y = tpr,
          xout = fpr_grid,
          ties = "ordered"
        )$y
      ),
      .groups = "drop"
    )
  
  data_interp <- data_interp %>%
    mutate(fpr = list(fpr_grid)) %>%
    unnest(c(fpr, interp)) %>%
    rename(tpr = interp)
  
  # Average tpr
  data_mean <- data_interp %>%
    group_by({{variable_param}}, fpr) %>%
    summarise(
      mean_tpr = mean(tpr, na.rm = TRUE),
      sd_tpr = sd(tpr, na.rm = TRUE),
      .groups = "drop"
    )
  
  # Mean AUC
  data <- data %>%
    group_by({{variable_param}}) %>%
    summarise(
      mean_auc = mean(auc),
      sd_auc   = sd(auc)
    )
  
  auc_labels <- data %>%
    mutate(
      config_label = paste0(
        {{variable_param}},
        " (AUC = ",
        round(mean_auc, 3),
        " ± ",
        round(sd_auc, 3),
        ")"
      )
    )
  
  data_mean <- data_mean %>%
    left_join(
      auc_labels %>% select({{variable_param}}, config_label),
      by = as_label(enquo(variable_param))
    )
  
  # Color mapping
  color_map <- auc_labels %>%
    mutate(color = variable_colors[node_metric]) %>%
    select(config_label, color) %>%
    deframe()
  
  # Plot
  p <- ggplot(data_mean, aes(x = fpr, y = mean_tpr, color = config_label, fill = config_label)) +
    geom_line(linewidth = 1) +
    geom_ribbon(aes(ymin = mean_tpr - sd_tpr, ymax = mean_tpr + sd_tpr), alpha = 0.2, color = NA) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
    theme_minimal() +
    labs(
      title = paste0("ROC Curve: ", metric, ', ', ranking_alg),
      x = "False Positive Rate",
      y = "True Positive Rate",
      color = str_to_title(str_replace_all(as_label(enquo(variable_param)), "_", " "))
    ) +
    scale_color_manual(values = color_map) +
    scale_fill_manual(values = color_map) +
    guides(fill = "none")
  
  return(p)
}


# Create recall vs. rank plot
recall_and_enrichment <- function(data, variable_param, variable_colors, rank_of_interest=10){
  data_long <- data %>%
    select(id, {{variable_param}}, tpr) %>%
    unnest(cols = c(tpr))
  
  # Create rank column
  data_long <- data_long %>%
    group_by({{variable_param}}, id) %>%
    mutate(rank = seq_along(tpr),
           max_rank = max(rank)) %>%
    ungroup()
  
  # Filter for rank of interest
  data_rank <- data_long %>%
    filter(rank == rank_of_interest)
  
  # Add enrichment in comparison to a random ranking
  data_rank <- data_rank %>%
    mutate(
      random_tpr = rank / max_rank,
      enrichment = tpr / random_tpr,
      config_label = as.character({{variable_param}})
    )
  
  # Average per rank
  data_mean <- data_long %>%
    group_by({{variable_param}}, rank) %>%
    summarise(
      mean_tpr = mean(tpr, na.rm = TRUE),
      sd_tpr = sd(tpr, na.rm = TRUE),
      .groups = "drop"
    )
  
  data_mean <- data_mean %>%
    mutate(config_label = as.character({{variable_param}}))
  
  # Color mapping
  unique_labels <- unique(data_mean$config_label)
  color_map <- variable_colors[unique_labels]
  
  # Plot
  p_recall <- ggplot(data_mean, aes(x = rank, y = mean_tpr, color = config_label, fill = config_label)) +
    geom_line(linewidth = 1) +
    geom_ribbon(aes(ymin = mean_tpr - sd_tpr, ymax = mean_tpr + sd_tpr), alpha = 0.2, color = NA) +
    theme_minimal() +
    labs(
      x = "Rank",
      y = "Recall (TPR)",
      color = str_to_title(str_replace_all(as_label(enquo(variable_param)), "_", " "))
    ) +
    scale_color_manual(values = color_map) +
    scale_fill_manual(values = color_map) +
    guides(fill = "none")
  
  p_enrichment <- ggplot(data_rank,
                       aes(x = config_label,
                           y = enrichment,
                           fill = config_label)) +
    geom_boxplot(alpha = 0.7, outlier.shape = NA) +
    geom_jitter(width = 0.15, alpha = 0.4) +
    theme_minimal() +
    scale_fill_manual(values = color_map) +
    labs(
      x = str_to_title(str_replace_all(as_label(enquo(variable_param)), "_", " ")),
      y = paste0("Enrichment at Rank ", rank_of_interest)
    ) +
    guides(fill = "none")
  
  return(list(
    recall_curve = p_recall,
    enrichment_boxplot = p_enrichment
  ))
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

# Store edge ranking independently
edge_ranking_dt <- summary_dt[algorithm == "direct_edge",]

# Remove direct_edge ranking
summary_dt <- summary_dt[summary_dt$algorithm != "direct_edge",]

# Calculate ROC statistics for each row
summary_dt[, roc_obj := mapply(calculate_ROC_statistics, 
                           ground_truth_nodes, 
                           ranking_file,
                           SIMPLIFY = FALSE)]

# Create columns for tpr and fpr
summary_dt[, `:=`(
  tpr = lapply(roc_obj, function(r) rev(r$sensitivities)),
  fpr = lapply(roc_obj, function(r) rev(1 - r$specificities)),
  auc = sapply(roc_obj, function(r) as.numeric(r$auc))
)]

# Number of ground truth nodes
ground_truth <- summary_dt$ground_truth_nodes[1]
ground_truth <- fread(ground_truth)
n_gt_nodes <- nrow(ground_truth)

#if (nrow(edge_ranking_dt) > 0) {
#  edge_ranking_dt[, roc_obj := mapply(calculate_ROC_statistics, 
#                                  ground_truth_edges, 
#                                  ranking_file,
#                                  node_ranking = FALSE,
#                                  SIMPLIFY = FALSE)]
#  
#  # Create columns for tpr and fpr
#  edge_ranking_dt[, `:=`(
#    tpr = lapply(roc_obj, function(r) rev(r$sensitivities)),
#    fpr = lapply(roc_obj, function(r) rev(1 - r$specificities))
#  )]
  
#  summary_dt <- rbind(summary_dt, edge_ranking_dt)
#}

for(metric in edge_metrics) {
  # Subset data
  summary_dt_subset <- summary_dt[edge_metric == metric]
  if (nrow(summary_dt_subset) < 1){
    next
  }
  
  roc_list <- list()
  recall_list <- list()
  enrichment_list <- list()
  
  # Vary the algorithm
  for (ranking_alg in algorithms){
    data <- summary_dt_subset[algorithm == ranking_alg]
    if (nrow(data) < 1){
      next
    }
    
    roc <- roc_curve(data = data, variable_param = node_metric, variable_colors = node_metrics_colors)
    roc_list <- c(roc_list, list(roc))
    
    recall <- recall_and_enrichment(data = data, variable_param = node_metric, variable_colors = node_metrics_colors)$recall_curve
    recall_list <- c(recall_list, list(recall))
    
    enrichment <- recall_and_enrichment(data = data, variable_param = node_metric, variable_colors = node_metrics_colors, rank_of_interest=n_gt_nodes)$enrichment_boxplot
    enrichment_list <- c(enrichment_list, list(enrichment))
    
  }
  
  # Combine plots
  combined_roc <- wrap_plots(roc_list, ncol = 1)
  ggsave(paste0('ROC_curves_', metric, '_algorithms.png'), combined_roc, width = 6, height = 3 * length(roc_list))
  
  combined_recall <- wrap_plots(recall_list, ncol = 1)
  ggsave(paste0('recall_curve_', metric, '_algorithms.png'), combined_recall, width = 6, height = 3 * length(recall_list))
  
  combined_enrichment <- wrap_plots(enrichment_list, ncol = 1)
  ggsave(paste0('enrichment_boxplot_', metric, '_algorithms.png'), combined_enrichment, width = 1 + length(enrichment_list), height = 3 * length(enrichment_list))
  
  roc_list <- list()
  recall_list <- list()
  enrichment_list <- list()
  
  # Vary the node metric
  for (met in node_metrics){
    data <- summary_dt_subset[node_metric == met]
    if (nrow(data) < 1){
      next
    }
    
    roc <- roc_curve(data = data, variable_param = algorithm, variable_colors = algorithm_colors)
    roc_list <- c(roc_list, list(roc))
    
    recall <- recall_and_enrichment(data = data, variable_param = algorithm, variable_colors = algorithm_colors)$recall_curve
    recall_list <- c(recall_list, list(recall))
    
    enrichment <- recall_and_enrichment(data = data, variable_param = algorithm, variable_colors = algorithm_colors, rank_of_interest=n_gt_nodes)$enrichment_boxplot
    enrichment_list <- c(enrichment_list, list(enrichment))
  }
  
  # Combine plots
  combined_roc <- wrap_plots(roc_list, ncol = 1)
  ggsave(paste0('ROC_curves_', metric, '_node_metrics.png'), combined_roc, width = 6, height = 3 * length(roc_list))
  
  combined_recall <- wrap_plots(recall_list, ncol = 1)
  ggsave(paste0('recall_curve_', metric, '_node_metrics.png'), combined_recall, width = 6, height = 3 * length(recall_list))
  
  combined_enrichment <- wrap_plots(enrichment_list, ncol = 1)
  ggsave(paste0('enrichment_boxplot_', metric, '_node_metrics.png'), combined_enrichment, width = 1 + length(enrichment_list), height = 3 * length(enrichment_list))
}

for(metric in node_metrics) {
  # Subset data
  summary_dt_subset <- summary_dt[node_metric == metric]
  if (nrow(summary_dt_subset) < 1){
    next
  }
  
  roc_list <- list()
  recall_list <- list()
  enrichment_list <- list()
  
  # Vary the algorithm
  for (ranking_alg in algorithms){
    data <- summary_dt_subset[algorithm == ranking_alg]
    if (nrow(data) < 1){
      next
    }
    
    roc <- roc_curve(data = data, variable_param = edge_metric, variable_colors = edge_metrics_colors)
    roc_list <- c(roc_list, list(roc))
    
    recall <- recall_and_enrichment(data = data, variable_param = edge_metric, variable_colors = edge_metrics_colors)$recall_curve
    recall_list <- c(recall_list, list(recall))
    
    enrichment <- recall_and_enrichment(data = data, variable_param = edge_metric, variable_colors = edge_metrics_colors, rank_of_interest=n_gt_nodes)$enrichment_boxplot
    enrichment_list <- c(enrichment_list, list(enrichment))
  }
  
  # Combine plots
  combined_roc <- wrap_plots(roc_list, ncol = 1)
  ggsave(paste0('ROC_curves_', metric, '_algorithms.png'), combined_roc, width = 6, height = 3 * length(roc_list))
  
  combined_recall <- wrap_plots(recall_list, ncol = 1)
  ggsave(paste0('recall_curve_', metric, '_algorithms.png'), combined_recall, width = 6, height = 3 * length(recall_list))
  
  combined_enrichment <- wrap_plots(enrichment_list, ncol = 1)
  ggsave(paste0('enrichment_boxplot_', metric, '_algorithms.png'), combined_enrichment, width = 1 + length(enrichment_list), height = 3 * length(enrichment_list))
  
  
  roc_list <- list()
  recall_list <- list()
  enrichment_list <- list()
  
  # Vary the edge metric
  for (met in edge_metrics){
    data <- summary_dt_subset[edge_metric == met]
    if (nrow(data) < 1){
      next
    }
    
    roc <- roc_curve(data = data, variable_param = algorithm, variable_colors = algorithm_colors)
    roc_list <- c(roc_list, list(roc))
    
    recall <- recall_and_enrichment(data = data, variable_param = algorithm, variable_colors = algorithm_colors)$recall_curve
    recall_list <- c(recall_list, list(recall))
    
    enrichment <- recall_and_enrichment(data = data, variable_param = algorithm, variable_colors = algorithm_colors, rank_of_interest=n_gt_nodes)$enrichment_boxplot
    enrichment_list <- c(enrichment_list, list(enrichment))
  }
  
  # Combine plots
  combined_roc <- wrap_plots(roc_list, ncol = 1)
  ggsave(paste0('ROC_curves_', metric, '_edge_metrics.png'), combined_roc, width = 6, height = 3 * length(roc_list))
  
  combined_recall <- wrap_plots(recall_list, ncol = 1)
  ggsave(paste0('recall_curve_', metric, '_edge_metrics.png'), combined_recall, width = 6, height = 3 * length(recall_list))
  
  combined_enrichment <- wrap_plots(enrichment_list, ncol = 1)
  ggsave(paste0('enrichment_boxplot_', metric, '_edge_metrics.png'), combined_enrichment, width = 1 + length(enrichment_list), height = 3 * length(enrichment_list))
}

for(ranking_alg in algorithms) {
  # Subset data
  summary_dt_subset <- summary_dt[algorithm == ranking_alg]
  if (nrow(summary_dt_subset) < 1){
    next
  }
  
  roc_list <- list()
  recall_list <- list()
  enrichment_list <- list()
  
  # Vary the edge metric
  for (met in edge_metrics){
    data <- summary_dt_subset[edge_metric == met]
    if (nrow(data) < 1){
      next
    }
    
    roc <- roc_curve(data = data, variable_param = node_metric, variable_colors = node_metrics_colors)
    roc_list <- c(roc_list, list(roc))
    
    recall <- recall_and_enrichment(data = data, variable_param = node_metric, variable_colors = node_metrics_colors)$recall_curve
    recall_list <- c(recall_list, list(recall))
    
    enrichment <- recall_and_enrichment(data = data, variable_param = node_metric, variable_colors = node_metrics_colors, rank_of_interest=n_gt_nodes)$enrichment_boxplot
    enrichment_list <- c(enrichment_list, list(enrichment))
  }
  
  # Combine plots
  combined_roc <- wrap_plots(roc_list, ncol = 1)
  ggsave(paste0('ROC_curves_', ranking_alg, '_edge_metrics.png'), combined_roc, width = 6, height = 3 * length(roc_list))
  
  combined_recall <- wrap_plots(recall_list, ncol = 1)
  ggsave(paste0('recall_curve_', ranking_alg, '_edge_metrics.png'), combined_recall, width = 6, height = 3 * length(recall_list))
  
  combined_enrichment <- wrap_plots(enrichment_list, ncol = 1)
  ggsave(paste0('enrichment_boxplot_', metric, '_edge_metrics.png'), combined_enrichment, width = 1 + length(enrichment_list), height = 3 * length(enrichment_list))
  
  roc_list <- list()
  recall_list <- list()
  enrichment_list <- list()
  
  # Vary the node metric
  for (met in node_metrics){
    data <- summary_dt_subset[node_metric == met]
    if (nrow(data) < 1){
      next
    }
    
    roc <- roc_curve(data = data, variable_param = edge_metric, variable_colors = edge_metrics_colors)
    roc_list <- c(roc_list, list(roc))
    
    recall <- recall_and_enrichment(data = data, variable_param = edge_metric, variable_colors = edge_metrics_colors)$recall_curve
    recall_list <- c(recall_list, list(recall))
    
    enrichment <- recall_and_enrichment(data = data, variable_param = edge_metric, variable_colors = edge_metrics_colors, rank_of_interest=n_gt_nodes)$enrichment_boxplot
    enrichment_list <- c(enrichment_list, list(enrichment))
  }
  
  # Combine plots
  combined_roc <- wrap_plots(roc_list, ncol = 1)
  ggsave(paste0('ROC_curves_', ranking_alg, '_node_metrics.png'), combined_roc, width = 6, height = 3 * length(roc_list))
  
  combined_recall <- wrap_plots(recall_list, ncol = 1)
  ggsave(paste0('recall_curve_', ranking_alg, '_node_metrics.png'), combined_recall, width = 6, height = 3 * length(recall_list))
  
  combined_enrichment <- wrap_plots(enrichment_list, ncol = 1)
  ggsave(paste0('enrichment_boxplot_', metric, '_node_metrics.png'), combined_enrichment, width = 1 + length(enrichment_list), height = 3 * length(enrichment_list))
}

