# Jitter plots of differential scores
library(patchwork)
library(ggplot2)
library(dplyr)
library(readr)
library(stringr)
library(data.table)
library(colorspace)

# Valid metrics
edge_metrics_subset = c('pre-P', 'post-P', 'pre-E', 'post-E', 'pre-CS', 'post-CS', 'int-IS', 'pre-LS', 'post-LS', 'pre-PE', 'post-PE')
node_metrics_subset = c('DC-P', 'DC-E', 'STC', 'PRC-P', 'PRC-E', 'WDC-P', 'WDC-E')

# Colors
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

# Run params
# TODO: read in summary.csv and additional params
summary <- as.data.table()
study <- #('sim' or 'real') 
simulations <- max(summary$id, na.rm = TRUE)

# Helper functions
# Create dictionary for ground truth edges
create_gt_edge_dict <- function(path){
  gt_table <- read_csv(path, col_names=c("node", "description"))
  gt_table$description <- str_trim(gt_table$description)
  
  gt_dict <- list()
  pair <- c()
  
  for (j in seq_len(nrow(gt_table))) {
    row <- gt_table[j, ]
    if (str_detect(row$description, "corr\\.")) {
      pair <- c(pair, row$node)
      if (length(pair) == 2) {
        edge <- paste(sort(pair), collapse = "_")
        gt_dict[[edge]] <- row$description
        pair <- c()
      }
    }
  }
  
  return(gt_dict)
}

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

# Jitter plot of differential scores
diff_scores_jitter <- function(configs, weight, mode = 'edges', study = 'sim'){
  if (study == 'sim'){
    
    # Loop through all provided configurations
    all_data <- list()
    for (i in seq_along(configs)) {
      config <- configs[[i]]
      
      # Extract scores and paths
      scores <- read_csv(config$scores)
      gt_path = config$groundtruth
      
      # Read ground truths
      gt_table <- read_csv(config$groundtruth, col_names=c("node", "description"))
      gt_table$description <- str_trim(gt_table$description)
      
      if (mode == "edges") {
        # Create dictionary containing all ground truth edges
        gt_dict <- create_gt_edge_dict(config$groundtruth)
        
        color_palette <- ground_truth_palette_edges
        color_palette_dark <- ground_truth_palette_edges_dark
      } else{
        # Create dictionary containing all ground truth nodes
        gt_table <- read_csv(config$groundtruth, col_names=c("node", "description"))
        gt_table$description <- str_trim(gt_table$description)
        gt_dict <- setNames(gt_table$description, gt_table$node)
        
        color_palette <- ground_truth_palette
        color_palette_dark <- ground_truth_palette_dark
        
        colnames(scores)[1] <- "node"
      }
      
      # Map scores using the ground truth dictionary and add gt column to scores
      gt <- t(apply(scores, 1, get_gt_info, mode=mode, gt_dict=gt_dict))
      scores <- cbind(scores, as.data.frame(gt))
      
      # Collect all data
      all_data[[i]] <- scores %>% select(all_of(weight), groundtruth, description)
    }
    
    # Combine all data
    combined_data <- bind_rows(all_data)
    combined_data$description <- factor(combined_data$description, levels = names(color_palette))
    
    # Point plot
    p <- ggplot(combined_data, aes(x=1, y = .data[[weight]], color = description)) +
      geom_jitter(
        aes(fill = description, color = description),
        shape = 21, size = 2.0, alpha = 0.6, stroke = 1,
        position = position_jitterdodge(jitter.width = 0.15, jitter.height = 0, dodge.width = 0.5)
      ) +
      guides(color = "none") +
      labs(
        y = weight,
        x = NULL,
        fill = "Ground truth"
      ) +
      scale_fill_manual(values = color_palette) +
      scale_color_manual(values = color_palette_dark) + 
      theme_minimal() +
      theme(legend.position = "right", 
            panel.grid.major.x = element_blank(),
            panel.grid.minor.x = element_blank(),
            axis.title.x = element_blank(),
            axis.text.x  = element_blank(),
            axis.ticks.x = element_blank(),
            axis.title.y = element_text(size = 16),
            axis.text.y  = element_text(size = 14))
  } else if (study == 'real'){
    # Loop through all provided configurations
    all_data <- list()
    for (i in seq_along(configs)) {
      config <- configs[[i]]
      
      # Extract scores
      scores <- read_csv(config$scores)
      
      if (mode == "nodes") {
        colnames(scores)[1] <- "node"
      }
      
      # Collect all data
      all_data[[i]] <- scores %>% select(all_of(weight))
    }
    
    # Combine all data
    combined_data <- bind_rows(all_data)
    
    # Point plot
    p <- ggplot(combined_data, aes(x=1, y = .data[[weight]])) +
      geom_jitter(
        shape = 21, size = 2.0, alpha = 0.6, stroke = 1, fill = 'gray70', color = 'gray50',
        position = position_jitterdodge(jitter.width = 0.3, jitter.height = 0, dodge.width = 0.5)
      ) +
      guides(color = "none") +
      labs(
        y = weight,
        x = NULL,
      ) +
      theme_minimal() +
      theme(panel.grid.major.x = element_blank(),
            panel.grid.minor.x = element_blank(),
            axis.title.x = element_blank(),
            axis.text.x  = element_blank(),
            axis.ticks.x = element_blank(),
            axis.title.y = element_text(size = 16),
            axis.text.y  = element_text(size = 14))
    
  }
  return(p)
}

# Plot differential scores  
if (study == 'sim'){
  # Edge metrics
  all_plots <- list()
  for (metric in edge_metrics_subset){
    configs = c()
    
    # Get scores and ground truth paths for each simulation and append to configs list
    for (i in 1:simulations){
      scores = summary[id == i, edge_metrics_file]
      groundtruth = summary[id == i, ground_truth_file]
      if (file.exists(scores)){
        configs <- c(configs, list(list(
          groundtruth = groundtruth,
          scores = scores
        )))
      }
    }
    if (identical(configs, c())) next
    
    # Plot and append to list
    p = diff_scores_jitter(configs, metric, mode='edges', study='sim')
    all_plots <- append(all_plots, list(p))
  }
  
  # Combine all plots
  edge_metrics_plot <- wrap_plots(all_plots, ncol = 1) +
    theme(legend.position = "bottom",
          legend.text = element_text(size = 14),
          legend.title = element_text(size = 16))
  
  ggsave('edge_metrics_point_plots.png',
         edge_metrics_plot, width = 8, height = 3 * length(all_plots))
  
  # Node metrics
  all_plots <- list()
  for (metric in node_metrics_subset){
    configs = c()
    
    # Get scores and ground truth paths for each simulation and append to configs list
    for (i in 1:simulations){
      scores = summary[id == i, node_metrics_file]
      groundtruth = summary[id == i, ground_truth_file]
      if (file.exists(scores)){
        configs <- c(configs, list(list(
          groundtruth = groundtruth,
          scores = scores
        )))
      }
    }
    if (identical(configs, c())) next
    
    # Plot and append to list
    p = diff_scores_jitter(configs, metric, mode='nodes', study='sim')
    all_plots <- append(all_plots, list(p))
  }
  
  # Combine all plots
  node_metrics_plot <- wrap_plots(all_plots, ncol = 1) +
    theme(legend.position = "bottom",
          legend.text = element_text(size = 14),
          legend.title = element_text(size = 16))
  
  ggsave("node_metrics_point_plots.png",
         node_metrics_plot, width = 8, height = 3 * length(all_plots))
} else if (study == 'real'){
  # Edge metrics
  all_plots <- list()
  for (metric in edge_metrics_subset){
    configs = c()
    
    # Get score paths for each simulation and append to configs list
    for (i in 1:simulations){
      scores = summary[id == i, edge_metrics_file]
      if (file.exists(scores)){
        configs <- c(configs, list(list(scores = scores)))
      }
    }
    if (identical(configs, c())) next
    
    # Plot and append to list
    p = diff_scores_jitter(configs, metric, mode='edges', study='real')
    all_plots <- append(all_plots, list(p))
  }
  
  # Combine all plots
  edge_metrics_plot <- wrap_plots(all_plots, ncol = 1)
  
  ggsave("edge_metrics_point_plots.png",
         edge_metrics_plot, width = 3, height = 3 * length(all_plots))
  
  # Node metrics
  all_plots <- list()
  for (metric in node_metrics_subset){
    configs = c()
    
    # Get score paths for each simulation and append to configs list
    for (i in 1:simulations){
      scores = summary[id == i, node_metrics_file]
      if (file.exists(scores)){
        configs <- c(configs, list(list(scores = scores)))
      }
    }
    if (identical(configs, c())) next
    
    # Plot and append to list
    p = diff_scores_jitter(configs, metric, mode='nodes', study='real')
    all_plots <- append(all_plots, list(p))
  }
  
  # Combine all plots
  node_metrics_plot <- wrap_plots(all_plots, ncol = 1)
  
  ggsave("node_metrics_point_plots.png",
         node_metrics_plot, width = 3, height = 3 * length(all_plots))
}
