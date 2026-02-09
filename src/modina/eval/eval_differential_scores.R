# Jitter plots of differential scores

source('./eval_config.R')
source('./eval_helpers.R')

# Params
# TODO: read in summary.csv and additional params
summary <- as.data.table()
study <- #('sim' or 'real') 
simulations <- max(summary$id, na.rm = TRUE)
  
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
    p = diff_scores_jitter(configs, metric, mode='edges')
    all_plots <- append(all_plots, list(p))
  }
  
  # Combine all plots
  edge_metrics_plot <- wrap_plots(all_plots, ncol = 1) +
    theme(legend.position = "bottom",
          legend.text = element_text(size = 14),
          legend.title = element_text(size = 16))
  
  # TODO: Adjust path to save plots
  ggsave(paste0(project_path, "/results/edge_metrics_point_plots.png"),
         edge_metrics_plot, width = 10, height = 3 * length(all_plots))
  
  
  # Node metrics
  all_plots <- list()
  for (metric in node_metrics_subset){
    configs = c()
    
    # Get scores and ground truth paths for each simulation and append to configs list
    for (sim in 1:simulations){
      scores = summary[id == i, node_metrics_file]
      groundtruth = summary[id == i, ground_truth_file]
      if (file.exists(scores)){
        configs <- c(configs, list(list(
          level = i,
          groundtruth = groundtruth,
          scores = scores
        )))
      }
    }
    if (identical(configs, c())) next
    
    # Plot and append to list
    p = diff_scores_jitter(configs, metric, mode='nodes')
    all_plots <- append(all_plots, list(p))
  }
  
  # Combine all plots
  node_metrics_plot <- wrap_plots(all_plots, ncol = 1) +
    theme(legend.position = "bottom",
          legend.text = element_text(size = 14),
          legend.title = element_text(size = 16))
  
  # TODO: Adjust path to save plots
  ggsave(paste0(project_path, "/results/node_metrics_point_plots.png"),
         node_metrics_plot, width = 10, height = 3 * length(all_plots))
}
