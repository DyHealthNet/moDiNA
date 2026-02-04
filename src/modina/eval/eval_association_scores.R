# Jitter plots of association scores

source('./eval_config.R')
source('./eval_helpers.R')

# Params
# TODO: read in summary.csv and additional params
summary <- as.data.table()
study <- #('sim' or 'real') 
simulations <- nrow(summary)
name1 <- 
name2 <- 

if (study == 'sim'){
  data <- data.table()
  
  for (i in 1:simulations){
    # Get paths
    gt_path <- summary[id == i, ground_truth_file]
    scores1 <- summary[id == i, network_context_1]
    scores2 <- summary[id == i, network_context_2]
    
    # Create dictionary with ground truth edges
    gt_dict <- create_gt_edge_dict(gt_path)
    
    # Read scores
    scores_a <- as.data.table(read.csv(scores1))
    scores_b <- as.data.table(read.csv(scores2))
    
    # Add context column and combine
    scores_a[, context := name1]
    scores_b[, context := name2]
    scores_ab <- rbind(scores_a, scores_b)
    
    # Map scores using the ground truth dictionary and add gt column to scores
    gt <- t(apply(scores_ab, 1, get_gt_info, mode='edges', gt_dict=gt_dict))
    scores_ab <- cbind(scores_ab, as.data.frame(gt))

    data <- rbind(data, scores_ab)
  }
  
  # Prepare data for plotting
  data <- data[, id := paste(label1, label2, sep = "_")]
  #data[description == "non-ground truth", id := paste0("nonGT_", .I)]
  data <- data %>%
    mutate(
      base_x = case_when(
        context == name1 & description != 'non-ground truth' ~ 1.3,
        context == name1 & description == 'non-ground truth' ~ 2.3,
        context == name2 & description != 'non-ground truth' ~ 1.8,
        context == name2 & description == 'non-ground truth' ~ 2.8,
        TRUE ~ NA_real_
      ),
      x_jitter = base_x + runif(n(), -0.15, 0.15)
    )
  
  label_positions <- data.frame(
    context = c(name1, name2, name1, name2),
    x = c(1.3, 1.8, 2.3, 2.8)
  )
  
  # Plot and save
  plot_p <- assoc_scores_jitter(data, 'P', study='sim', label_positions=label_positions)
  plot_e <- assoc_scores_jitter(data, 'E', study='sim', label_positions=label_positions)
  
  n_tests <- data[, uniqueN(test_type)]
  
  # TODO: adjust path to save plots
  ggsave(paste0(project_path, '/results/association_scores_raw-P.png'), plot_p, width = 7, height = 3 * n_tests)
  ggsave(paste0(project_path, '/results/association_scores_raw-E.png'), plot_e, width = 7, height = 3 * n_tests)
  
} else if (study == 'real'){
  # Get paths
  scores1 <- summary[id == 1, network_context_1]
  scores2 <- summary[id == 1, network_context_2]
  
  # Read scores
  scores_a <- as.data.table(read.csv(scores1))
  scores_b <- as.data.table(read.csv(scores2))
  
  # Add context column and combine
  scores_a[, context:=name1]
  scores_b[, context:=name2]
  scores_ab <- rbind(scores_a, scores_b)
  
  # Plot and save
  plot_p <- assoc_scores_jitter(scores_ab, 'P', study='real')
  plot_e <- assoc_scores_jitter(scores_ab, 'E', study='real')
  
  n_tests <- scores_ab[, uniqueN(test_type)]
  
  # TODO: adjust path to save plots
  ggsave(paste0(project_path, '/results/association_scores_raw-P.png'), plot_p, width = 4, height = 3 * n_tests)
  ggsave(paste0(project_path, '/results/association_scores_raw-E.png'), plot_e, width = 4, height = 3 * n_tests)
}


