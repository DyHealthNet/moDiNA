# Helper functions for evaluation

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


# Jitter plot of association scores
assoc_scores_jitter <- function(scores, metric, study, label_positions = NULL){
  if (study == 'sim'){
    if (metric == 'P'){
      p <- ggplot(scores, aes(x = x_jitter, y = raw.P, color = description)) +
        geom_point(aes(x = x_jitter, fill = description, color = description),
                   shape = 21, size = 2, alpha = 0.6, stroke = 1) +
        geom_line(aes(group = id), alpha = 0.2) +
        guides(color = "none") +
        labs(
          y = 'Adjusted p-value',
          x = "",
          fill = "Ground truth"
        ) +
        scale_fill_manual(values = ground_truth_palette_edges) +
        scale_color_manual(values = ground_truth_palette_edges_dark) + # Make circles a little darker
        theme_minimal() +
        theme(legend.position = "right", 
              panel.grid.major.x = element_blank(),
              axis.title.x = element_text(size = 12),  
              axis.title.y = element_text(size = 12),
              axis.text.x  = element_text(size = 10),
              axis.text.y  = element_text(size = 10),
              strip.text = element_text(size = 12),
              panel.spacing.y = unit(1.7, "lines"),
              panel.spacing.x = unit(0.1, "lines"),
              legend.text = element_text(size=10),
              legend.title = element_text(size=12),
              strip.background = element_rect(fill = "grey90", color = "black", linewidth = 0.5)) +
        facet_grid(test_type ~ ., scales = "free_y") +
        scale_x_continuous(
          breaks = label_positions$x,
          labels = label_positions$context
        )
    } else if (metric == 'E'){
      p <- ggplot(scores, aes(x = x_jitter, y = raw.E, color = description)) +
        geom_point(aes(x = x_jitter, fill = description, color = description),
                   shape = 21, size = 2, alpha = 0.6, stroke = 1) +
        geom_line(aes(group = id), alpha = 0.2) +
        guides(color = "none") +
        labs(
          y = 'Raw effect size',
          x = "",
          fill = "Ground truth"
        ) +
        scale_fill_manual(values = ground_truth_palette_edges) +
        scale_color_manual(values = ground_truth_palette_edges_dark) + # Make circles a little darker
        theme_minimal() +
        theme(legend.position = "right", 
              panel.grid.major.x = element_blank(),
              axis.title.x = element_text(size = 12),  
              axis.title.y = element_text(size = 12),
              axis.text.x  = element_text(size = 10),
              axis.text.y  = element_text(size = 10),
              strip.text = element_text(size = 12),
              panel.spacing.y = unit(1.7, "lines"),
              panel.spacing.x = unit(0.1, "lines"),
              legend.text = element_text(size=10),
              legend.title = element_text(size=12),
              strip.background = element_rect(fill = "grey90", color = "black", linewidth = 0.5)) +
        facet_grid(test_type ~ ., scales = "free_y") +
        scale_x_continuous(
          breaks = label_positions$x,
          labels = label_positions$context
        )
    }
  } else if (study == 'real'){
    if (metric == 'P'){
      p <- ggplot(scores, aes(x = factor(context), y = raw.P)) +
        geom_jitter(
          fill = 'grey90', 
          color = 'grey90',
          shape = 21, size = 2.0, alpha = 0.6, stroke = 1,
          position = position_jitter(width = 0.15, height = 0)
        ) +
        labs(
          y = 'Adjusted p-value',
          x = ""
        ) +
        theme_minimal() +
        theme(
          panel.grid.major.x = element_blank(),
          axis.title = element_text(size = 12),
          axis.text  = element_text(size = 10),
          strip.text = element_text(size = 12),
          panel.spacing.y = unit(1.7, "lines"),
          panel.spacing.x = unit(0.1, "lines"),
          strip.background = element_rect(fill = "grey90", color = "black", linewidth = 0.5)) +
        facet_grid(test_type ~ ., scales = "free_y") +
        scale_x_discrete(expand = expansion(add = 0.7))
    } else if (metric == 'E'){
      p <- ggplot(scores, aes(x = factor(context), y = raw.E)) +
        geom_jitter(
          fill = 'grey90', 
          color = 'grey90',
          shape = 21, size = 2.0, alpha = 0.6, stroke = 1,
          position = position_jitter(width = 0.15, height = 0)
        ) +
        labs(
          y = 'Raw effect size',
          x = ""
        ) +
        theme_minimal() +
        theme(
          panel.grid.major.x = element_blank(),
          axis.title = element_text(size = 12),
          axis.text  = element_text(size = 10),
          strip.text = element_text(size = 12),
          panel.spacing.y = unit(1.7, "lines"),
          panel.spacing.x = unit(0.1, "lines"),
          strip.background = element_rect(fill = "grey90", color = "black", linewidth = 0.5)) +
        facet_grid(test_type ~ ., scales = "free_y") +
        scale_x_discrete(expand = expansion(add = 0.7))
    }
  }
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

