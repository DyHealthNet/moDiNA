# Helper functions for evaluation

# Create dictionary for ground truth edges
create_gt_edge_dict <- function(path, mode){
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

