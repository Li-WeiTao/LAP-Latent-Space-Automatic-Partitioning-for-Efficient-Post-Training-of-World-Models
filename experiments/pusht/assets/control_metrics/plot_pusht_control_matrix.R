#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(ggplot2))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg) == 1) {
  normalizePath(sub("^--file=", "", script_arg), winslash = "/")
} else {
  normalizePath("plot_pusht_control_matrix.R", winslash = "/")
}
out_dir <- dirname(script_path)
repo_root_env <- Sys.getenv("LAP_REPO_ROOT", unset = "")
repo_root <- if (nzchar(repo_root_env)) {
  normalizePath(repo_root_env, winslash = "/")
} else {
  normalizePath(file.path(out_dir, "../../../.."), winslash = "/")
}

inputs <- data.frame(
  horizon = c("Short horizon", "Long horizon"),
  path = c(
    file.path(repo_root, "experiments/pusht/matrix/matrix_summary.csv"),
    file.path(repo_root, "experiments/pusht/matrix_long/matrix_summary.csv")
  ),
  stringsAsFactors = FALSE
)

method_order <- c(
  "baseline", "joint3", "globalft50", "random", "kmeans", "spectral"
)
source_names <- c(
  baseline = "Official baseline",
  joint3 = "Joint-Continue 3ep",
  globalft50 = "Global-FT50",
  random = "Random-Voronoi",
  kmeans = "K-means++",
  spectral = "Spectral"
)
label_order <- c(
  "Official\nbaseline",
  "Joint-Continue\n3ep",
  "Global-FT (LAP)\n50ep",
  "Random-Voronoi\nK3-50",
  "K-means++\nK3-50",
  "Spectral\nK3-50"
)
label_by_id <- setNames(label_order, method_order)

read_summary <- function(horizon, path) {
  raw <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  rows <- lapply(method_order, function(method_id) {
    source_name <- unname(source_names[[method_id]])
    hit <- raw[raw$method == source_name, , drop = FALSE]
    if (nrow(hit) != 1) {
      stop("Expected exactly one row for method: ", source_name, " in ", path)
    }
    data.frame(
      horizon = horizon,
      method_id = method_id,
      method_label = unname(label_by_id[[method_id]]),
      mean_percent = hit$mean_percent,
      sd_percent = hit$sd_across_finetuning_seeds_percent,
      n_finetuning_seeds = hit$num_finetuning_seeds,
      n_partition_seeds = hit$num_partition_seeds,
      n_eval_seeds = hit$num_eval_seeds,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

summary_df <- do.call(
  rbind,
  Map(read_summary, inputs$horizon, inputs$path)
)
summary_df$method_id <- factor(summary_df$method_id, levels = method_order)
summary_df$method_label <- factor(summary_df$method_label, levels = label_order)
summary_df$x_position <- match(as.character(summary_df$method_id), method_order)
summary_df$lower <- summary_df$mean_percent - summary_df$sd_percent
summary_df$upper <- summary_df$mean_percent + summary_df$sd_percent

write.csv(
  summary_df,
  file.path(out_dir, "pusht_control_method_summary.csv"),
  row.names = FALSE
)

method_colors <- c(
  baseline = "#5F6B78",
  joint3 = "#E69F00",
  globalft50 = "#009E73",
  random = "#56B4E9",
  kmeans = "#0072B2",
  spectral = "#D55E00"
)
method_shapes <- c(
  baseline = 18,
  joint3 = 23,
  globalft50 = 22,
  random = 24,
  kmeans = 21,
  spectral = 8
)

make_plot <- function(horizon, title, y_limits, y_breaks, label_offset, stem) {
  plot_df <- summary_df[summary_df$horizon == horizon, , drop = FALSE]
  plot_df$label_y <- ifelse(
    is.na(plot_df$sd_percent),
    plot_df$mean_percent + label_offset,
    plot_df$upper + label_offset
  )
  plot_df$value_label <- ifelse(
    is.na(plot_df$sd_percent),
    sprintf("%.1f%%", plot_df$mean_percent),
    sprintf("%.1f \u00B1 %.1f%%", plot_df$mean_percent, plot_df$sd_percent)
  )
  baseline_value <- plot_df$mean_percent[plot_df$method_id == "baseline"]
  error_df <- plot_df[!is.na(plot_df$sd_percent), , drop = FALSE]

  p <- ggplot(
    plot_df,
    aes(
      x = x_position,
      y = mean_percent,
      color = method_id,
      fill = method_id,
      shape = method_id
    )
  ) +
    geom_hline(
      yintercept = baseline_value,
      color = method_colors[["baseline"]],
      linewidth = 0.75,
      linetype = "22"
    ) +
    geom_errorbar(
      data = error_df,
      aes(ymin = lower, ymax = upper),
      width = 0.16,
      linewidth = 1.05,
      show.legend = FALSE
    ) +
    geom_point(size = 4.7, stroke = 1.05) +
    geom_text(
      aes(y = label_y, label = value_label),
      fontface = "bold",
      size = 4.15,
      family = "sans",
      show.legend = FALSE
    ) +
    scale_color_manual(
      name = "Method",
      values = method_colors,
      breaks = method_order,
      labels = gsub("\n", " ", label_order)
    ) +
    scale_fill_manual(
      name = "Method",
      values = method_colors,
      breaks = method_order,
      labels = gsub("\n", " ", label_order)
    ) +
    scale_shape_manual(
      name = "Method",
      values = method_shapes,
      breaks = method_order,
      labels = gsub("\n", " ", label_order)
    ) +
    scale_x_continuous(
      breaks = seq_along(label_order),
      labels = label_order,
      expand = expansion(add = 0.34)
    ) +
    scale_y_continuous(
      limits = y_limits,
      breaks = y_breaks,
      labels = function(x) paste0(x, "%"),
      expand = expansion(mult = c(0, 0))
    ) +
    labs(
      title = title,
      subtitle = paste0(
        "Mean \u00B1 SD across three fine-tuning seeds; ",
        "baseline has no error bar"
      ),
      x = "Post-training method",
      y = "Task success rate"
    ) +
    guides(
      color = guide_legend(
        title.position = "left",
        nrow = 1,
        byrow = TRUE,
        override.aes = list(
          shape = unname(method_shapes[method_order]),
          fill = unname(method_colors[method_order]),
          size = 4.2
        )
      ),
      fill = "none",
      shape = "none"
    ) +
    theme_bw(base_size = 12, base_family = "sans") +
    theme(
      plot.title = element_text(size = 20, face = "bold", hjust = 0),
      plot.subtitle = element_text(
        size = 12.2,
        color = "#485464",
        margin = margin(b = 13)
      ),
      axis.title.x = element_text(size = 14.2, margin = margin(t = 14)),
      axis.title.y = element_text(size = 14.2, margin = margin(r = 10)),
      axis.text.x = element_text(
        size = 11.2,
        lineheight = 0.95,
        color = "#1F2933"
      ),
      axis.text.y = element_text(size = 11.5, color = "#1F2933"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "#D8DEE8", linewidth = 0.65),
      panel.border = element_rect(color = "#687386", linewidth = 0.75),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.title = element_text(face = "bold", size = 12.5),
      legend.text = element_text(size = 10.5),
      legend.spacing.x = unit(0.12, "cm"),
      legend.margin = margin(t = 9, b = 0),
      plot.margin = margin(t = 30, r = 20, b = 16, l = 18)
    )

  ggsave(
    file.path(out_dir, paste0(stem, ".png")),
    plot = p,
    width = 13.4,
    height = 7.35,
    units = "in",
    dpi = 300,
    bg = "white"
  )
  ggsave(
    file.path(out_dir, paste0(stem, ".pdf")),
    plot = p,
    width = 13.4,
    height = 7.35,
    units = "in",
    device = grDevices::cairo_pdf,
    bg = "white"
  )
}

make_plot(
  horizon = "Short horizon",
  title = "PushT: short-horizon task success rate",
  y_limits = c(82.0, 96.5),
  y_breaks = seq(84, 96, by = 4),
  label_offset = 0.48,
  stem = "pusht_short_horizon_main"
)

make_plot(
  horizon = "Long horizon",
  title = "PushT: long-horizon task success rate",
  y_limits = c(34.0, 43.5),
  y_breaks = seq(35, 43, by = 2),
  label_offset = 0.34,
  stem = "pusht_long_horizon_main"
)

capture.output(sessionInfo(), file = file.path(out_dir, "R_sessionInfo.txt"))
message("Wrote PushT short- and long-horizon figures to: ", out_dir)
