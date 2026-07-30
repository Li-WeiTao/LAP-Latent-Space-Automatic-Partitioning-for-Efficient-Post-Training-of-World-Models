#!/usr/bin/env Rscript

# Publication figures for the seed-42 geometry trajectory-deviation experiment.
# Reads the five per-region trajectory_deviation.csv files produced by
# run_geometry_train_trajectory_deviation_exp5.sh.

suppressPackageStartupMessages({
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (!is.na(idx) && idx < length(args)) return(args[[idx + 1L]])
  default
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg) > 0L) sub("^--file=", "", script_arg[[1L]]) else NA_character_
script_dir <- if (!is.na(script_file)) dirname(normalizePath(script_file, mustWork = FALSE)) else getwd()

result_candidates <- c(
  file.path(dirname(script_dir), "results"),
  file.path(dirname(dirname(script_dir)), "results")
)
existing_results <- result_candidates[dir.exists(result_candidates)]
default_input_dir <- if (length(existing_results) > 0L) existing_results[[1L]] else script_dir
input_dir <- normalizePath(get_arg("--input-dir", default_input_dir), mustWork = TRUE)
output_dir <- normalizePath(get_arg("--output-dir", script_dir), mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

regions <- c("common", "doorway_corridor", "left_room", "near_wall", "right_room")

resolve_region_csv <- function(region) {
  candidates <- c(
    file.path(input_dir, paste0(region, ".csv")),
    file.path(
      input_dir,
      paste0("tworoom_geometry_train_trajectory_deviation_", region, "_vs_train_exp5"),
      "trajectory_deviation.csv"
    )
  )
  existing <- candidates[file.exists(candidates)]
  if (length(existing) == 0L) {
    stop("Missing trajectory-deviation CSV for region: ", region)
  }
  normalizePath(existing[[1L]], mustWork = TRUE)
}

source_paths <- setNames(vapply(regions, resolve_region_csv, character(1)), regions)
source_tables <- lapply(source_paths, read.csv, check.names = FALSE)

required_columns <- c(
  "step", "pairwise_mse_mean", "predictor_a_mse_vs_gt_mean",
  "predictor_b_mse_vs_gt_mean"
)
for (region in regions) {
  missing_columns <- setdiff(required_columns, names(source_tables[[region]]))
  if (length(missing_columns) > 0L) {
    stop(region, " is missing columns: ", paste(missing_columns, collapse = ", "))
  }
}

pairwise_long <- do.call(
  rbind,
  lapply(regions, function(region) {
    tab <- source_tables[[region]]
    data.frame(
      metric = "pairwise_vs_global",
      step = tab$step,
      predictor = paste0("P_train_", region),
      mse = tab$pairwise_mse_mean,
      predictor_label = region,
      stringsAsFactors = FALSE
    )
  })
)

gt_region_long <- do.call(
  rbind,
  lapply(regions, function(region) {
    tab <- source_tables[[region]]
    data.frame(
      metric = "gt_rollout",
      step = tab$step,
      predictor = paste0("P_train_", region),
      mse = tab$predictor_a_mse_vs_gt_mean,
      predictor_label = region,
      stringsAsFactors = FALSE
    )
  })
)

# The global predictor is identical in every source CSV; use one copy.
global_tab <- source_tables[[regions[[1L]]]]
gt_global_long <- data.frame(
  metric = "gt_rollout",
  step = global_tab$step,
  predictor = "P_train_global",
  mse = global_tab$predictor_b_mse_vs_gt_mean,
  predictor_label = "global",
  stringsAsFactors = FALSE
)
gt_long <- rbind(gt_region_long, gt_global_long)

pairwise_levels <- regions
gt_levels <- c(regions, "global")
pairwise_long$predictor_label <- factor(pairwise_long$predictor_label, levels = pairwise_levels)
gt_long$predictor_label <- factor(gt_long$predictor_label, levels = gt_levels)

combined_data <- rbind(
  transform(pairwise_long, predictor_label = as.character(predictor_label)),
  transform(gt_long, predictor_label = as.character(predictor_label))
)
write.csv(
  combined_data,
  file.path(output_dir, "real_gauge_drift_metrics_long.csv"),
  row.names = FALSE
)

palette <- c(
  common = "#0072B2",
  doorway_corridor = "#009E73",
  left_room = "#D55E00",
  near_wall = "#CC79A7",
  right_room = "#E69F00",
  global = "#000000"
)

linetypes <- c(
  common = "solid",
  doorway_corridor = "longdash",
  left_room = "dotdash",
  near_wall = "twodash",
  right_room = "dotted",
  global = "solid"
)

shapes <- c(
  common = 16,
  doorway_corridor = 17,
  left_room = 15,
  near_wall = 18,
  right_room = 3,
  global = 4
)

base_theme <- theme_bw(base_size = 9) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(size = 8),
    legend.text = element_text(size = 8),
    legend.key.width = unit(10, "mm"),
    plot.margin = margin(5, 6, 5, 5)
  )

plot_curve <- function(data, plot_title) {
  ggplot(
    data,
    aes(
      x = step,
      y = mse,
      colour = predictor_label,
      linetype = predictor_label,
      shape = predictor_label,
      group = predictor_label
    )
  ) +
    geom_line(linewidth = 0.55) +
    geom_point(size = 1.7, stroke = 0.45) +
    scale_x_continuous(breaks = 1:10) +
    scale_y_continuous(expand = expansion(mult = c(0.02, 0.06))) +
    scale_colour_manual(values = palette, drop = FALSE) +
    scale_linetype_manual(values = linetypes, drop = FALSE) +
    scale_shape_manual(values = shapes, drop = FALSE) +
    guides(
      colour = guide_legend(nrow = 2, byrow = TRUE),
      linetype = guide_legend(nrow = 2, byrow = TRUE),
      shape = guide_legend(nrow = 2, byrow = TRUE)
    ) +
    labs(
      title = plot_title,
      x = "Rollout step",
      y = "MSE",
      colour = "Predictor",
      linetype = "Predictor",
      shape = "Predictor"
    ) +
    base_theme
}

pairwise_plot <- plot_curve(pairwise_long, "Region predictor vs global predictor")
gt_plot <- plot_curve(gt_long, "Inference error")

export_plot <- function(plot, stem) {
  pdf_path <- file.path(output_dir, paste0(stem, ".pdf"))
  png_path <- file.path(output_dir, paste0(stem, ".png"))
  pdf_device <- if (capabilities("cairo")) grDevices::cairo_pdf else "pdf"

  ggsave(pdf_path, plot, width = 170, height = 95, units = "mm", device = pdf_device)
  ggsave(png_path, plot, width = 170, height = 95, units = "mm", dpi = 320, bg = "white")
  c(pdf = pdf_path, png = png_path)
}

outputs <- c(
  export_plot(pairwise_plot, "pairwise_vs_global_rollout_latent_mse"),
  export_plot(gt_plot, "gt_latent_rollout_mse")
)

session_path <- file.path(output_dir, "plot_real_gauge_drift_metrics_session_info.txt")
writeLines(
  c(
    capture.output(sessionInfo()),
    "",
    "Input files:",
    paste(names(source_paths), source_paths, sep = " = "),
    "",
    "Generated files:",
    outputs
  ),
  session_path
)

message("Wrote figures to: ", output_dir)
message(paste(outputs, collapse = "\n"))
