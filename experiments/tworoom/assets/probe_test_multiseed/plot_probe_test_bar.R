#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(ggplot2))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg) == 1) {
  normalizePath(sub("^--file=", "", script_arg), winslash = "/")
} else {
  normalizePath("plot_probe_test_bar.R", winslash = "/")
}
script_dir <- dirname(script_path)
args <- commandArgs(trailingOnly = TRUE)
rooms3_csv <- if (length(args) >= 1) args[[1]] else file.path(script_dir, "data", "rooms3_multiseed_summary.csv")
priority5_csv <- if (length(args) >= 2) args[[2]] else file.path(script_dir, "data", "priority5_multiseed_summary.csv")
output_dir <- if (length(args) >= 3) args[[3]] else script_dir
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

read_partition <- function(path, partition) {
  data <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c("model", "metric", "mean", "std")
  missing <- setdiff(required, names(data))
  if (length(missing) > 0) {
    stop(paste("Missing columns in", path, ":", paste(missing, collapse = ", ")))
  }
  data <- data[data$model %in% c("linear_softmax_probe", "rff_rbf_probe"), ]
  data$partition <- partition
  data
}

df <- rbind(
  read_partition(rooms3_csv, "3-Partition"),
  read_partition(priority5_csv, "5-Partition")
)

probe_labels <- c(
  "linear_softmax_probe" = "Linear",
  "rff_rbf_probe" = "Non-linear"
)
df$probe <- factor(probe_labels[df$model], levels = c("Linear", "Non-linear"))
df$label_offset <- ifelse(df$probe == "Linear", 0.012, 0.040)

metric_labels <- c(
  "accuracy" = "Accuracy",
  "balanced_accuracy" = "Balanced\naccuracy",
  "macro_f1" = "Macro-F1",
  "left_room_f1" = "Left room\nF1",
  "doorway_corridor_f1" = "Doorway\nF1",
  "right_room_f1" = "Right room\nF1",
  "near_wall_f1" = "Near wall\nF1",
  "common_f1" = "Common\nF1"
)

rooms3_order <- c(
  "accuracy", "balanced_accuracy", "macro_f1",
  "left_room_f1", "doorway_corridor_f1", "right_room_f1"
)
priority5_order <- c(
  "accuracy", "balanced_accuracy", "macro_f1",
  "doorway_corridor_f1", "near_wall_f1", "common_f1",
  "right_room_f1", "left_room_f1"
)

metric_levels <- c(
  paste("3-Partition", rooms3_order, sep = "::"),
  paste("5-Partition", priority5_order, sep = "::")
)
df$metric_key <- factor(
  paste(df$partition, df$metric, sep = "::"),
  levels = metric_levels
)
axis_labels <- setNames(
  metric_labels[sub("^[^:]+::", "", metric_levels)],
  metric_levels
)

probe_colors <- c("Linear" = "#0072B2", "Non-linear" = "#D55E00")
dodge <- position_dodge(width = 0.78)

p <- ggplot(
  df,
  aes(x = metric_key, y = mean, fill = probe, group = probe)
) +
  geom_col(
    position = dodge,
    width = 0.70,
    color = "white",
    linewidth = 0.35
  ) +
  geom_errorbar(
    aes(ymin = pmax(0, mean - std), ymax = mean + std),
    position = dodge,
    width = 0.18,
    linewidth = 0.75,
    color = "#222222"
  ) +
  geom_text(
    aes(
      y = mean + std + label_offset,
      label = sprintf("%.2f%%", 100 * mean),
      color = probe
    ),
    position = dodge,
    size = 2.9,
    vjust = 0
  ) +
  facet_grid(
    cols = vars(partition),
    scales = "free_x",
    space = "free_x"
  ) +
  scale_x_discrete(labels = axis_labels, drop = TRUE) +
  scale_y_continuous(
    limits = c(0, 1.09),
    breaks = seq(0, 1.00, by = 0.20),
    labels = function(x) sprintf("%.0f%%", 100 * x),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_fill_manual(values = probe_colors) +
  scale_color_manual(values = probe_colors, guide = "none") +
  labs(
    title = "Probe Test",
    subtitle = expression("Test mean" %+-% "SD across five seeds"),
    x = "Metric",
    y = "Metric value",
    fill = "Probe"
  ) +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(size = 21, face = "bold", hjust = 0),
    plot.subtitle = element_text(size = 12.5, color = "#444444", margin = margin(b = 10)),
    axis.title = element_text(size = 14),
    axis.text.x = element_text(size = 10.5, lineheight = 0.95, color = "#222222"),
    axis.text.y = element_text(size = 11.5, color = "#222222"),
    strip.background = element_rect(fill = "#F2F2F2", color = "#B8B8B8"),
    strip.text = element_text(size = 13.5, face = "bold"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "#D9D9D9", linewidth = 0.45),
    legend.position = "bottom",
    legend.title = element_text(size = 12, face = "bold"),
    legend.text = element_text(size = 12),
    legend.key.width = grid::unit(18, "pt"),
    panel.spacing.x = grid::unit(12, "pt"),
    plot.margin = margin(12, 16, 10, 12)
  )

png_path <- file.path(output_dir, "probe_test_multiseed_bar.png")
pdf_path <- file.path(output_dir, "probe_test_multiseed_bar.pdf")

ggsave(
  png_path,
  plot = p,
  width = 13.2,
  height = 6.8,
  units = "in",
  dpi = 320,
  bg = "white"
)
ggsave(
  pdf_path,
  plot = p,
  width = 13.2,
  height = 6.8,
  units = "in",
  device = cairo_pdf,
  bg = "white"
)

capture.output(sessionInfo(), file = file.path(output_dir, "ggplot2_bar_session_info.txt"))
message("Wrote: ", normalizePath(png_path, winslash = "/", mustWork = FALSE))
message("Wrote: ", normalizePath(pdf_path, winslash = "/", mustWork = FALSE))
