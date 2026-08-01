#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(ggplot2))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg) == 1) {
  normalizePath(sub("^--file=", "", script_arg), winslash = "/")
} else {
  normalizePath("plot_tworoom_main.R", winslash = "/")
}
args <- commandArgs(trailingOnly = TRUE)
horizon <- if (length(args) >= 1) args[[1]] else "long"
if (!horizon %in% c("short", "long")) {
  stop("Usage: plot_tworoom_main.R [short|long]")
}
assets_root <- dirname(dirname(script_path))
out_dir <- file.path(assets_root, paste0(horizon, "_horizon_metrics"))
stem <- paste0("tworoom_", horizon, "_horizon")
input_csv <- file.path(out_dir, paste0(stem, "_method_seeds.csv"))

raw <- read.csv(input_csv, check.names = FALSE, stringsAsFactors = FALSE)

method_order <- c(
  "baseline", "joint3", "globalft50", "random",
  "kmeans", "spectral", "rooms3"
)
label_order <- c(
  "Official\nbaseline",
  "Joint-Continue\n3ep",
  "Global-FT\n50ep",
  "Random-Voronoi\nK3-50",
  "K-means++\nK3-50",
  "Spectral\nK3-50",
  "Human partition\nrooms3-50"
)
label_by_id <- setNames(label_order, method_order)

summary_rows <- lapply(method_order, function(method) {
  rows <- raw[raw$method_id == method, , drop = FALSE]
  values <- rows$value_percent
  data.frame(
    method_id = method,
    method_label = unname(label_by_id[[method]]),
    seed_type = rows$seed_type[[1]],
    n_method_seeds = length(values),
    mean_percent = mean(values),
    sd_percent = if (length(values) > 1) sd(values) else NA_real_,
    stringsAsFactors = FALSE
  )
})
summary_df <- do.call(rbind, summary_rows)
summary_df$method_id <- factor(summary_df$method_id, levels = method_order)
summary_df$method_label <- factor(summary_df$method_label, levels = label_order)
summary_df$x_position <- match(as.character(summary_df$method_id), method_order)
summary_df$lower <- summary_df$mean_percent - summary_df$sd_percent
summary_df$upper <- summary_df$mean_percent + summary_df$sd_percent
label_offset <- if (horizon == "short") 0.34 else 0.72
summary_df$label_y <- ifelse(
  is.na(summary_df$sd_percent),
  summary_df$mean_percent + label_offset,
  summary_df$upper + label_offset
)
summary_df$value_label <- ifelse(
  is.na(summary_df$sd_percent),
  sprintf("%.1f%%", summary_df$mean_percent),
  sprintf("%.1f ± %.1f%%", summary_df$mean_percent, summary_df$sd_percent)
)
summary_df$value_label <- ifelse(
  is.na(summary_df$sd_percent),
  sprintf("%.1f%%", summary_df$mean_percent),
  sprintf("%.1f \u00B1 %.1f%%", summary_df$mean_percent, summary_df$sd_percent)
)

write.csv(
  summary_df,
  file.path(out_dir, paste0(stem, "_method_summary.csv")),
  row.names = FALSE
)

method_colors <- c(
  baseline = "#5F6B78",
  joint3 = "#E69F00",
  globalft50 = "#009E73",
  random = "#56B4E9",
  kmeans = "#0072B2",
  spectral = "#D55E00",
  rooms3 = "#CC79A7"
)
method_shapes <- c(
  baseline = 18,
  joint3 = 23,
  globalft50 = 22,
  random = 24,
  kmeans = 21,
  spectral = 8,
  rooms3 = 25
)

baseline_value <- summary_df$mean_percent[summary_df$method_id == "baseline"]
error_df <- summary_df[!is.na(summary_df$sd_percent), , drop = FALSE]

p <- ggplot(
  summary_df,
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
    limits = if (horizon == "short") c(88.4, 94.2) else c(43.2, 66.4),
    breaks = if (horizon == "short") seq(89, 94, by = 1) else seq(44, 64, by = 4),
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    title = paste0("TwoRoom: ", horizon, "-horizon task success rate"),
    subtitle = paste0(
      "Mean ± SD across three method seeds; Random uses partition seeds; ",
      "baseline has no error bar"
    ),
    x = "Post-training method",
    y = "Task success rate"
  ) +
  guides(
    color = guide_legend(
      title.position = "left",
      nrow = 2,
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
    plot.subtitle = element_text(size = 12.2, color = "#485464", margin = margin(b = 13)),
    axis.title.x = element_text(size = 14.2, margin = margin(t = 14)),
    axis.title.y = element_text(size = 14.2, margin = margin(r = 10)),
    axis.text.x = element_text(size = 11.2, lineheight = 0.95, color = "#1F2933"),
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

p <- p + labs(
  subtitle = paste0(
    "Mean \u00B1 SD across three fine-tuning seeds; ",
    "baseline has no error bar"
  )
)

png_path <- file.path(out_dir, paste0(stem, "_main.png"))
pdf_path <- file.path(out_dir, paste0(stem, "_main.pdf"))

ggsave(
  png_path,
  plot = p,
  width = 13.4,
  height = 7.35,
  units = "in",
  dpi = 300,
  bg = "white"
)
ggsave(
  pdf_path,
  plot = p,
  width = 13.4,
  height = 7.35,
  units = "in",
  device = grDevices::cairo_pdf,
  bg = "white"
)

capture.output(sessionInfo(), file = file.path(out_dir, "R_sessionInfo.txt"))
message("Wrote: ", png_path)
message("Wrote: ", pdf_path)
