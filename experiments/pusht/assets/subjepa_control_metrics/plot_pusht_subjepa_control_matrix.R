#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  index <- match(flag, args)
  if (!is.na(index) && index < length(args)) return(args[[index + 1L]])
  default
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) stop("Run this script with Rscript.")
script_path <- normalizePath(sub("^--file=", "", script_arg), winslash = "/", mustWork = TRUE)
script_dir <- dirname(script_path)

input_csv <- normalizePath(
  get_arg("--input", file.path(script_dir, "pusht_subjepa_control_summary.csv")),
  winslash = "/",
  mustWork = TRUE
)
output_dir <- normalizePath(
  get_arg("--output-dir", script_dir),
  winslash = "/",
  mustWork = FALSE
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

results <- read.csv(
  input_csv,
  check.names = FALSE,
  stringsAsFactors = FALSE,
  na.strings = c("NA", "")
)

required_columns <- c(
  "horizon", "method_id", "method_label", "mean_percent", "sd_percent",
  "num_finetuning_seeds", "num_partition_seeds", "num_eval_seeds",
  "seed_type", "source_scope"
)
missing_columns <- setdiff(required_columns, names(results))
if (length(missing_columns) > 0L) {
  stop("Missing result columns: ", paste(missing_columns, collapse = ", "))
}

method_order <- c("baseline", "globalft50", "kmeans", "spectral", "autolap")
label_order <- c(
  "Official\nSub-JEPA",
  "Global-FT\n50ep",
  "K-means++\nK3-50",
  "Spectral\nK3-50",
  "Auto-LAP"
)

if (!setequal(unique(results$horizon), c("short", "long"))) {
  stop("Expected exactly short and long horizons.")
}
for (horizon in c("short", "long")) {
  observed <- results$method_id[results$horizon == horizon]
  if (!identical(observed, method_order)) {
    stop(horizon, " method order/content differs from the locked five-method matrix.")
  }
}

for (horizon in c("short", "long")) {
  global_row <- results[results$horizon == horizon & results$method_id == "globalft50", ]
  auto_row <- results[results$horizon == horizon & results$method_id == "autolap", ]
  if (
    nrow(global_row) != 1L || nrow(auto_row) != 1L ||
    !isTRUE(all.equal(global_row$mean_percent, auto_row$mean_percent)) ||
    !isTRUE(all.equal(global_row$sd_percent, auto_row$sd_percent))
  ) {
    stop("Auto-LAP statistics must exactly match Global-FT for ", horizon, ".")
  }
}

method_colors <- c(
  baseline = "#5F6B78",
  globalft50 = "#009E73",
  kmeans = "#0072B2",
  spectral = "#D55E00",
  autolap = "#6A3D9A"
)
method_shapes <- c(
  baseline = 18,
  globalft50 = 22,
  kmeans = 21,
  spectral = 8,
  autolap = 15
)

make_plot <- function(horizon) {
  plot_data <- results[results$horizon == horizon, , drop = FALSE]
  plot_data$method_id <- factor(plot_data$method_id, levels = method_order)
  plot_data$x_position <- seq_len(nrow(plot_data))
  plot_data$lower <- plot_data$mean_percent - plot_data$sd_percent
  plot_data$upper <- plot_data$mean_percent + plot_data$sd_percent
  plot_data$value_label <- ifelse(
    is.na(plot_data$sd_percent),
    sprintf("%.1f%%", plot_data$mean_percent),
    sprintf("%.1f ± %.1f%%", plot_data$mean_percent, plot_data$sd_percent)
  )

  label_offset <- if (horizon == "short") 0.18 else 0.45
  plot_data$label_y <- ifelse(
    is.na(plot_data$upper),
    plot_data$mean_percent + label_offset,
    plot_data$upper + label_offset
  )
  error_data <- plot_data[!is.na(plot_data$sd_percent), , drop = FALSE]
  baseline_value <- plot_data$mean_percent[plot_data$method_id == "baseline"]

  y_limits <- if (horizon == "short") c(91.75, 96.25) else c(37.2, 48.4)
  y_breaks <- if (horizon == "short") seq(92, 96, by = 1) else seq(38, 48, by = 2)

  p <- ggplot(
    plot_data,
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
      data = error_data,
      aes(ymin = lower, ymax = upper),
      width = 0.14,
      linewidth = 1.05,
      show.legend = FALSE
    ) +
    geom_point(size = 4.8, stroke = 1.05) +
    geom_text(
      aes(y = label_y, label = value_label),
      fontface = "bold",
      size = 4.2,
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
      expand = expansion(add = 0.28)
    ) +
    scale_y_continuous(
      limits = y_limits,
      breaks = y_breaks,
      labels = function(x) paste0(format(x, trim = TRUE), "%"),
      expand = expansion(mult = c(0, 0))
    ) +
    labs(
      title = paste0("PushT Sub-JEPA: ", horizon, "-horizon task success rate"),
      subtitle = "Mean ± SD across three fine-tuning seeds; baseline has no error bar",
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
      legend.spacing.x = grid::unit(0.12, "cm"),
      legend.margin = margin(t = 9, b = 0),
      plot.margin = margin(t = 28, r = 20, b = 16, l = 18)
    )

  stem <- paste0("pusht_subjepa_", horizon, "_horizon_main")
  ggsave(
    file.path(output_dir, paste0(stem, ".png")),
    plot = p,
    width = 13.4,
    height = 7.8,
    units = "in",
    dpi = 320,
    bg = "white"
  )
  pdf_device <- if (capabilities("cairo")) grDevices::cairo_pdf else "pdf"
  ggsave(
    file.path(output_dir, paste0(stem, ".pdf")),
    plot = p,
    width = 13.4,
    height = 7.8,
    units = "in",
    device = pdf_device,
    bg = "white"
  )
}

make_plot("short")
make_plot("long")

writeLines(
  c(
    capture.output(sessionInfo()),
    "",
    paste0("input_csv = ", input_csv),
    "primary_error_bar = sample SD across fine-tuning seeds after averaging partition and evaluation seeds",
    "figure_gate_annotation = none"
  ),
  file.path(output_dir, "R_sessionInfo.txt"),
  useBytes = TRUE
)

message("Wrote PushT Sub-JEPA short- and long-horizon PNG/PDF figures to: ", output_dir)
