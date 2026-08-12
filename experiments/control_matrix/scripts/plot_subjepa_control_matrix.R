#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  index <- match(flag, args)
  if (!is.na(index) && index < length(args)) return(args[[index + 1L]])
  default
}

required_arg <- function(flag) {
  value <- get_arg(flag)
  if (is.null(value) || !nzchar(value)) stop("Missing required argument: ", flag)
  value
}

task_name <- required_arg("--task")
input_csv <- normalizePath(required_arg("--input"), winslash = "/", mustWork = TRUE)
output_dir <- get_arg("--output-dir", dirname(input_csv))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_dir <- normalizePath(output_dir, winslash = "/", mustWork = TRUE)
file_prefix <- required_arg("--file-prefix")
auto_source <- get_arg("--auto-source", "globalft50")

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
if (!auto_source %in% setdiff(method_order, c("baseline", "autolap"))) {
  stop("Unsupported --auto-source: ", auto_source)
}
for (horizon in c("short", "long")) {
  source_row <- results[
    results$horizon == horizon & results$method_id == auto_source,
    , drop = FALSE
  ]
  auto_row <- results[
    results$horizon == horizon & results$method_id == "autolap",
    , drop = FALSE
  ]
  if (
    nrow(source_row) != 1L || nrow(auto_row) != 1L ||
    !isTRUE(all.equal(source_row$mean_percent, auto_row$mean_percent)) ||
    !isTRUE(all.equal(source_row$sd_percent, auto_row$sd_percent))
  ) {
    stop("Auto-LAP statistics do not match --auto-source for ", horizon, ".")
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
    sprintf("%.1f \u00B1 %.1f%%", plot_data$mean_percent, plot_data$sd_percent)
  )

  lower_effective <- ifelse(is.na(plot_data$lower), plot_data$mean_percent, plot_data$lower)
  upper_effective <- ifelse(is.na(plot_data$upper), plot_data$mean_percent, plot_data$upper)
  data_span <- max(upper_effective) - min(lower_effective)
  data_span <- max(data_span, 1.0)
  label_offset <- max(0.24, 0.055 * data_span)
  plot_data$label_y <- upper_effective + label_offset
  y_min <- max(0, min(lower_effective) - max(0.45, 0.14 * data_span))
  y_max <- min(100, max(plot_data$label_y) + max(0.38, 0.10 * data_span))
  y_breaks <- pretty(c(y_min, y_max), n = 5)
  y_breaks <- y_breaks[y_breaks >= y_min & y_breaks <= y_max]

  error_data <- plot_data[!is.na(plot_data$sd_percent), , drop = FALSE]
  baseline_value <- plot_data$mean_percent[plot_data$method_id == "baseline"]

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
      limits = c(y_min, y_max),
      breaks = y_breaks,
      labels = function(x) paste0(format(x, trim = TRUE), "%"),
      expand = expansion(mult = c(0, 0))
    ) +
    labs(
      title = paste0(task_name, " Sub-JEPA: ", horizon, "-horizon task success rate"),
      subtitle = paste0(
        "Mean \u00B1 sample SD across three fine-tuning seeds after averaging ",
        "partition and evaluation seeds; official checkpoint has no error bar"
      ),
      caption = paste0(
        "Partitioned methods average three partition seeds; all methods use ",
        "five paired evaluation seeds."
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
      plot.subtitle = element_text(size = 11.7, color = "#485464", margin = margin(b = 13)),
      plot.caption = element_text(
        size = 10.8,
        color = "#485464",
        hjust = 0,
        lineheight = 1.12,
        margin = margin(t = 10)
      ),
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

  stem <- paste0(file_prefix, "_", horizon, "_horizon_main")
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
    paste0("task = ", task_name),
    paste0("auto_source = ", auto_source),
    "primary_error_bar = sample SD across fine-tuning seeds after averaging partition and evaluation seeds",
    "figure_gate_annotation = none"
  ),
  file.path(output_dir, paste0(file_prefix, "_R_sessionInfo.txt")),
  useBytes = TRUE
)

message("Wrote ", task_name, " Sub-JEPA short- and long-horizon PNG/PDF figures to: ", output_dir)
