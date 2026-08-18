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

write_csv_lf <- function(data, path) {
  connection <- file(path, open = "wb")
  on.exit(close(connection), add = TRUE)
  write.csv(data, connection, row.names = FALSE)
}

task_name <- required_arg("--task")
model_name <- get_arg("--model-name", "LeWM")
short_input <- normalizePath(
  required_arg("--short-input"), winslash = "/", mustWork = TRUE
)
long_input <- normalizePath(
  required_arg("--long-input"), winslash = "/", mustWork = TRUE
)
output_dir <- required_arg("--output-dir")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_dir <- normalizePath(output_dir, winslash = "/", mustWork = TRUE)
file_prefix <- required_arg("--file-prefix")
summary_output <- get_arg(
  "--summary-output",
  paste0(file_prefix, "_control_method_summary.csv")
)
auto_source <- get_arg("--auto-source", "globalft50")
deployment_seed <- suppressWarnings(as.integer(get_arg("--deployment-seed", "0")))

if (basename(summary_output) != summary_output) {
  stop("--summary-output must be a filename, not a path")
}

method_order <- c(
  "baseline", "joint3", "globalft50", "random", "kmeans", "spectral",
  "autolap"
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
  "Global-FT\n50ep",
  "Random-Voronoi\nK3-50",
  "K-means++\nK3-50",
  "Spectral\nK3-50",
  "Auto-LAP"
)
label_by_id <- setNames(label_order, method_order)

if (!auto_source %in% names(source_names)[-1L]) {
  stop("Unsupported --auto-source: ", auto_source)
}
if (is.na(deployment_seed) || deployment_seed < 0L) {
  stop("--deployment-seed must be a non-negative integer")
}

read_summary <- function(horizon, path) {
  raw <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  required_columns <- c(
    "method", "mean_percent", "sd_across_finetuning_seeds_percent",
    "num_finetuning_seeds", "num_partition_seeds", "num_eval_seeds"
  )
  missing_columns <- setdiff(required_columns, names(raw))
  if (length(missing_columns) > 0L) {
    stop("Missing columns in ", path, ": ", paste(missing_columns, collapse = ", "))
  }

  base_rows <- lapply(method_order[method_order != "autolap"], function(method_id) {
    source_name <- unname(source_names[[method_id]])
    hit <- raw[raw$method == source_name, , drop = FALSE]
    if (nrow(hit) != 1L) {
      stop("Expected exactly one row for ", source_name, " in ", path)
    }
    data.frame(
      horizon = horizon,
      method_id = method_id,
      method_label = unname(label_by_id[[method_id]]),
      mean_percent = hit$mean_percent,
      sd_percent = hit$sd_across_finetuning_seeds_percent,
      num_finetuning_seeds = hit$num_finetuning_seeds,
      num_partition_seeds = hit$num_partition_seeds,
      num_eval_seeds = hit$num_eval_seeds,
      source_method = source_name,
      stringsAsFactors = FALSE
    )
  })
  result <- do.call(rbind, base_rows)
  source_row <- result[result$method_id == auto_source, , drop = FALSE]
  if (nrow(source_row) != 1L) stop("Auto-LAP source row is not unique")
  auto_row <- source_row
  auto_row$method_id <- "autolap"
  auto_row$method_label <- unname(label_by_id[["autolap"]])
  rbind(result, auto_row)
}

summary_df <- rbind(
  read_summary("short", short_input),
  read_summary("long", long_input)
)
summary_df$method_id <- factor(summary_df$method_id, levels = method_order)
summary_df$method_label <- factor(summary_df$method_label, levels = label_order)
summary_df$x_position <- match(as.character(summary_df$method_id), method_order)
summary_df$lower <- summary_df$mean_percent - summary_df$sd_percent
summary_df$upper <- summary_df$mean_percent + summary_df$sd_percent

write_csv_lf(
  summary_df,
  file.path(output_dir, summary_output)
)

method_colors <- c(
  baseline = "#5F6B78",
  joint3 = "#E69F00",
  globalft50 = "#009E73",
  random = "#56B4E9",
  kmeans = "#0072B2",
  spectral = "#D55E00",
  autolap = "#6A3D9A"
)
method_shapes <- c(
  baseline = 18,
  joint3 = 23,
  globalft50 = 22,
  random = 24,
  kmeans = 21,
  spectral = 8,
  autolap = 15
)

auto_caption <- switch(
  auto_source,
  globalft50 = "Auto-LAP = Global-FT (the gate-selected fallback).",
  spectral = paste0(
    "Auto-LAP = Spectral K3-50 with preset deployment seed ",
    deployment_seed,
    "."
  ),
  paste0("Auto-LAP = ", unname(source_names[[auto_source]]), ".")
)

make_plot <- function(horizon) {
  plot_df <- summary_df[summary_df$horizon == horizon, , drop = FALSE]
  effective_lower <- ifelse(
    is.na(plot_df$lower), plot_df$mean_percent, plot_df$lower
  )
  effective_upper <- ifelse(
    is.na(plot_df$upper), plot_df$mean_percent, plot_df$upper
  )
  data_span <- max(effective_upper) - min(effective_lower)
  data_span <- max(data_span, 1.0)
  label_offset <- max(0.24, 0.055 * data_span)
  plot_df$label_y <- effective_upper + label_offset
  plot_df$value_label <- ifelse(
    is.na(plot_df$sd_percent),
    sprintf("%.1f%%", plot_df$mean_percent),
    sprintf("%.1f \u00B1 %.1f%%", plot_df$mean_percent, plot_df$sd_percent)
  )
  y_min <- max(0, min(effective_lower) - max(0.45, 0.14 * data_span))
  y_max <- min(100, max(plot_df$label_y) + max(0.38, 0.10 * data_span))
  y_breaks <- pretty(c(y_min, y_max), n = 5)
  y_breaks <- y_breaks[y_breaks >= y_min & y_breaks <= y_max]
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
      limits = c(y_min, y_max),
      breaks = y_breaks,
      labels = function(x) paste0(format(x, trim = TRUE), "%"),
      expand = expansion(mult = c(0, 0))
    ) +
    labs(
      title = paste0(
        task_name,
        ": ",
        horizon,
        "-horizon task success rate"
      ),
      subtitle = paste0(
        "Mean \u00B1 SD across three fine-tuning seeds; ",
        "baseline has no error bar"
      ),
      caption = auto_caption,
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
      plot.caption = element_text(
        size = 11.2,
        color = "#485464",
        hjust = 0,
        margin = margin(t = 10)
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
      legend.spacing.x = grid::unit(0.12, "cm"),
      legend.margin = margin(t = 9, b = 0),
      plot.margin = margin(t = 30, r = 20, b = 16, l = 18)
    )

  stem <- paste0(file_prefix, "_", horizon, "_horizon_main")
  ggsave(
    file.path(output_dir, paste0(stem, ".png")),
    plot = p,
    width = 13.4,
    height = 7.8,
    units = "in",
    dpi = 300,
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
    paste0("task = ", task_name),
    paste0("model = ", model_name),
    paste0("short_input = ", short_input),
    paste0("long_input = ", long_input),
    paste0("auto_source = ", auto_source),
    paste0("summary_output = ", summary_output),
    paste0("deployment_seed = ", deployment_seed),
    paste0("figure_caption = ", auto_caption),
    paste0(
      "primary_error_bar = sample SD across fine-tuning seeds after averaging ",
      "partition and evaluation seeds"
    ),
    "figure_gate_check_values = none"
  ),
  file.path(output_dir, paste0(file_prefix, "_R_sessionInfo.txt")),
  useBytes = TRUE
)

message("Wrote ", task_name, " short- and long-horizon figures to: ", output_dir)
