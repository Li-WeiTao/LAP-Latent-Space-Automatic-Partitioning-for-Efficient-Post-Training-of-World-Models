#!/usr/bin/env Rscript

# Long-horizon TwoRoom success-rate figure from raw five-seed result tables.

suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (!is.na(idx) && idx < length(args)) return(args[[idx + 1L]])
  default
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) stop("Run this figure script with Rscript.")
script_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
script_dir <- dirname(script_path)

result_candidates <- c(
  file.path(dirname(script_dir), "results"),
  file.path(dirname(dirname(script_dir)), "results")
)
existing_results <- result_candidates[dir.exists(result_candidates)]
default_input_dir <- if (length(existing_results) > 0L) existing_results[[1L]] else script_dir
input_dir <- normalizePath(get_arg("--input-dir", default_input_dir), mustWork = TRUE)
output_dir <- normalizePath(get_arg("--output-dir", script_dir), mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

resolve_input <- function(flat_name, result_name) {
  candidates <- c(file.path(input_dir, flat_name), file.path(input_dir, result_name))
  existing <- candidates[file.exists(candidates)]
  if (length(existing) == 0L) stop("Missing input CSV: ", result_name)
  normalizePath(existing[[1L]], mustWork = TRUE)
}

source_paths <- c(
  ep30_50 = resolve_input(
    "ep30_50.csv",
    "tworoom_success_rate_exp6_30ep_50ep_5seed_summary.csv"
  ),
  baseline_ep80 = resolve_input(
    "baseline_ep80.csv",
    "tworoom_success_rate_exp6_5seed_summary.csv"
  )
)
tables <- lapply(source_paths, read.csv, check.names = FALSE)

summarize_mode <- function(tab, mode, configuration, model) {
  values <- tab$success_rate[tab$mode == mode]
  if (length(values) != 5L) {
    stop(configuration, " expected 5 seed rows, found ", length(values))
  }
  data.frame(
    configuration = configuration,
    model = model,
    mean = mean(values),
    sd = sd(values),
    stringsAsFactors = FALSE
  )
}

results <- rbind(
  summarize_mode(tables$baseline_ep80, "baseline", "Baseline", "Baseline"),
  summarize_mode(tables$ep30_50, "rooms3_30ep", "rooms3\n30ep", "rooms3"),
  summarize_mode(tables$ep30_50, "priority5_30ep", "priority5\n30ep", "priority5"),
  summarize_mode(tables$ep30_50, "rooms3_50ep", "rooms3\n50ep", "rooms3"),
  summarize_mode(tables$ep30_50, "priority5_50ep", "priority5\n50ep", "priority5"),
  summarize_mode(tables$baseline_ep80, "rooms3", "rooms3\n80ep", "rooms3"),
  summarize_mode(tables$baseline_ep80, "priority5", "priority5\n80ep", "priority5")
)

results$configuration <- factor(results$configuration, levels = results$configuration)
results$lower <- results$mean - results$sd
results$upper <- results$mean + results$sd
results$value_label <- sprintf("%.1f \u00b1 %.1f%%", results$mean, results$sd)

model_colors <- c(
  Baseline = "#5C6670",
  rooms3 = "#0072B2",
  priority5 = "#D55E00"
)

model_shapes <- c(Baseline = 18, rooms3 = 16, priority5 = 17)

p <- ggplot(results, aes(x = configuration, y = mean, color = model, shape = model)) +
  geom_hline(
    yintercept = results$mean[results$model == "Baseline"],
    linewidth = 0.45,
    linetype = "dashed",
    color = model_colors[["Baseline"]]
  ) +
  geom_errorbar(
    aes(ymin = lower, ymax = upper),
    width = 0.18,
    linewidth = 0.75,
    show.legend = FALSE
  ) +
  geom_point(size = 3.2, stroke = 0.8) +
  geom_text(
    aes(y = upper + 0.35, label = value_label),
    size = 3.0,
    fontface = "bold",
    vjust = 0,
    show.legend = FALSE
  ) +
  scale_color_manual(values = model_colors, breaks = c("Baseline", "rooms3", "priority5")) +
  scale_shape_manual(values = model_shapes, breaks = c("Baseline", "rooms3", "priority5")) +
  scale_y_continuous(
    limits = c(43.2, 66.2),
    breaks = seq(44, 64, by = 4),
    labels = function(x) sprintf("%d%%", x),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    title = "Long-horizon task success rate",
    subtitle = "Mean \u00b1 std across five seeds",
    x = "Model and fine-tuning round",
    y = "Task success rate",
    color = "Model",
    shape = "Model"
  ) +
  theme_bw(base_size = 10) +
  theme(
    plot.title = element_text(face = "bold", size = 12, hjust = 0),
    plot.subtitle = element_text(size = 9, color = "#4B5563", margin = margin(b = 8)),
    axis.title.x = element_text(margin = margin(t = 8)),
    axis.title.y = element_text(margin = margin(r = 8)),
    axis.text.x = element_text(size = 8.5, lineheight = 0.95, color = "#20242A"),
    axis.text.y = element_text(color = "#20242A"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(linewidth = 0.35, color = "#D9DEE5"),
    panel.border = element_rect(linewidth = 0.55, color = "#6B7280"),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    legend.box.margin = margin(t = -2),
    plot.margin = margin(10, 14, 8, 10)
  )

png_path <- file.path(output_dir, "long_horizon_task_success_mean_std.png")
pdf_path <- file.path(output_dir, "long_horizon_task_success_mean_std.pdf")
csv_path <- file.path(output_dir, "long_horizon_task_success_summary.csv")
session_path <- file.path(output_dir, "plot_long_horizon_success_session_info.txt")

ggsave(png_path, p, width = 180, height = 105, units = "mm", dpi = 320, bg = "white")
pdf_device <- if (capabilities("cairo")) grDevices::cairo_pdf else "pdf"
ggsave(pdf_path, p, width = 180, height = 105, units = "mm", device = pdf_device, bg = "white")

write.csv(results[c("configuration", "model", "mean", "sd")], csv_path, row.names = FALSE)
writeLines(
  c(
    capture.output(sessionInfo()),
    "",
    "Input files:",
    paste(names(source_paths), source_paths, sep = " = ")
  ),
  session_path
)

message("Saved preview: ", png_path)
message("Saved vector figure: ", pdf_path)
