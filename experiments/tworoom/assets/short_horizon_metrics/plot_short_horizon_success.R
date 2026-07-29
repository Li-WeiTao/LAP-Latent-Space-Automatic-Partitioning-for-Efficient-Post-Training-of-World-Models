#!/usr/bin/env Rscript

# Short-horizon TwoRoom success-rate figure.
# Fine-tuned models use sample SD across predictor fine-tuning seeds 0/42/625.
# Baseline has no fine-tuning seed and is shown only as a fixed reference.

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

aggregate_path <- file.path(
  input_dir,
  "tworoom_success_rate_geometry_global65_finetune_seed_summary.csv"
)
baseline_path <- file.path(input_dir, "tworoom_success_rate_5seed_summary.csv")
ft45_seeds <- c(0, 42, 625)
ft45_paths <- setNames(
  file.path(
    input_dir,
    sprintf(
      "tworoom_success_rate_global_ft_45ep_trainseed%s_short_5eval_summary.csv",
      ft45_seeds
    )
  ),
  ft45_seeds
)
if (!file.exists(aggregate_path)) stop("Missing input CSV: ", aggregate_path)
if (!file.exists(baseline_path)) stop("Missing input CSV: ", baseline_path)
missing_ft45 <- ft45_paths[!file.exists(ft45_paths)]
if (length(missing_ft45) > 0L) {
  stop("Missing Global-FT45 input CSV(s): ", paste(missing_ft45, collapse = ", "))
}

aggregate <- read.csv(aggregate_path, check.names = FALSE)
baseline <- read.csv(baseline_path, check.names = FALSE)
aggregate <- aggregate[aggregate$horizon == "short", , drop = FALSE]

method_row <- function(method, configuration, model) {
  row <- aggregate[aggregate$method == method, , drop = FALSE]
  if (nrow(row) != 1L) stop(method, " expected one aggregate row, found ", nrow(row))
  data.frame(
    configuration = configuration,
    model = model,
    mean = row$fine_tuning_seed_mean,
    sd = row$fine_tuning_seed_sample_std,
    train_seed_0_mean = row$train_seed_0_mean,
    train_seed_42_mean = row$train_seed_42_mean,
    train_seed_625_mean = row$train_seed_625_mean,
    uncertainty = "sample SD across fine-tuning seeds 0/42/625",
    stringsAsFactors = FALSE
  )
}

baseline_values <- baseline$success_rate[baseline$mode == "baseline"]
if (length(baseline_values) != 5L) stop("Baseline expected five eval-seed rows.")

ft45_seed_means <- vapply(ft45_paths, function(path) {
  rows <- read.csv(path, check.names = FALSE)
  if (nrow(rows) != 5L) stop(path, " expected five eval-seed rows.")
  if (!all(rows$num_eval == 50L)) stop(path, " expected num_eval=50 for every row.")
  mean(rows$success_rate)
}, numeric(1L))

results <- rbind(
  data.frame(
    configuration = "Baseline",
    model = "Baseline",
    mean = mean(baseline_values),
    sd = NA_real_,
    train_seed_0_mean = NA_real_,
    train_seed_42_mean = NA_real_,
    train_seed_625_mean = NA_real_,
    uncertainty = "fixed reference; no fine-tuning seed",
    stringsAsFactors = FALSE
  ),
  data.frame(
    configuration = "Global-FT\n45ep",
    model = "Global-FT45",
    mean = mean(ft45_seed_means),
    sd = sd(ft45_seed_means),
    train_seed_0_mean = unname(ft45_seed_means[["0"]]),
    train_seed_42_mean = unname(ft45_seed_means[["42"]]),
    train_seed_625_mean = unname(ft45_seed_means[["625"]]),
    uncertainty = "sample SD across fine-tuning seeds 0/42/625",
    stringsAsFactors = FALSE
  ),
  method_row("rooms3_30ep", "rooms3\n30ep", "rooms3"),
  method_row("priority5_30ep", "priority5\n30ep", "priority5"),
  method_row("rooms3_50ep", "rooms3\n50ep", "rooms3"),
  method_row("priority5_50ep", "priority5\n50ep", "priority5"),
  method_row("rooms3_doorway80", "rooms3\n80ep", "rooms3"),
  method_row("priority5_doorway80", "priority5\n80ep", "priority5")
)

results$configuration <- factor(
  results$configuration,
  levels = c(
    "Baseline", "Global-FT\n45ep", "rooms3\n30ep", "priority5\n30ep", "rooms3\n50ep",
    "priority5\n50ep", "rooms3\n80ep", "priority5\n80ep"
  )
)
results$lower <- ifelse(is.na(results$sd), results$mean, results$mean - results$sd)
results$upper <- ifelse(is.na(results$sd), results$mean, results$mean + results$sd)
results$value_label <- ifelse(
  is.na(results$sd),
  sprintf("%.1f%%", results$mean),
  sprintf("%.1f \u00b1 %.1f%%", results$mean, results$sd)
)

model_colors <- c(
  Baseline = "#5C6670", `Global-FT45` = "#009E73",
  rooms3 = "#0072B2", priority5 = "#D55E00"
)
model_shapes <- c(Baseline = 18, `Global-FT45` = 15, rooms3 = 16, priority5 = 17)

p <- ggplot(results, aes(x = configuration, y = mean, color = model, shape = model)) +
  geom_hline(
    yintercept = results$mean[results$model == "Baseline"],
    linewidth = 0.45,
    linetype = "dashed",
    color = model_colors[["Baseline"]]
  ) +
  geom_errorbar(
    data = results[results$model != "Baseline", , drop = FALSE],
    aes(ymin = lower, ymax = upper),
    width = 0.18,
    linewidth = 0.75,
    show.legend = FALSE
  ) +
  geom_point(size = 3.2, stroke = 0.8) +
  geom_text(
    aes(y = upper + 0.55, label = value_label),
    size = 3.0,
    fontface = "bold",
    show.legend = FALSE
  ) +
  scale_color_manual(
    values = model_colors,
    breaks = c("Baseline", "Global-FT45", "rooms3", "priority5")
  ) +
  scale_shape_manual(
    values = model_shapes,
    breaks = c("Baseline", "Global-FT45", "rooms3", "priority5")
  ) +
  scale_x_discrete(limits = levels(results$configuration)) +
  scale_y_continuous(
    limits = c(83.8, 98.2),
    breaks = seq(84, 98, by = 2),
    labels = function(x) sprintf("%d%%", x),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    title = "Short-horizon task success rate",
    subtitle = "Mean \u00b1 SD across three fine-tuning seeds; baseline shown without an error bar",
    x = "Model and fine-tuning round",
    y = "Task success rate",
    color = "Model",
    shape = "Model"
  ) +
  theme_bw(base_size = 10) +
  theme(
    plot.title = element_text(face = "bold", size = 12, hjust = 0),
    plot.subtitle = element_text(size = 8.5, color = "#4B5563", margin = margin(b = 8)),
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

png_path <- file.path(output_dir, "short_horizon_task_success_mean_std.png")
pdf_path <- file.path(output_dir, "short_horizon_task_success_mean_std.pdf")
csv_path <- file.path(output_dir, "short_horizon_task_success_summary.csv")
session_path <- file.path(output_dir, "plot_short_horizon_success_session_info.txt")

ggsave(png_path, p, width = 180, height = 105, units = "mm", dpi = 320, bg = "white")
pdf_device <- if (capabilities("cairo")) grDevices::cairo_pdf else "pdf"
ggsave(pdf_path, p, width = 180, height = 105, units = "mm", device = pdf_device, bg = "white")
write.csv(
  results[c(
    "configuration", "model", "mean", "sd", "train_seed_0_mean",
    "train_seed_42_mean", "train_seed_625_mean", "uncertainty"
  )],
  csv_path,
  row.names = FALSE
)
writeLines(
  c(
    capture.output(sessionInfo()),
    "",
    "Input files:",
    paste0("aggregate = ", normalizePath(aggregate_path)),
    paste0("baseline = ", normalizePath(baseline_path)),
    paste0("global_ft45 = ", paste(normalizePath(ft45_paths), collapse = ", "))
  ),
  session_path
)

message("Saved preview: ", png_path)
message("Saved vector figure: ", pdf_path)
