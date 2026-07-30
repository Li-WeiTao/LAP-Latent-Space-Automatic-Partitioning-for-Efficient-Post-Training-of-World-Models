suppressPackageStartupMessages(library(ggplot2))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1) {
  stop("Run this figure script with Rscript.")
}
script_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
output_dir <- file.path(dirname(script_path), "outputs")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

timing <- data.frame(
  mode = c("baseline", "rooms3", "priority5"),
  batch_mean_s = c(44.67, 45.80, 46.26),
  batch_std_s = c(0.71, 0.33, 0.37),
  overhead_pct = c(0.0, 2.5, 3.5),
  stringsAsFactors = FALSE
)

timing$mode <- factor(timing$mode, levels = timing$mode)
timing$latency_ms <- timing$batch_mean_s * 1000 / 50
timing$latency_std_ms <- timing$batch_std_s * 1000 / 50
timing$lower <- timing$latency_ms - timing$latency_std_ms
timing$upper <- timing$latency_ms + timing$latency_std_ms
timing$value_label <- sprintf("%.0f ms", timing$latency_ms)
timing$overhead_label <- ifelse(
  timing$mode == "baseline",
  "baseline",
  sprintf("+%.1f%%", timing$overhead_pct)
)

mode_colors <- c(
  "baseline" = "#5C6670",
  "rooms3" = "#0072B2",
  "priority5" = "#D55E00"
)

p <- ggplot(timing, aes(x = mode, y = latency_ms, fill = mode)) +
  geom_col(
    width = 0.58,
    color = "white",
    linewidth = 0.6,
    show.legend = FALSE
  ) +
  geom_hline(
    yintercept = timing$latency_ms[timing$mode == "baseline"],
    color = "#4B5563",
    linewidth = 0.5,
    linetype = "dashed"
  ) +
  geom_errorbar(
    aes(ymin = lower, ymax = upper),
    width = 0.18,
    linewidth = 0.7,
    color = "#252A31"
  ) +
  geom_text(
    aes(y = upper + 27, label = value_label, color = mode),
    size = 3.6,
    fontface = "bold",
    show.legend = FALSE
  ) +
  geom_text(
    aes(y = latency_ms - 78, label = overhead_label),
    size = 3.5,
    color = "white",
    fontface = "bold",
    show.legend = FALSE
  ) +
  scale_fill_manual(values = mode_colors) +
  scale_color_manual(values = mode_colors) +
  scale_y_continuous(
    limits = c(0, 1025),
    breaks = seq(0, 1000, by = 200),
    labels = function(x) sprintf("%d", x),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    title = "Inference speed",
    x = NULL,
    y = "Plan time (ms)"
  ) +
  theme_bw(base_size = 10) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    axis.title.y = element_text(margin = margin(r = 8)),
    axis.text.x = element_text(size = 10, face = "bold", color = "#20242A", margin = margin(t = 5)),
    axis.text.y = element_text(color = "#20242A"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(linewidth = 0.35, color = "#D9DEE5"),
    panel.border = element_rect(linewidth = 0.55, color = "#6B7280"),
    plot.margin = margin(10, 14, 8, 10)
  )

png_path <- file.path(output_dir, "single_env_replan_latency_comparison.png")
pdf_path <- file.path(output_dir, "single_env_replan_latency_comparison.pdf")
csv_path <- file.path(output_dir, "single_env_replan_latency_summary.csv")
session_path <- file.path(output_dir, "plot_single_replan_latency_session_info.txt")

ggsave(png_path, p, width = 125, height = 92, units = "mm", dpi = 320, bg = "white")
ggsave(pdf_path, p, width = 125, height = 92, units = "mm", device = cairo_pdf, bg = "white")

write.csv(
  timing[c(
    "mode", "batch_mean_s", "batch_std_s", "latency_ms",
    "latency_std_ms", "overhead_pct"
  )],
  csv_path,
  row.names = FALSE
)

capture.output(sessionInfo(), file = session_path)

message("Saved preview: ", png_path)
message("Saved vector figure: ", pdf_path)
