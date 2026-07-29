#!/usr/bin/env Rscript

# TwoRoom geometry region masks drawn with ggplot2.
# The first five panels show region masks. The lower-right panel shows only
# the TwoRoom task geometry, without any region-mask overlay.
#
# Source of constants:
# experiments/tworoom/README.md
# experiments/tworoom/gauge_drift.py

suppressPackageStartupMessages({
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (!is.na(idx) && idx < length(args)) {
    return(args[[idx + 1L]])
  }
  default
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg) > 0L) sub("^--file=", "", script_arg[[1L]]) else NA_character_
script_dir <- if (!is.na(script_file)) dirname(normalizePath(script_file, mustWork = FALSE)) else getwd()
output_dir <- normalizePath(get_arg("--output-dir", file.path(script_dir, "outputs")), mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

img_size <- 224
pos_min <- 14
pos_max <- 209
wall_lo <- 107
wall_center <- 112
wall_hi <- 117
near_lo <- 29
near_hi <- 194
common_y_lo <- 34
common_y_hi <- 189
common_left_lo <- 34
common_left_hi <- 87
common_right_lo <- 137
common_right_hi <- 189

region_levels <- c(
  "left_room",
  "doorway_corridor",
  "right_room",
  "near_wall",
  "common"
)

panel_levels <- c(
  "TwoRoom task",
  region_levels
)

panel_labels <- c(
  left_room = "left_room",
  doorway_corridor = "doorway_corridor",
  right_room = "right_room",
  near_wall = "near_wall",
  common = "common",
  "TwoRoom task" = "TwoRoom task"
)

canvas <- data.frame(
  panel = panel_levels,
  xmin = 0,
  xmax = img_size,
  ymin = 0,
  ymax = img_size
)

playable <- data.frame(
  panel = panel_levels,
  xmin = pos_min,
  xmax = pos_max,
  ymin = pos_min,
  ymax = pos_max
)

mask_rects <- rbind(
  data.frame(panel = "left_room", region = "left_room", xmin = pos_min, xmax = wall_lo, ymin = pos_min, ymax = pos_max),
  data.frame(panel = "doorway_corridor", region = "doorway_corridor", xmin = wall_lo, xmax = wall_hi, ymin = pos_min, ymax = pos_max),
  data.frame(panel = "right_room", region = "right_room", xmin = wall_hi, xmax = pos_max, ymin = pos_min, ymax = pos_max),
  data.frame(panel = "near_wall", region = "near_wall", xmin = c(pos_min, pos_min, pos_min, near_hi), xmax = c(pos_max, pos_max, near_lo, pos_max), ymin = c(pos_min, near_hi, near_lo, near_lo), ymax = c(near_lo, pos_max, near_hi, near_hi)),
  data.frame(panel = "common", region = "common", xmin = c(common_left_lo, common_right_lo), xmax = c(common_left_hi, common_right_hi), ymin = c(common_y_lo, common_y_lo), ymax = c(common_y_hi, common_y_hi))
)

task_rooms <- rbind(
  data.frame(panel = "TwoRoom task", part = "left room", xmin = pos_min, xmax = wall_lo, ymin = pos_min, ymax = pos_max),
  data.frame(panel = "TwoRoom task", part = "center wall", xmin = wall_lo, xmax = wall_hi, ymin = pos_min, ymax = pos_max),
  data.frame(panel = "TwoRoom task", part = "right room", xmin = wall_hi, xmax = pos_max, ymin = pos_min, ymax = pos_max)
)

task_labels <- data.frame(
  panel = "TwoRoom task",
  label = c("room", "room"),
  x = c((pos_min + wall_lo) / 2, (wall_hi + pos_max) / 2),
  y = c(wall_center, wall_center)
)

canvas$panel <- factor(canvas$panel, levels = panel_levels)
playable$panel <- factor(playable$panel, levels = panel_levels)
mask_rects$panel <- factor(mask_rects$panel, levels = panel_levels)
mask_rects$region <- factor(mask_rects$region, levels = region_levels)
task_rooms$panel <- factor(task_rooms$panel, levels = panel_levels)
task_labels$panel <- factor(task_labels$panel, levels = panel_levels)

mask_colors <- c(
  left_room = "#56B4E9",
  doorway_corridor = "#CC79A7",
  right_room = "#009E73",
  near_wall = "#E69F00",
  common = "#0072B2"
)

task_colors <- c(
  "left room" = "#F8FAFC",
  "center wall" = "#B8BEC9",
  "right room" = "#F8FAFC"
)

region_plot <- ggplot() +
  geom_rect(
    data = canvas,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    fill = "#F5F6F8",
    color = "#D0D5DD",
    linewidth = 0.25
  ) +
  geom_rect(
    data = playable,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    fill = "#FFFFFF",
    color = "#222222",
    linewidth = 0.35
  ) +
  geom_rect(
    data = task_rooms,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = part),
    color = "#2F3542",
    linewidth = 0.28,
    alpha = 1
  ) +
  geom_rect(
    data = mask_rects,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = region),
    alpha = 0.78,
    color = "#222222",
    linewidth = 0.2
  ) +
  geom_vline(xintercept = wall_center, linetype = "22", linewidth = 0.25, color = "#444444") +
  geom_text(
    data = task_labels,
    aes(x = x, y = y, label = label),
    size = 3,
    color = "#4B5563",
    fontface = "plain"
  ) +
  facet_wrap(~ panel, ncol = 3, labeller = as_labeller(panel_labels)) +
  coord_fixed(xlim = c(0, img_size), ylim = c(img_size, 0), expand = FALSE, clip = "off") +
  scale_y_reverse(breaks = c(0, wall_center, img_size)) +
  scale_x_continuous(breaks = wall_center, labels = "112") +
  scale_fill_manual(values = c(mask_colors, task_colors), guide = "none") +
  labs(
    title = "TwoRoom geometry region masks",
    x = "x",
    y = "y"
  ) +
  theme_bw(base_size = 9) +
  theme(
    panel.grid.major = element_line(color = "#E6E8EC", linewidth = 0.2),
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "#F0F2F5", color = "#B8C0CC", linewidth = 0.3),
    strip.text = element_text(face = "bold", size = 8),
    plot.title = element_text(face = "bold", size = 10, hjust = 0),
    axis.title = element_text(size = 8),
    axis.text = element_text(size = 6),
    panel.spacing.x = unit(5, "mm"),
    panel.spacing.y = unit(3, "mm"),
    plot.margin = margin(5, 5, 5, 5)
  )

pdf_path <- file.path(output_dir, "tworoom_geometry_region_masks_with_task.pdf")
png_path <- file.path(output_dir, "tworoom_geometry_region_masks_with_task.png")

pdf_device <- if (capabilities("cairo")) grDevices::cairo_pdf else "pdf"

ggsave(pdf_path, region_plot, width = 150, height = 105, units = "mm", device = pdf_device)
ggsave(png_path, region_plot, width = 150, height = 105, units = "mm", dpi = 300)

session_path <- file.path(output_dir, "plot_tworoom_geometry_regions_session_info.txt")
writeLines(c(capture.output(sessionInfo()), "", "Generated files:", pdf_path, png_path), session_path)

message("Wrote ggplot2 region masks to: ", output_dir)
message(png_path)
