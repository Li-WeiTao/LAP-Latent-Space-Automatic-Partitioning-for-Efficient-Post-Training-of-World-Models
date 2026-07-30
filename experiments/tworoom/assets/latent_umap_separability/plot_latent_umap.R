#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(ggplot2))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg) == 1) {
  normalizePath(sub("^--file=", "", script_arg), winslash = "/")
} else {
  normalizePath("plot_latent_umap.R", winslash = "/")
}
script_dir <- dirname(script_path)
args <- commandArgs(trailingOnly = TRUE)
input_csv <- if (length(args) >= 1) args[[1]] else file.path(script_dir, "data", "latent_umap_test_coordinates.csv")
output_dir <- if (length(args) >= 2) args[[2]] else script_dir
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(input_csv, stringsAsFactors = FALSE, check.names = FALSE)
required_cols <- c("umap_1", "umap_2", "rooms3", "priority5")
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop(paste("Missing columns:", paste(missing_cols, collapse = ", ")))
}

df$rooms3 <- factor(
  df$rooms3,
  levels = c("Left room", "Doorway corridor", "Right room")
)
df$priority5[df$priority5 == "Common interior"] <- "Common"
df$priority5 <- factor(
  df$priority5,
  levels = c(
    "Left room", "Doorway corridor", "Right room",
    "Near wall", "Common"
  )
)

region_colors <- c(
  "Left room" = "#0066CC",
  "Doorway corridor" = "#111111",
  "Right room" = "#D73027",
  "Near wall" = "#7B3294",
  "Common" = "#00A878"
)

base_theme <- theme_classic(base_size = 16) +
  theme(
    plot.title = element_text(size = 19, face = "bold", hjust = 0),
    plot.subtitle = element_text(size = 12.5, color = "#444444", margin = margin(b = 9)),
    axis.title = element_text(size = 15),
    axis.text = element_text(size = 11, color = "#333333"),
    axis.ticks = element_blank(),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 12),
    legend.key.width = grid::unit(1.1, "lines"),
    plot.margin = margin(12, 16, 10, 12)
  )

make_umap_plot <- function(data, label_col, title, subtitle) {
  draw_order <- if (label_col == "rooms3") {
    c("Left room", "Right room", "Doorway corridor")
  } else {
    c("Common", "Left room", "Right room", "Near wall", "Doorway corridor")
  }
  data$.draw_order <- match(as.character(data[[label_col]]), draw_order)
  data <- data[order(data$.draw_order), ]

  ggplot(data, aes(x = umap_1, y = umap_2, color = .data[[label_col]])) +
    geom_point(size = 0.82, alpha = 0.68, stroke = 0) +
    scale_color_manual(values = region_colors, drop = FALSE) +
    coord_equal() +
    labs(
      title = title,
      subtitle = subtitle,
      x = "x",
      y = "y",
      color = NULL
    ) +
    guides(color = guide_legend(override.aes = list(size = 3.2, alpha = 1), nrow = 1)) +
    base_theme
}

p_rooms3 <- make_umap_plot(
  df,
  "rooms3",
  "Linear Separability in the Latent Space (3-Partition)",
  expression("Linear Probe Macro-F1 = 99.67%" %+-% "0.04% (5 seeds)")
)

p_priority5 <- make_umap_plot(
  df,
  "priority5",
  "Linear Separability in the Latent Space (5-Partition)",
  expression("Linear Probe Macro-F1 = 98.01%" %+-% "0.08% (5 seeds)")
)

save_plot <- function(plot, stem) {
  ggsave(
    file.path(output_dir, paste0(stem, ".png")),
    plot = plot,
    width = 9.4,
    height = 7.2,
    units = "in",
    dpi = 320,
    bg = "white"
  )
  ggsave(
    file.path(output_dir, paste0(stem, ".pdf")),
    plot = plot,
    width = 9.4,
    height = 7.2,
    units = "in",
    device = cairo_pdf,
    bg = "white"
  )
}

save_plot(p_rooms3, "latent_umap_rooms3")
save_plot(p_priority5, "latent_umap_priority5")

capture.output(sessionInfo(), file = file.path(output_dir, "ggplot2_session_info.txt"))
message("Wrote UMAP figures to: ", normalizePath(output_dir, winslash = "/", mustWork = FALSE))
