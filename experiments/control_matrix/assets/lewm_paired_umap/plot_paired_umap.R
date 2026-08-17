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

input_dir <- normalizePath(required_arg("--input-dir"), winslash = "/", mustWork = TRUE)
output_dir <- get_arg("--output-dir", input_dir)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_dir <- normalizePath(output_dir, winslash = "/", mustWork = TRUE)

read_task <- function(task) {
  main <- read.csv(
    file.path(input_dir, task, paste0(task, "_lewm_paired_umap_audit.csv")),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  draws <- read.csv(
    file.path(input_dir, task, paste0(task, "_lewm_paired_umap_draws.csv")),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  list(main = main, draws = draws)
}

tworoom <- read_task("tworoom")
pusht <- read_task("pusht")
main <- rbind(tworoom$main, pusht$main)
draws <- rbind(tworoom$draws, pusht$draws)

required_main <- c(
  "task", "global_idx", "umap_1", "umap_2", "nominal_cluster",
  "stability_fraction", "stability_count"
)
required_draws <- c(
  "task", "seed", "knn", "global_idx", "umap_1", "umap_2",
  "aligned_cluster", "agrees_with_nominal"
)
if (length(setdiff(required_main, names(main))) > 0L) stop("Main CSV schema mismatch")
if (length(setdiff(required_draws, names(draws))) > 0L) stop("Draw CSV schema mismatch")
if (nrow(main) != 40000L) stop("Expected 20,000 audit points per task")
if (nrow(draws) != 360000L) stop("Expected 20,000 points x 9 draws x 2 tasks")

stable_by_task <- aggregate(
  stability_count ~ task,
  data = main,
  FUN = function(x) mean(x == 9L)
)
stable_percent <- setNames(
  floor(10000 * stable_by_task$stability_count + 1e-9) / 100,
  stable_by_task$task
)
task_labels <- c(
  tworoom = sprintf("TwoRoom — %.2f%% stable across all 9 runs", stable_percent[["tworoom"]]),
  pusht = sprintf("PushT — %.2f%% stable across all 9 runs", stable_percent[["pusht"]])
)
cluster_labels <- c("Region 1", "Region 2", "Region 3")
cluster_colors <- c(
  "Region 1" = "#0072B2",
  "Region 2" = "#D55E00",
  "Region 3" = "#009E73",
  "Changed in ≥1 run" = "#CC79A7"
)

main$task_panel <- factor(task_labels[main$task], levels = unname(task_labels))
main$cluster <- factor(
  paste("Region", main$nominal_cluster),
  levels = cluster_labels
)
main$display_group <- ifelse(
  main$stability_count == 9L,
  as.character(main$cluster),
  "Changed in ≥1 run"
)
main$display_group <- factor(
  main$display_group,
  levels = c(cluster_labels, "Changed in ≥1 run")
)
main <- main[order(main$task, main$stability_count != 9L, main$global_idx), ]

base_theme <- theme_bw(base_size = 12, base_family = "serif") +
  theme(
    plot.title = element_text(size = 19, face = "bold", hjust = 0),
    plot.subtitle = element_text(size = 11.6, color = "#485464", margin = margin(b = 10)),
    plot.caption = element_text(size = 10.5, color = "#485464", hjust = 0, lineheight = 1.1),
    axis.title = element_text(size = 13),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "#687386", linewidth = 0.7),
    strip.background = element_rect(fill = "#F2F5F9", color = "#687386", linewidth = 0.7),
    strip.text = element_text(size = 12.2, face = "bold", lineheight = 1.12, margin = margin(7, 5, 7, 5)),
    legend.position = "bottom",
    legend.box = "vertical",
    legend.title = element_text(face = "bold", size = 11.5),
    legend.text = element_text(size = 10.5),
    plot.margin = margin(16, 16, 12, 14)
  )

p_main <- ggplot(main, aes(x = umap_1, y = umap_2)) +
  geom_point(
    aes(color = display_group),
    size = 0.42,
    alpha = 0.72,
    stroke = 0
  ) +
  facet_wrap(~task_panel, scales = "free", nrow = 1) +
  scale_color_manual(values = cluster_colors, drop = FALSE) +
  guides(
    color = guide_legend(
      title = "Nine-run stability",
      override.aes = list(size = 3.4, alpha = 1),
      nrow = 1
    )
  ) +
  labs(
    title = "LeWM latent spectral-partition stability",
    subtitle = paste0(
      "Points unchanged across all 9 runs retain their nominal ",
      "Spectral K3 color; every changed point is highlighted in purple"
    ),
    caption = paste0(
      "Each panel uses an independent label-free StandardScaler → PCA(50) → UMAP fit with identical parameters\n",
      "on 20,000 landmark-disjoint audit latents. Candidate IDs are aligned with maximum-overlap Hungarian matching."
    ),
    x = "UMAP 1",
    y = "UMAP 2",
    color = NULL
  ) +
  base_theme +
  theme(aspect.ratio = 1)

save_plot <- function(plot, stem, width, height) {
  ggsave(
    file.path(output_dir, paste0(stem, ".png")),
    plot = plot,
    width = width,
    height = height,
    units = "in",
    dpi = 320,
    bg = "white"
  )
  pdf_device <- if (capabilities("cairo")) grDevices::cairo_pdf else "pdf"
  ggsave(
    file.path(output_dir, paste0(stem, ".pdf")),
    plot = plot,
    width = width,
    height = height,
    units = "in",
    device = pdf_device,
    bg = "white"
  )
}

save_plot(p_main, "lewm_tworoom_pusht_paired_umap_main", 13.4, 7.8)

make_grid <- function(task) {
  data <- draws[draws$task == task, , drop = FALSE]
  data$cluster <- factor(
    paste("Region", data$aligned_cluster),
    levels = cluster_labels
  )
  data$seed_panel <- factor(
    paste("Landmark seed", data$seed),
    levels = paste("Landmark seed", 0:2)
  )
  data$knn_panel <- factor(
    paste("kNN", data$knn),
    levels = paste("kNN", c(27, 30, 33))
  )
  data <- data[order(data$seed, data$knn, data$global_idx), ]
  subtitle <- "Every panel is aligned to the nominal seed-0/kNN-30 partition"
  title <- if (task == "tworoom") {
    "TwoRoom LeWM: nine Spectral K3 runs"
  } else {
    "PushT LeWM: nine Spectral K3 runs"
  }
  ggplot(data, aes(x = umap_1, y = umap_2, color = cluster)) +
    geom_point(size = 0.23, alpha = 0.58, stroke = 0) +
    facet_grid(seed_panel ~ knn_panel, scales = "free") +
    scale_color_manual(values = cluster_colors, drop = FALSE) +
    guides(
      color = guide_legend(
        title = "Hungarian-aligned label",
        override.aes = list(size = 3.2, alpha = 1),
        nrow = 1
      )
    ) +
    labs(
      title = title,
      subtitle = subtitle,
      caption = paste0(
        "All nine panels reuse the same label-free UMAP coordinates; only the ",
        "candidate partition labels change."
      ),
      x = "UMAP 1",
      y = "UMAP 2",
      color = NULL
    ) +
    base_theme +
    theme(
      strip.text = element_text(size = 10.6, face = "bold"),
      legend.box = "horizontal"
    )
}

save_plot(
  make_grid("tworoom"),
  "lewm_tworoom_spectral_diagnostic_grid",
  10.8,
  9.2
)
save_plot(
  make_grid("pusht"),
  "lewm_pusht_spectral_diagnostic_grid",
  10.8,
  9.2
)

writeLines(
  c(
    capture.output(sessionInfo()),
    "",
    paste0("input_dir = ", input_dir),
    "main_rows = 40000",
    "draw_rows = 360000",
    "body_font = Times New Roman via the Windows serif alias",
    "stability_definition = fraction of 9 Hungarian-aligned labels equal to nominal seed-0/kNN-30 label",
    "figure_annotation = dataset names and partition stability only; gate diagnostics remain in README"
  ),
  file.path(output_dir, "R_sessionInfo.txt"),
  useBytes = TRUE
)

message("Wrote paired LeWM UMAP PNG/PDF figures to: ", output_dir)
