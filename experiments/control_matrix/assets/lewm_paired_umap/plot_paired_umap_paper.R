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
output_dir <- get_arg("--output-dir", file.path(input_dir, "figures"))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_dir <- normalizePath(output_dir, winslash = "/", mustWork = TRUE)

read_audit <- function(task) {
  path <- file.path(
    input_dir,
    task,
    paste0(task, "_lewm_paired_umap_audit.csv")
  )
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

tworoom <- read_audit("tworoom")
pusht <- read_audit("pusht")
audit <- rbind(tworoom, pusht)

required_columns <- c(
  "task", "global_idx", "umap_1", "umap_2", "nominal_cluster",
  "stability_fraction", "stability_count"
)
if (length(setdiff(required_columns, names(audit))) > 0L) {
  stop("Audit CSV schema mismatch")
}
if (nrow(tworoom) != 20000L || nrow(pusht) != 20000L) {
  stop("Expected exactly 20,000 landmark-disjoint audit points per task")
}
if (!identical(sort(unique(audit$nominal_cluster)), 1:3)) {
  stop("Expected stored nominal_cluster values 1, 2, and 3")
}

# These checks guard the existing audit data; they do not recompute UMAP,
# spectral partitions, landmark draws, or Hungarian label alignments.
stable_fraction <- vapply(
  split(audit$stability_count, audit$task),
  function(x) mean(x == 9L),
  numeric(1)
)
if (!isTRUE(all.equal(unname(stable_fraction[["tworoom"]]), 0.97325))) {
  stop("Unexpected TwoRoom nine-run stability fraction")
}
if (!isTRUE(all.equal(unname(stable_fraction[["pusht"]]), 0.65435))) {
  stop("Unexpected PushT nine-run stability fraction")
}

panel_labels <- c(
  tworoom = "(a) TwoRoom \u2014 Accepted (97.32% stable)",
  pusht = "(b) PushT \u2014 Rejected (65.43% stable)"
)
legend_levels <- c(
  "Nominal region 0",
  "Nominal region 1",
  "Nominal region 2",
  "Changed in >=1 run"
)
region_colors <- c(
  "Nominal region 0" = "#0072B2",
  "Nominal region 1" = "#D55E00",
  "Nominal region 2" = "#009E73",
  "Changed in >=1 run" = "#CC79A7"
)

audit$task_panel <- factor(
  panel_labels[audit$task],
  levels = unname(panel_labels)
)
audit$display_group <- ifelse(
  audit$stability_count == 9L,
  paste("Nominal region", audit$nominal_cluster - 1L),
  "Changed in >=1 run"
)
audit$display_group <- factor(audit$display_group, levels = legend_levels)

# Draw changed points last so that instability remains visible in dense regions.
audit <- audit[
  order(audit$task, audit$stability_count != 9L, audit$global_idx),
  ,
  drop = FALSE
]

paper_plot <- ggplot(audit, aes(x = umap_1, y = umap_2)) +
  geom_point(
    aes(color = display_group),
    size = 0.30,
    alpha = 0.78,
    stroke = 0
  ) +
  facet_wrap(~task_panel, scales = "free", nrow = 1) +
  scale_color_manual(values = region_colors, limits = legend_levels, drop = FALSE) +
  guides(
    color = guide_legend(
      title = NULL,
      nrow = 1,
      byrow = TRUE,
      override.aes = list(size = 2.5, alpha = 1),
      keywidth = grid::unit(12, "pt"),
      keyheight = grid::unit(9, "pt")
    )
  ) +
  labs(x = "UMAP 1", y = "UMAP 2", color = NULL) +
  theme_classic(base_size = 8.5, base_family = "serif") +
  theme(
    plot.title = element_blank(),
    plot.subtitle = element_blank(),
    plot.caption = element_blank(),
    strip.background = element_blank(),
    strip.text = element_text(
      size = 9.4,
      face = "bold",
      color = "#181818",
      margin = margin(t = 1.5, b = 4.5)
    ),
    axis.title = element_text(size = 8.5, face = "plain", color = "#202020"),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    axis.line = element_blank(),
    panel.grid = element_blank(),
    panel.border = element_blank(),
    panel.spacing.x = grid::unit(8, "pt"),
    aspect.ratio = 0.80,
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.text = element_text(size = 8.2, face = "plain", color = "#202020"),
    legend.spacing.x = grid::unit(4, "pt"),
    legend.margin = margin(t = 1, r = 0, b = 0, l = 0),
    plot.margin = margin(t = 3, r = 3, b = 2, l = 3)
  )

stem <- file.path(output_dir, "lewm_tworoom_pusht_paired_umap_paper")
ggsave(
  paste0(stem, ".png"),
  plot = paper_plot,
  width = 7.16,
  height = 3.62,
  units = "in",
  dpi = 400,
  bg = "white"
)

pdf_device <- if (capabilities("cairo")) grDevices::cairo_pdf else "pdf"
ggsave(
  paste0(stem, ".pdf"),
  plot = paper_plot,
  width = 7.16,
  height = 3.62,
  units = "in",
  device = pdf_device,
  bg = "white"
)

writeLines(
  c(
    capture.output(sessionInfo()),
    "",
    paste0("input_dir = ", input_dir),
    "audit_rows = 40000",
    "stored_coordinates_reused = true",
    "stored_nominal_assignments_reused = true",
    "stored_nine_run_stability_counts_reused = true",
    "output_width_in = 7.16",
    "output_height_in = 3.62"
  ),
  file.path(output_dir, "lewm_tworoom_pusht_paired_umap_paper_sessionInfo.txt"),
  useBytes = TRUE
)

message("Wrote paper paired UMAP PDF/PNG to: ", output_dir)
