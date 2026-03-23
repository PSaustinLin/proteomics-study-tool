library(readxl)
library(ggplot2)
library(ggbeeswarm)
library(dplyr)
library(tidyr)
library(ggsignif)  # For statistical significance brackets
library(broom)     # For tidy ANOVA results
library(patchwork) # For combining plots

# File and sheet configuration
file_path <- "C:/tmp_space/ApoA2_A1AT_manuscript/BC_APOA2_redo.xlsm"
sheet <- "SUM (2)"

# Column configuration
column_num <- 1 # elements in first_columns <- c(), or variable number
# Original column names in xlsx file

first_columns <- c('fD/B')
second_columns <- c('0D/B')
third_columns <- c('1D/B')
forth_columns <- c('2D/B')

# Column names for data frames (as character strings)
new_labels <- c('D/B')

# Mathematical expressions for plot y-axis title (moved from x-axis)
plot_labels <- c(expression(frac(P[60], P[20])))

# Read the Excel file
df1 <- read_excel(file_path, sheet = sheet, col_names = TRUE) %>%
    select(all_of(first_columns))
names(df1) <- new_labels

df2 <- read_excel(file_path, sheet = sheet, col_names = TRUE) %>%
    select(all_of(second_columns))
names(df2) <- new_labels

df3 <- read_excel(file_path, sheet = sheet, col_names = TRUE) %>%
    select(all_of(third_columns))
names(df3) <- new_labels

df4 <- read_excel(file_path, sheet = sheet, col_names = TRUE) %>%
    select(all_of(forth_columns))
names(df4) <- new_labels


# Add dataset identifier
df1$dataset <- "HS"
df2$dataset <- "0"
df3$dataset <- "I"
df4$dataset <- "II"

# Combine all datasets
df_combined <- rbind(df1, df2, df3, df4)

# Convert to long format for ggplot and remove missing values
df_long <- df_combined %>%
    pivot_longer(cols = all_of(new_labels), names_to = "group", values_to = "value") %>%
    filter(!is.na(value) & is.finite(value)) %>%  # Remove NA and infinite values
    mutate(
        group = factor(group, levels = new_labels),
        dataset = factor(dataset, levels = c("HS", "0", "I", "II"))
    )

# Create paired grouping variable
df_long <- df_long %>%
    mutate(
        pair = case_when(
            group %in% c("HS") ~ "Pair1",
            group %in% c("0") ~ "Pair2",
            group %in% c("I") ~ "Pair3",
            group %in% c("II") ~ "Pair4",
            TRUE ~ "Single"
        )
    )

# Set figure dimensions based on column_num
fig_width <- 8
fig_height <- 9
x_rotation <- 0

# Create proper positioning variables for dodging (set 0.25 for 5 columns)
position_dodge_width <- 0.25 # 組內數據間距

# Calculate centering offset based on number of datasets
n_datasets <- length(unique(df_long$dataset))
center_offset <- (n_datasets + 1) / 2

df_long <- df_long %>%
    mutate(
        # Create numeric position for manual dodging
        group_numeric = as.numeric(group),
        dataset_numeric = as.numeric(dataset),
        # Calculate x position with proper centering for any number of datasets
        x_position = group_numeric + (dataset_numeric - center_offset) * 2 * position_dodge_width / n_datasets,
        group_dataset = interaction(group, dataset, sep = "_")
    )

# Calculate means for each group with proper positioning
means <- df_long %>%
    group_by(group, dataset, group_numeric, dataset_numeric) %>%
    summarise(
        mean_value = mean(value, na.rm = TRUE),
        x_position = first(x_position),
        .groups = 'drop'
    )

# Perform one-way ANOVA for each group
anova_results <- list()
pairwise_results <- list()

for (group_name in new_labels) {
    # Subset data for current group
    group_data <- df_long %>% filter(group == group_name)

    # Perform one-way ANOVA
    anova_model <- aov(value ~ dataset, data = group_data)
    anova_summary <- summary(anova_model)

    # Extract p-value
    p_value <- anova_summary[[1]][["Pr(>F)"]][1]

    # Store results
    anova_results[[group_name]] <- list(
        model = anova_model,
        p_value = p_value,
        summary = anova_summary
    )

    # Perform pairwise t-tests if ANOVA is significant
    if (p_value < 0.05) {
        pairwise <- pairwise.t.test(group_data$value, group_data$dataset,
                                    p.adjust.method = "bonferroni")
        pairwise_results[[group_name]] <- pairwise
    }
}

# Function to convert p-value to scientific notation expression
format_p_value_expression <- function(p_val) {
    if (is.na(p_val)) return("NA")

    # Convert to scientific notation
    exponent <- floor(log10(p_val))
    mantissa <- p_val / (10^exponent)

    # Round mantissa to 1 decimal place
    mantissa <- round(mantissa, 1)

    # Create expression string for parsing
    expr_string <- paste0(mantissa, " %*% 10^", exponent)
    return(expr_string)
}

# Create pairwise comparison annotations - FIXED to use same positioning as beeswarm
create_pairwise_annotations <- function(df_long, new_labels) {
    annotations <- list()
    datasets <- c("HS", "0", "I", "II")

    # Create lookup table for x_positions by group and dataset
    x_position_lookup <- df_long %>%
        select(group, dataset, x_position) %>%
        distinct()

    # Create specific comparisons: 1 vs 2, and all combinations among 2-5
    pairwise_combinations <- list(
        c(1, 2),  # HS vs 0
        c(1, 3),  # HS vs I
        c(1, 4)   # HS vs II
    )

    for (group_idx in seq_along(new_labels)) {
        group_name <- new_labels[group_idx]
        group_data <- df_long %>% filter(group == group_name)

        for (comp_idx in seq_along(pairwise_combinations)) {
            pair <- pairwise_combinations[[comp_idx]]
            dataset1 <- datasets[pair[1]]
            dataset2 <- datasets[pair[2]]

            # Get data for comparison
            data1 <- group_data %>% filter(dataset == dataset1) %>% pull(value)
            data2 <- group_data %>% filter(dataset == dataset2) %>% pull(value)

            # Perform t-test
            if (length(data1) > 1 && length(data2) > 1) {
                t_test <- t.test(data1, data2)
                p_val <- t_test$p.value
            } else {
                p_val <- NA
            }

            # FIXED: Get x positions directly from lookup table (same as beeswarm)
            x1 <- group_idx + (pair[1] - center_offset) * position_dodge_width * 2.5 / n_datasets
            x2 <- group_idx + (pair[2] - center_offset) * position_dodge_width * 2.5 / n_datasets

            # Store annotation data
            annotations[[paste0(group_name, "_", comp_idx)]] <- list(
                group = group_name,
                group_idx = group_idx,
                x1 = x1,
                x2 = x2,
                p_value = p_val,
                p_expression = format_p_value_expression(p_val),
                comparison = paste0(dataset1, " vs ", dataset2),
                comp_idx = comp_idx
            )
        }
    }

    return(annotations)
}

# Get pairwise annotations with corrected positioning
pairwise_annotations <- create_pairwise_annotations(df_long, new_labels)

# serial_colors <- c("white", "lightgrey", "#c0dee5", "#61a4ad", "#07575b")
serial_colors <- c("white", "lightgrey", "#c0dee5", "#61a4ad")
# Create the main data plot
main_plot <- ggplot(df_long, aes(x = group, y = value, fill = dataset)) +

    # Add boxplot with outliers - aligned with swarm positions
    geom_boxplot(width = 0.3,
                 color = "black",
                 linewidth = 0.5,
                 staplewidth = 0.5,
                 lineend = "square",
                 fatten = 1,
                 outlier.size = 1.5,
                 outlier.alpha = 0.5,
                 outlier.shape = NA,
                 position = position_dodge(width = 2 * position_dodge_width, preserve = "single")) +

    # Add mean points - aligned with swarm positions
    geom_point(data = means,
               aes(x = x_position, y = mean_value, fill = dataset),
               shape = 3,  # plus sign
               size = 3,
               color = "black",
               stroke = 0.7,
               inherit.aes = FALSE) +

    # Add jittered points (swarm-like effect) - in front
    geom_beeswarm(aes(x = x_position, group = group_dataset),
                  alpha = 0.2, size = 2.0, pch = 16, # transparency, point size, shape
                  color = "black", cex = 1.2,  # spacing
                  method = "square") +

    # color the dodge pairs
    scale_fill_manual(values = serial_colors, name = "",
                      labels = c("HS", "0", "I", "II")) +

    # Customize y-axis to focus ONLY on data range
    scale_y_continuous(
        breaks = function(x) {
            breaks <- pretty(c(0, max(x, na.rm = TRUE)), n = 6)
            # Ensure 0 is included
            if (!0 %in% breaks) {
                breaks <- sort(c(0, breaks))
            }
            return(breaks)
        },
        limits = function(x) {
            max_val <- max(x, na.rm = TRUE)
            breaks <- pretty(c(0, max_val), n = 6)
            if (!0 %in% breaks) {
                breaks <- sort(c(0, breaks))
            }
            next_tick <- min(breaks[breaks > max_val])
            c(0, next_tick)
        },
        expand = expansion(mult = c(0, 0))
    ) +

    # Customize x-axis - REMOVE ticks and text
    scale_x_discrete(
        breaks = new_labels,
        labels = NULL,  # Remove x-axis labels
        expand = c(0.15, 0.15)
    ) +

    theme_minimal() +
    theme(
        # Remove grid
        panel.grid = element_blank(),

        # Remove x-axis text and ticks
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),

        # Customize y-axis text
        axis.text.y = element_text(size = 22, color = "black", hjust = 1, vjust = 0.5,
                                   margin = margin(r = 5)),

        # Customize legend text size
        legend.text = element_text(size = 30, color = "black", margin = margin(r = 15)),

        # Customize axis lines
        axis.line.x = element_line(color = "black", linewidth = 0.8, lineend = "square"),
        axis.line.y = element_line(color = "black", linewidth = 0.8, lineend = "square"),
        # Customize y-axis title - rotate 90 degrees
        axis.title.y = element_text(size = 24, color = "black", angle = 0, vjust = 0.5, margin = margin(r = 25)),
        # Keep y-axis ticks, remove x-axis ticks
        axis.ticks.y = element_line(color = "black", linewidth = 0.8),
        axis.ticks.length.y = unit(6, "pt"),

        # Remove background
        panel.background = element_blank(),
        plot.background = element_blank(),

        # Adjust margins
        plot.margin = margin(t = 0, r = 10, b = 0, l = 10, unit = "pt")
    ) +

    # Add y-axis title with mathematical expression
    labs(x = NULL, y = bquote(.(plot_labels[[1]]) ~ ""), size = 30)

# Create annotation plot
# Get the x-axis limits from df_long to match main_plot
x_min <- min(df_long$x_position, na.rm = TRUE) - 0.15  # Match main_plot's expand = c(0.15, 0.15)
x_max <- max(df_long$x_position, na.rm = TRUE) + 0.15

annotation_plot <- ggplot() +
    # Set x-axis limits to match the dodged positions in main_plot
    scale_x_continuous(limits = c(x_min, x_max),
                       breaks = unique(df_long$group_numeric),
                       labels = NULL) +  # Remove x-axis labels from annotation plot too
    ylim(0, 4) +  # Adjusted y-limit to accommodate 7 comparisons

    # Add horizontal comparison lines and p-values
    {
        # Create horizontal lines
        line_layers <- lapply(pairwise_annotations, function(ann) {
            if (!is.na(ann$p_value)) {
                # Adjusted positioning for 7 comparisons
                line_y <- ann$comp_idx * 0.8  # Spacing for 7 comparisons

                list(
                    # Horizontal line
                    annotate("segment", x = ann$x1, xend = ann$x2,
                             y = line_y, yend = line_y,
                             color = "black", linewidth = 0.5, lineend = "square"),
                    # Vertical lines at ends
                    annotate("segment", x = ann$x1, xend = ann$x1,
                             y = line_y - 0.1, yend = line_y,
                             color = "black", linewidth = 0.5, lineend = "square"),
                    annotate("segment", x = ann$x2, xend = ann$x2,
                             y = line_y - 0.1, yend = line_y,
                             color = "black", linewidth = 0.5, lineend = "square"),
                    # P-value text with expression formatting
                    annotate("text", x = (ann$x1 + ann$x2) / 2, y = line_y + 0.15,
                             label = ann$p_expression,
                             size = 8, hjust = 0.5, vjust = 0, color = "black", parse = TRUE)
                )
            }
        })
        # Flatten the list of lists
        unlist(line_layers, recursive = FALSE)
    } +

    theme_void() +  # Completely clean theme
    theme(
        plot.background = element_blank(),
        panel.background = element_blank(),
        plot.margin = margin(t = 0, r = 10, b = 0, l = 10, unit = "pt")
    )

# Combine plots using patchwork with height ratio and transparent background
combined_plot <- annotation_plot / main_plot +
    plot_layout(heights = c(1, 2)) +  # Adjust height ratio for comparisons
    plot_annotation(theme = theme(plot.background = element_rect(fill = "transparent", color = NA)))

# Display the combined plot
print(combined_plot)

# Save the combined plot
ggsave("plot_1var_female.png", plot = combined_plot, width = fig_width, height = fig_height,
       units = "in", dpi = 600)
