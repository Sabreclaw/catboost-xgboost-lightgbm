# --- 1. Load Libraries ---
library(ggplot2)
library(gridExtra)
library(patchwork)
library(grid)  # For adding separators

# --- 2. Load and Filter Data ---
file_name <- "experiment-results/run_table.csv" 
df_raw <- read.csv(file_name)
df <- subset(df_raw, total_requests >= 15000)
df$mean_memory_mb <- df$mean_memory_gb * 1000

# --- 3. Prepare Data for Plotting ---
df$database <- as.factor(df$database)
df$model <- factor(df$model, levels = c("xgboost", "lgbm", "catboost"))

df$database <- recode(df$database,
                      "credit_card_transactions" = "Credit Card Transactions Dataset",
                      "diabetic" = "Diabetics Prediction Dataset", 
                      "healthcare-dataset-stroke" = "Healthcare Stroke Dataset",
                      "UNSW_NB15_merged" = "UNSW_NB15 Dataset")

# --- 4. Define Plot Function with Better Margins ---
create_plot <- function(metric_to_plot, metric_ylabel, metric_title) {
  ggplot(df, aes(x = model, y = .data[[metric_to_plot]], fill = model)) +
    geom_boxplot(alpha = 0.8, outlier.shape = NA) +
    geom_jitter(width = 0.2, alpha = 0.3, size = 1) +
    facet_wrap(~ database, scales = "free_y") +
    labs(title = metric_title, x = "Algorithm", y = metric_ylabel) +
    theme_minimal() +
    theme(
      legend.position = "none",
      plot.title = element_text(hjust = 0.5, size = 10),
      strip.text = element_text(face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.margin = margin(15, 15, 15, 15)  # Added margin around each plot
    )
}

# --- 5. Create All Plots (without legends) ---
plot1 <- create_plot("mean_memory_mb", "Memory Usage (mb)", "Memory Usage")
plot2 <- create_plot("mean_latency_ms", "Execution Time (ms)", "Execution Time")
plot3 <- create_plot("mean_cpu_percent", "CPU Usage (%)", "CPU Usage")
plot4 <- create_plot("energy_j", "Energy (J)", "Energy Consumption")

# --- 6. Extract Legend from one plot ---
get_legend <- function(plot) {
  temp_plot <- plot + 
    theme(legend.position = "bottom") +
    guides(fill = guide_legend(title = "Algorithm"))
  legend <- cowplot::get_legend(temp_plot)
  return(legend)
}

plot_with_legend <- ggplot(df, aes(x = model, y = mean_memory_mb, fill = model)) +
  geom_boxplot() +
  labs(fill = "Algorithm") +
  theme(legend.position = "bottom")

legend <- cowplot::get_legend(plot_with_legend)

# --- 7. Create Separator Functions ---
# Function to create a vertical separator line
vertical_separator <- function() {
  grid::linesGrob(x = unit(0.5, "npc"), 
                  y = unit(c(0, 1), "npc"),
                  gp = grid::gpar(col = "gray70", lty = "dashed", lwd = 1))
}

# Function to create a horizontal separator line
horizontal_separator <- function() {
  grid::linesGrob(y = unit(0.5, "npc"), 
                  x = unit(c(0, 1), "npc"),
                  gp = grid::gpar(col = "gray70", lty = "dashed", lwd = 1))
}

# Function to create empty space as separator
empty_space <- function(width = unit(0.5, "cm")) {
  grid::rectGrob(gp = grid::gpar(col = NA, fill = NA), width = width)
}

# --- 8. Arrange plots with separators ---

# Option A: With dotted line separators
final_plot <- grid.arrange(
  # First row: plot1, vertical separator, plot2
  arrangeGrob(
    plot1, 
    vertical_separator(), 
    plot2,
    ncol = 3, 
    widths = c(1, unit(0.02, "npc"), 1)  # Small space for vertical line
  ),
  
  # Horizontal separator between rows
  horizontal_separator(),
  
  # Second row: plot3, vertical separator, plot4
  arrangeGrob(
    plot3, 
    vertical_separator(), 
    plot4,
    ncol = 3, 
    widths = c(1, unit(0.02, "npc"), 1)
  ),
  
  # Horizontal separator before legend
  horizontal_separator(),
  
  # Legend at bottom
  legend,
  
  nrow = 5,
  heights = c(1, unit(0.02, "npc"), 1, unit(0.02, "npc"), 0.1)  # Adjust heights
)

# Display the final plot
print(final_plot)

