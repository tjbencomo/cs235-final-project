library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(stringr)

data_dir <- "~/code/cs235-data/figures"
genes <- c("ER", "HER2", "PR")
dropout <- c("small", "medium", "high")
aucs <- list()
i <- 1
for (g in genes) {
  for (d in dropout) {
    x <- read_csv(file.path(data_dir, paste(g, "dropout", d, "classifier", "training", "results.csv", sep="_")))
    x$gene <- g
    x$dropout <- d
    aucs[[i]] <- x
    i <- i + 1
  }
}
df <- bind_rows(aucs)
df <- df %>%
  bind_rows() %>%
  group_by(gene, dropout) %>%
  mutate(epoch = row_number()) %>%
  pivot_longer(!all_of(c("epoch", "gene", "dropout")), names_to = "accuracy_type", values_to = "accuracy")

acc_plot <- df %>%
  filter(str_detect(accuracy_type, "accuracy")) %>%
  mutate(accuracy_type = ifelse(accuracy_type == "train_accuracy", "Train Accuracy", "Validation Accuracy")) %>%
  ggplot(aes(epoch, accuracy)) +
  geom_path(aes(color = accuracy_type)) +
  theme_bw() +
  facet_grid(rows = vars(dropout), cols = vars(gene)) +
  guides(color = guide_legend(title = "")) +
  theme(legend.position = "top") +
  labs(x = "Epoch", y = "Accuracy")
print(acc_plot)

auc_plot <- df %>%
  filter(str_detect(accuracy_type, "auc")) %>%
  mutate(accuracy_type = ifelse(accuracy_type == "train_auc", "Train AUC-ROC", "Validation AUC-ROC")) %>%
  ggplot(aes(epoch, accuracy)) +
  geom_path(aes(color = accuracy_type)) +
  theme_bw() +
  facet_grid(rows = vars(dropout), cols = vars(gene)) +
  guides(color = guide_legend(title = "")) +
  theme(legend.position = "top") +
  labs(x = "Epoch", y = "AUC-ROC")
print(auc_plot)







