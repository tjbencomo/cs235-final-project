# Generate ROC curves

library(readr)
library(dplyr)
library(ggplot2)

data_dir <- "~/code/cs235-data/figures"
genes <- c("ER", "HER2", "PR")
dropout <- c("small", "medium", "high")

aucs <- list()
i <- 1
for (g in genes) {
  for (d in dropout) {
    x <- read_csv(file.path(data_dir, paste(g, "dropout", d, "classifier", "test", "roc.csv", sep="_")))
    x$gene <- g
    x$dropout <- d
    aucs[[i]] <- x
    i <- i + 1
  }
}
df <- bind_rows(aucs)
roc <- df %>%
  ggplot(aes(fpr, tpr)) +
  geom_path(aes(color = dropout)) +
  geom_abline(slope = 1, linetype = "dashed", color = "red") +
  theme_bw() +
  facet_wrap(~gene) +
  guides(color = guide_legend(title = "")) +
  theme(legend.position = "top") +
  labs(x = "False Positive Rate", y = "True Positive Rate")
print(roc)

