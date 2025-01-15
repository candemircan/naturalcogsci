library(lme4)
library(tidyverse)

args <- commandArgs(trailingOnly = TRUE)
feature <- args[1]
task <- args[2]
penalty <- args[3]
transform <- args[4]

project_root <- Sys.getenv("NATURALCOGSCI_ROOT")

feature <- gsub("/", "_", feature)

df_path <- file.path(
  project_root, "data", "learner_behavioural", task,
  sprintf("%s_%s_%s.csv", feature, penalty, transform)
)

if (file.exists(df_path)) {
  q()  # This will exit the R script
}

print(df_path)

temp_df <- read_csv(df_path)
if (grepl("category_learning", df_path, fixed = TRUE)) {
  temp_df <- temp_df %>%
    mutate(diff = right_value)
} else {
  temp_df <- temp_df %>%
    mutate(diff = right_value - left_value)
}

row_no <- nrow(temp_df)
par_no <- temp_df %>%
  select(participant) %>%
  n_distinct()
trial_no <- temp_df %>%
  select(trial) %>%
  n_distinct()

prob <- matrix(nrow = row_no, ncol = 1)

pb <- txtProgressBar(min = 0, max = row_no, style = 3)

for (i in 1:row_no) {
  training <- temp_df %>% slice(-i)
  test <- temp_df %>% slice(i)
  # get the scaling parameters first
  training_mean <- training %>%
    pull(diff) %>%
    mean()
  training_std <- training %>%
    pull(diff) %>%
    sd()

  # scale training
  training <- training %>%
    mutate(scaled_diff = (diff - training_mean) / training_std)

  # scale test
  test <- test %>%
    mutate(scaled_diff = (diff - training_mean) / training_std)


  temp_glmm <- glmer(
    choice ~ -1 + scaled_diff +
      (-1 + scaled_diff | participant),
    family = "binomial", data = training
  )

  prediction <- predict(temp_glmm, newdata = test, type = "response")
  prob[i, 1] <- prediction


  setTxtProgressBar(pb, i)
}

temp_df <- temp_df %>% select(-diff)
temp_df <- temp_df %>% mutate(prob = prob)
close(pb)


write.csv(temp_df, df_path, row.names = FALSE)