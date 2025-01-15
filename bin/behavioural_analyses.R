library(lme4)
library(tidyverse)
library(broom.mixed)

project_root <- Sys.getenv("NATURALCOGSCI_ROOT")


# category learning

category_df <- read_csv(
  file.path(
    project_root,
    "data",
    "human_behavioural",
    "category_learning",
    "above_chance.csv"
  )
)

category_df <- category_df %>% mutate(trial = scale(trial))
category_df <- category_df %>% mutate(dimension = as.factor(dimension))

category_glmm <- glmer(
  correct ~ 1 + trial + (1 + trial + dimension | participant),
  data = category_df,
  family = binomial
)

category_glmm_df <- tidy(category_glmm)[1:2, ]

# write tibble as csv

write_csv(category_glmm_df, file.path(
  project_root,
  "data",
  "human_behavioural",
  "category_learning",
  "glmm.csv"
))



# reward_learning

reward_df <- read_csv(file.path(
  project_root,
  "data",
  "human_behavioural",
  "reward_learning",
  "above_chance.csv"
))

reward_df <- reward_df %>% mutate(trial = scale(trial))
reward_df <- reward_df %>% mutate(dimension = as.factor(dimension))
reward_df <- reward_df %>% mutate(rl_diff = scale(right_reward - left_reward))

reward_glmm <- glmer(
  choice ~ -1 + trial * rl_diff + 
    (-1 + trial + dimension + rl_diff | participant),
  data = reward_df,
  family = binomial
)

reward_glmm_df <- tidy(reward_glmm)[1:3, ]
write_csv(reward_glmm_df, file.path(
  project_root,
  "data",
  "human_behavioural",
  "reward_learning",
  "glmm.csv"
))
