library(dplyr)

geno <- data.table::fread("output/geno_ok.csv", data.table = F)
pheno <- data.table::fread("data/Training_data/1_Training_Trait_Data_2014_2023.csv", data.table = F)
pheno <- pheno[pheno$Hybrid %in% geno$name, ]
train <- pheno |>
  filter(Year %in% 2022) |> # we could use more years here
  group_by(Env, Hybrid) |>
  summarize(Yield_Mg_ha = mean(Yield_Mg_ha)) |> # take unadjusted means (this is not BLUEs!)
  filter(!is.na(Yield_Mg_ha)) |>
  mutate(dataset = "train")
val <- pheno |>
  filter(Year %in% 2023) |>
  group_by(Env, Hybrid) |>
  summarize(Yield_Mg_ha = mean(Yield_Mg_ha)) |>
  filter(!is.na(Yield_Mg_ha)) |>
  mutate(dataset = "val")
test <- read.csv("data/Testing_data/1_Submission_Template_2024.csv") |>
  mutate(dataset = "test") |>
  mutate(Yield_Mg_ha = 0)

# intersections
length(unique(pheno$Env))
length(unique(train$Env))
length(unique(val$Env))
length(unique(test$Env))
length(unique(pheno$Hybrid))
length(unique(train$Hybrid))
length(unique(val$Hybrid))
length(unique(test$Hybrid))
length(intersect(pheno$Hybrid, test$Hybrid))  # only 104 (10%) of Hybrids in 2014-2023 are present in 2024
length(intersect(train$Hybrid, test$Hybrid))
length(intersect(val$Hybrid, test$Hybrid))

# unique_envs <- c(unique(train$Env), unique(train$Env), unique(test$Env))
tab <- rbind(train, val, test)
tab$Hybrid <- as.factor(tab$Hybrid)
write.csv(tab, "output/train_val_test.csv", row.names = F)

# prepare for MET GBLUP
# https://igorkf.github.io/posts/2024-01-15-gblup-bglr-asreml/
# Z <- as.matrix(model.matrix(lm(Yield_Mg_ha ~ Hybrid - 1, data = tab)))
# colnames(Z) <- gsub("Hybrid", "", colnames(Z))
# dim(Z)
# Z[1:5, 1:6]
# load("output/G.RData")
# G <- G + diag(nrow(G)) * 1e-6
# G <- G[rownames(G) %in% colnames(Z), colnames(G) %in% colnames(Z)]
# dim(G)
# ZGZt <- Z %*% tcrossprod(G, Z)
# dim(ZGZt)
# EVD_ZGZt <- eigen(ZGZt)
# save(EVD_ZGZt, file = "output/EVD_ZGZt.RData")
