# we need two packages from Bioconductor
pkgs <- installed.packages()
if (!"BiocManager" %in% pkgs) install.packages("BiocManager")
if (!"SNPRelate" %in% pkgs) BiocManager::install("SNPRelate")
if (!"gdsfmt" %in% pkgs) BiocManager::install("gdsfmt")

library(gdsfmt)
library(SNPRelate)
library(dplyr)
library(ggplot2)

filter_maf <- function(tab, cutoff) {
  ns <- nrow(tab)
  ss <- colSums(tab, na.rm = T)
  freq <- 0.5 * ss / ns
  maf_matrix <- rbind(freq, 1 - freq, deparse.level = 0)
  maf <- apply(maf_matrix, 2, min)
  snps <- names(maf[maf < cutoff])
  return(tab[, !colnames(tab) %in% snps])
}

# read
sub <- read.csv("data/Testing_data/1_Submission_Template_2024.csv")
geno <- data.table::fread("data/Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt", data.table = F)
colnames(geno)[colnames(geno) == "<Marker>"] <- "Hybrid"
# geno <- geno[geno$Hybrid %in% sub$Hybrid, ]
name <- geno[, 1]
geno <- as.matrix(geno[, -1])
rownames(geno) <- name
geno[geno == 0.5] <- -1
geno <- geno + 1
dim(geno)

# missing
perc_missing <- colSums(is.na(geno)) / nrow(geno)
missing_idx <- which(perc_missing > 0.5)
geno <- geno[, -missing_idx]
dim(geno)

# imputation
geno <- apply(geno, 2, function(x) {
  x[which(is.na(x))] <- mean(x, na.rm = T)
  return(x)
})
geno <- round(geno)
dim(geno)

# MAF filtering
geno <- filter_maf(geno, cutoff = 0.05)
dim(geno)

# LD pruning
thr <- 0.95
chr <- as.integer(gsub("S", "", gsub("_.*", "", colnames(geno))))
pos <- as.integer(gsub(".*_", "", colnames(geno)))
geno_meta <- data.frame(snp = colnames(geno), allele = NA, chr = chr, pos = pos, cm = NA)
out_gds <- "output/geno.gds"
showfile.gds(closeall = T, verbose = F)
snpgdsCreateGeno(out_gds, genmat = geno, snpfirstdim = F, sample.id = rownames(geno),
                 snp.id = colnames(geno), snp.chromosome = chr, snp.position = pos)
genofile <- snpgdsOpen(out_gds)
set.seed(1)
sel_pruned <- snpgdsLDpruning(genofile, ld.threshold = thr, start.pos = "first", method = "corr", verbose = T)
sel_pruned <- unname(unlist(sel_pruned))
geno <- geno[, sel_pruned]
dim(geno)
prop.table(table(geno))

# write
geno <- cbind(data.frame(name = rownames(geno)), geno)
data.table::fwrite(geno, "output/geno_ok.csv", row.names = F)
write.csv(geno_meta[geno_meta$snp %in% colnames(geno), ], "output/geno_meta_ok.csv", row.names = F)
# G <- AGHmatrix::Gmatrix(as.matrix(geno[, -1]), maf = 0)
# rownames(G) <- colnames(G) <- geno[, 1]
# save(G, file = "output/G.RData")
# data.table::fwrite(G, "output/G.csv", row.names = F)

# check genotypic frequency
# table(as.vector(geno))
# prop.table(table(as.vector(geno)))

############
# SOME EDA
############

# population structure
# maybe remove those points to the far right?
pc <- princomp(geno[, -1])
var_exp <- scales::percent(pc$sdev ^ 2 / sum(pc$sdev ^ 2), accuracy = 0.1)
tab_pc <- data.frame(PC1 = pc$scores[, 1], PC2 = pc$scores[, 2], hybrid = rownames(geno))
tab_pc$tester <- as.factor(gsub(".*\\/", "", tab_pc$hybrid))
tab_pc$dataset <- "all"
tab_pc$dataset <- as.factor(ifelse(geno$name %in% sub$Hybrid, "2024", "all"))
ggplot(tab_pc, aes(x = PC1, y = PC2, color = dataset)) +
  geom_point(alpha = 0.7) +
  labs(x = paste0("PC1 (", var_exp[1], ")"), y = paste0("PC2 (", var_exp[2], ")")) +
  theme_bw()

# distant hybrids from training population
tab_pc_far <- droplevels(tab_pc[tab_pc$PC1 >= 20.5, ])
hybrids_remove <- tab_pc_far$hybrid
table(tab_pc_far$tester)  # DK3IIH6 (only 1) and PHT69 (545)
sub[sub$Hybrid %in% hybrids_remove, ]  # not in test anyway

tab_pc_new <- droplevels(tab_pc[!tab_pc$hybrid %in% hybrids_remove, ])
ggplot(tab_pc_new, aes(x = PC1, y = PC2, color = dataset)) +
  geom_point(alpha = 0.7) +
  labs(x = paste0("PC1 (", var_exp[1], ")"), y = paste0("PC2 (", var_exp[2], ")")) +
  theme_bw()

# molecular density
barplot(table(chr))
