# rm(list=ls()) 
# setwd('C:/Users/User/Documents/Programming/R/R_code/MSU_BNA')

library(corrr)
library(purrr)
library(tidyr)
library(dplyr)
library(RSQLite)
library(data.table)
library(FactoMineR)
library(ggplot2)
library(readr)

# loading data
data <- fread('Data/Out_data.csv')

govrate <- select(data, Country, year, `GovB-Rate`) %>% spread(Country, `GovB-Rate`)
govrate <- data.frame(govrate)

# deleting nas
(nas <- sapply(govrate, function(x) mean(is.na(x))))

govrate <- govrate[ , !(names(govrate) %in% names(nas[nas > 0.45]))]

dim(govrate[complete.cases(govrate), -1])
govrate[complete.cases(govrate), -1] %>% correlate() %>% network_plot(min_cor = .2, legend = TRUE)

adj_matrix <- correlate(govrate[, -1])

write_csv(adj_matrix, 'Data/Simple_adj_matrix.csv')
