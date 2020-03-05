data(mtcars) # some vars need to be factors
data(Vocab, package = 'car')
data(msleep)
load('C:/Users/Asus/Documents/Knowledge Base/R/ggplot2 in R/recess.RData')
mamsleep <- read.csv('C:/Users/Asus/Documents/Knowledge Base/R/ggplot2 in R/mamsleep.csv', sep = ';')
library(ggplot2)
library(RColorBrewer)


# Aesthetics, Geometry and Data layers ----------------------------------------

# Scatterplots: dealing with overplotting 
ggplot(Vocab, aes(education, vocabulary)) + geom_point()
ggplot(Vocab, aes(education, vocabulary)) + geom_jitter(alpha = 0.2, shape = 1)

# Bar plots: overlapping bars 
posn_d <- position_dodge(width = 0.2)
ggplot(mtcars, aes(x = as.factor(cyl), fill = as.factor(am))) + 
         geom_bar(position = posn_d, alpha = 0.6)

# Overlapping histograms 
ggplot(mtcars, aes(mpg, col = as.factor(cyl))) +
  geom_freqpoly(binwidth = 1, position = 'identity')

ggplot(mtcars, aes(mpg, fill = as.factor(cyl))) +
  geom_histogram(binwidth = 1, position = 'identity', alpha = 0.4)

# Bar plots with color ramp 
ggplot(mtcars, aes(x = as.factor(cyl), fill = as.factor(am))) +
  geom_bar() +
  scale_fill_brewer(palette = "Set1")

blues <- brewer.pal(9, "Blues") # from the RColorBrewer package
blue_range <- colorRampPalette(blues)
ggplot(Vocab, aes(x = education, fill = as.factor(vocabulary))) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = blue_range(11))

# Line plots: adding background region 
ggplot(economics, aes(x = date, y = unemploy/pop)) +
  geom_rect(data = recess,
            aes(xmin = begin, xmax = end, ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE, fill = "red", alpha = 0.2) +
  geom_line()


# Statistics layer ------------------------------------------------------------

# Regression lines 
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  geom_smooth(se = F, span = 0.7)

myColors <- c(brewer.pal(3, "Dark2"), "black")
ggplot(mtcars, aes(x = wt, y = mpg, col = factor(cyl))) +
  geom_point() +
  stat_smooth(method = "lm", se = F) +
  stat_smooth(method = "loess",
              aes(group = 1, col = 'All'),
              se = F, span = 0.7) + 
  scale_color_manual("Cylinders", values = myColors)

ggplot(Vocab, aes(x = education, y = vocabulary, col = year, group = factor(year))) +
  stat_smooth(method = "lm", se = F, alpha = 0.6, size = 2) +
  scale_color_gradientn(colors = brewer.pal(9,"YlOrRd"))

# Summing counts 
ggplot(Vocab, aes(x = education, y = vocabulary)) +
  stat_sum() + 
  scale_size(range = c(1,8)) +
  stat_smooth(method = "loess", aes(col = "x"), se = F) +
  stat_smooth(method = "lm", aes(col = "y"), se = F) +
  scale_color_discrete("Model", labels = c("x" = "LOESS", "y" = "lm"))

# Stats outside geoms: plotting location and scatter 
mtcars$cyl <- factor(mtcars$cyl)
mtcars$am  <- factor(mtcars$am)
posn.d  <- position_dodge(width = 0.1)
posn.jd <- position_jitterdodge(jitter.width = 0.1, dodge.width = 0.2)
posn.j  <- position_jitter(width = 0.2)
wt.cyl.am <- ggplot(mtcars, aes(cyl, wt, col = am, fill = am, group = am))

wt.cyl.am +
  geom_point(position = posn.jd, alpha = 0.6)

wt.cyl.am +
  stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), position = posn.d) # 1 sd

wt.cyl.am +
  stat_summary(fun.data = mean_cl_normal, position = posn.d) # 95% CI

wt.cyl.am +
  stat_summary(geom = "point", fun.y = mean,
               position = posn.d) +
  stat_summary(geom = "errorbar", fun.data = mean_sdl,
               position = posn.d, fun.args = list(mult = 1), width = 0.1)

# Total range
gg_range <- function(x) {
  data.frame(ymin = min(x), # Min
             ymax = max(x)) # Max
}
# Inter-qunartile range
med_IQR <- function(x) {
  data.frame(y = median(x), # Median
             ymin = quantile(x)[2], # 1st quartile
             ymax = quantile(x)[4])  # 3rd quartile
}
wt.cyl.am +
  stat_summary(geom = "linerange", fun.data = med_IQR,
               position = posn.d, size = 3) +
  stat_summary(geom = "linerange", fun.data = gg_range,
               position = posn.d, size = 3,
               alpha = 0.4) +
                 stat_summary(geom = "point", fun.y = median,
                              position = posn.d, size = 3,
                              col = "black", shape = "X")


# Coordinates and Facets layers -----------------------------------------------

# Zooming in the plot 
(p <- ggplot(mtcars, aes(x = wt, y = hp, col = factor(am))) + geom_point() + geom_smooth())
p + coord_cartesian(xlim = c(3,6))

# Aspect ratio 
(base.plot <- ggplot(iris, aes(Sepal.Length, Sepal.Width, col = Species)) +
  geom_jitter() +
  geom_smooth(method = "lm", se = F))
base.plot + coord_equal() # 1:1 ratio

# Pie charts 
(p <- ggplot(mtcars, aes(1, fill = factor(cyl))) + geom_bar())
p + coord_polar(theta = "y")

# Separate rows and columns 
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  facet_grid(factor(am) ~ factor(cyl))#  Separate rows according to am and columns according to cyl

# Many variables in one plot
mtcars$cyl_am <- paste(mtcars$cyl, mtcars$am, sep = "_")
myCol <- rbind(brewer.pal(9, "Blues")[c(3,6,8)],
               brewer.pal(9, "Reds")[c(3,6,8)])

ggplot(mtcars, aes(x = wt, y = mpg, col = cyl_am, size = disp)) +
  geom_point() +
  scale_color_manual(values = myCol) +
  facet_grid(gear ~ vs)

# Dropping unused levels 
ggplot(mamsleep, aes(time, name, col = sleep)) + 
  geom_point() +
  facet_grid(vore ~ .)

ggplot(mamsleep, aes(time, name, col = sleep)) + 
  geom_point() + 
  facet_grid(vore ~ ., scale = "free_y", space = "free_y")


# Themes layer ----------------------------------------------------------------

# Ractangles
(z <- ggplot(mtcars, aes(wt, mpg, col = factor(cyl))) +
  geom_point() +
  facet_grid(. ~ factor(cyl)))

(z <- z + theme(plot.background = element_rect(fill = "#FEE0D2", 
                                         color = "black", 
                                         size = 3)) + 
          theme(panel.background  = element_blank(), 
                legend.key        = element_blank(), 
                legend.background = element_blank(), 
                strip.background  = element_blank()))

# Lines 
(z <- z + theme(panel.grid = element_blank(),
                axis.line  = element_line(color = "black"),
                axis.ticks = element_line(color = "black")))

# Text 
(z <- z + theme(strip.text   = element_text(size = 16, color = "#99000D"), 
                axis.title.y = element_text(color = "#99000D", hjust = 0, face = 'italic'),
                axis.title.x = element_text(color = "#99000D", hjust = 0, face = 'italic'),
                axis.text    = element_text(color = "black")))

# Legend 
z + theme(legend.position = c(0.85, 0.85)) # Move legend by position
z + theme(legend.direction = 'horizontal') # Change direction
z + theme(legend.position = 'bottom') # Change location by name
z <- (z + theme(legend.position = 'none')) # Remove legend entirely

# Positions 
z + theme(panel.spacing.x = unit(2, 'cm')) # Increase spacing between facets
z + theme(panel.spacing.x = unit(2, 'cm'),
          plot.margin = unit(c(0,0,0,0), 'cm')) # Remove any excess plot margin space


# Parallel coordinates plot using GGally --------------------------------------
library(GGally)

# All columns except am
group_by_am <- which(colnames(mtcars) == 'am')
my_names_am <- (1:11)[-group_by_am]

# Basic parallel plot - each variable plotted as a z-score transformation
ggparcoord(mtcars, my_names_am, groupColumn = group_by_am, alpha = 0.8)

# Matrix plot 
GGally::ggpairs(mtcars)


# Heat map --------------------------------------------------------------------
data(barley, package = "lattice")

# Create color palette
myColors <- brewer.pal(9, "Reds")

# Build the heat map from scratch
ggplot(barley, aes(x = year, y = variety, fill = yield)) +
  geom_tile() +
  facet_wrap(~ site, ncol = 1) +
  scale_fill_gradientn(colors = myColors)


# Hear map alternatives -------------------------------------------------------

# Line plots
ggplot(barley, aes(year, yield, col = variety, group = variety)) +
  geom_line() +
  facet_wrap(~ site, nrow = 1)

# Overlapping ribbon plot
ggplot(barley, aes(year, yield, col = site, fill = site, group = site)) +
  stat_summary(fun.y = mean, geom = 'line') +
  stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), geom = 'ribbon', alpha = 0.1, col = NA)


# Multiple Histograms ---------------------------------------------------------
adult <- read.csv('C:/Users/Asus/Documents/Knowledge Base/R/ggplot2 in R/adult.csv', sep = ';')

# The color scale used in the plot
BMI_fill <- scale_fill_brewer("BMI Category", palette = "Reds")

# Theme to fix category display in faceted plot
fix_strips <- theme(strip.text.y = element_text(angle = 0, hjust = 0, vjust = 0.1, size = 14),
                    strip.background = element_blank(),
                    legend.position = "none")

# Histogram, add BMI_fill and customizations
ggplot(adult, aes (x = SRAGE_P, fill= factor(RBMI))) +
  geom_histogram(binwidth = 1) +
  fix_strips +
  BMI_fill +
  facet_grid(RBMI ~ .)

# Plot 1 - Count histogram
ggplot(adult, aes (x = SRAGE_P, fill= factor(RBMI))) +
  geom_histogram(binwidth = 1) +
  BMI_fill

# Plot 2 - Density histogram
ggplot(adult, aes (x = SRAGE_P, fill= factor(RBMI))) +
  geom_histogram(binwidth = 1,  aes(y = ..density..)) +
  BMI_fill

# Plot 3 - Faceted count histogram
ggplot(adult, aes (x = SRAGE_P, fill= factor(RBMI))) +
  geom_histogram(binwidth = 1) +
  BMI_fill +
  facet_grid(RBMI ~ .)

# Plot 4 - Faceted density histogram
ggplot(adult, aes (x = SRAGE_P, fill= factor(RBMI))) +
  geom_histogram(binwidth = 1,  aes(y = ..density..)) +
  BMI_fill +
  facet_grid(RBMI ~ .)

# Plot 5 - Density histogram with position = "fill"
ggplot(adult, aes (x = SRAGE_P, fill= factor(RBMI))) +
  geom_histogram(binwidth = 1,  aes(y = ..density..), position = 'fill') +
  BMI_fill

# Plot 6 - The accurate histogram
ggplot(adult, aes (x = SRAGE_P, fill= factor(RBMI))) +
  geom_histogram(binwidth = 1,  aes(y = ..count../sum(..count..)), position = 'fill') + 
  BMI_fill

# Generic mosaic plot function ------------------------------------------------
library(ggplot2)
library(reshape2)
library(dplyr)
library(ggthemes)

# Script generalized into a function
mosaicGG <- function(data, X, FILL) {
  # Chi-Squared test's residuals tell what is the difference between
  # the observed data and the expected values under the null hypothesis
  # of equal proportions. The higher the residual, the more over-represented
  # the segment is. Low residual indicates an under-represented category.
  # (Category groups observations with a given combination of 2 categorical vars.)
  
  # Proportions in raw data
  DF <- as.data.frame.matrix(table(data[[X]], data[[FILL]]))
  DF$groupSum <- rowSums(DF)
  DF$xmax <- cumsum(DF$groupSum)
  DF$xmin <- DF$xmax - DF$groupSum
  DF$X <- row.names(DF)
  DF$groupSum <- NULL
  DF_melted <- melt(DF, id = c("X", "xmin", "xmax"), variable.name = "FILL")
  DF_melted <- DF_melted %>%
    group_by(X) %>%
    mutate(ymax = cumsum(value/sum(value)),
           ymin = ymax - value/sum(value))
  
  # Chi-sq test
  results <- chisq.test(table(data[[FILL]], data[[X]])) # fill and then x
  resid <- melt(results$residuals)
  names(resid) <- c("FILL", "X", "residual")
  
  # Merge data
  DF_all <- merge(DF_melted, resid)
  
  # Positions for labels
  DF_all$xtext <- DF_all$xmin + (DF_all$xmax - DF_all$xmin)/2
  index <- DF_all$xmax == max(DF_all$xmax)
  DF_all$ytext <- DF_all$ymin[index] + (DF_all$ymax[index] - DF_all$ymin[index])/2
  
  # plot:
  g <- ggplot(DF_all, aes(ymin = ymin,  ymax = ymax, xmin = xmin,
                          xmax = xmax, fill = residual)) +
    geom_rect(col = "white") +
    geom_text(aes(x = xtext, label = X),
              y = 1, size = 3, angle = 90, hjust = 1, show.legend = FALSE) +
    geom_text(aes(x = max(xmax),  y = ytext, label = FILL),
              size = 3, hjust = 1, show.legend = FALSE) +
    scale_fill_gradient2("Residuals") +
    scale_x_continuous("Individuals", expand = c(0,0)) +
    scale_y_continuous("Proportion", expand = c(0,0)) +
    theme_tufte() +
    theme(legend.position = "bottom")
  print(g)
}

# BMI described by age
mosaicGG(adult, 'SRAGE_P', 'RBMI')

# Poverty described by age
mosaicGG(adult, 'POVLL','SRAGE_P')

# mtcars: am described by cyl
mosaicGG(mtcars, 'am', 'cyl')

# Vocab: vocabulary described by education
library(car)
mosaicGG(Vocab, 'vocabulary', 'education')  


# Box plots -------------------------------------------------------------------
library(ggplot2movies)
movies_small <- movies[sample(nrow(movies), 1000), ]
movies_small$rating <- factor(round(movies_small$rating))

d <- ggplot(movies_small, aes(x = rating, y = votes)) +
  geom_point() +
  geom_boxplot() +
  stat_summary(fun.data = "mean_cl_normal",
               geom = "crossbar",
               width = 0.2,
               col = "red")

# Transformation happens before calculating the statistics
d + scale_y_log10()

# Transformation happens after calculating the statistics
d + coord_trans(y = "log10")

# Converting continuous variables to ordinal ones:
# cut_interval(x, n) makes n groups from vector x with equal range.
# cut_number(x, n) makes n groups from vector x with (approximately) equal numbers of observations.
# cut_width(x, width) makes groups of width width from vector x
p <- ggplot(diamonds, aes(x = carat, y = price))
p + geom_boxplot(aes(group = cut_interval(carat, n = 10)))
p + geom_boxplot(aes(group = cut_number(carat, n = 10)))
p + geom_boxplot(aes(group = cut_width(carat, width = 0.25)))

# Box plots with varying width, showing sample size of each group
ggplot(diamonds, aes(x = cut, y = price)) +
  geom_boxplot(varwidth = TRUE)

ggplot(diamonds, aes(x = cut, y = price, col = color)) +
  geom_boxplot(varwidth = TRUE) +
  facet_grid(. ~ color)


# Density plots ---------------------------------------------------------------
test_data <- read.csv('C:/Users/Asus/Documents/Knowledge Base/R/ggplot2 in R/test_data.csv', sep = ';')
test_data2 <- read.csv('C:/Users/Asus/Documents/Knowledge Base/R/ggplot2 in R/test_data2.csv', sep = ';')
mammals <- read.csv('C:/Users/Asus/Documents/Knowledge Base/R/ggplot2 in R/mammals.csv', sep = ';')
small_data <- data.frame(x = c(-3.5, 0, 0.5, 6))

d <- density(test_data$norm)
mode <- d$x[which.max(d$y)]

ggplot(test_data, aes(x = norm)) +
  geom_rug() +
  geom_density() +
  geom_vline(xintercept = mode, col = "red")

ggplot(test_data, aes(x = norm)) +
  geom_histogram(aes(y = ..density..)) +
  geom_density(col = "red") +
  stat_function(fun = dnorm, 
                args = list(mean = mean(test_data$norm), sd = sd(test_data$norm)), 
                col = "blue")

get_bw <- density(small_data$x)$bw
p <- ggplot(small_data, aes(x = x)) +
  geom_rug() +
  coord_cartesian(ylim = c(0,0.5))
p + geom_density()
p + geom_density(adjust = 0.25)
p + geom_density(bw = 0.25 * get_bw)
p + geom_density(kernel = "r")
p + geom_density(kernel = "e")


# Multiple density plots ------------------------------------------------------
ggplot(test_data2, aes(x = value, fill = dist, col = dist)) +
  geom_density(alpha = 0.6) +
  geom_rug(alpha = 0.6)

# Individual densities facetted
ggplot(mammals, aes(x = sleep_total, fill = vore)) +
  geom_density(col = NA, alpha = 0.35) +
  scale_x_continuous(limits = c(0, 24)) +
  coord_cartesian(ylim = c(0, 0.3)) +
  facet_wrap( ~ vore, nrow = 2)

# Combined in one plot
ggplot(mammals, aes(x = sleep_total, fill = vore)) +
  geom_density(col = NA, alpha = 0.35) +
  scale_x_continuous(limits = c(0, 24)) +
  coord_cartesian(ylim = c(0, 0.3))

# Trim densities so that the range does not exceed the true data range
# (areas under curves are not equal to 1 anymore)
ggplot(mammals, aes(x = sleep_total, fill = vore)) +
  geom_density(col = NA, alpha = 0.35, trim = TRUE) +
  scale_x_continuous(limits=c(0,24)) +
  coord_cartesian(ylim = c(0, 0.3))

# Violin plot
ggplot(mammals, aes(x = vore, y = sleep_total, fill = vore)) +
  geom_violin()


# Weighted density plots ------------------------------------------------------
# Each density plot is adjusted according to what proportion of the total data 
# set each sub-group represents
mammals2 <- mammals %>%
  group_by(vore) %>%
  mutate(n = n() / nrow(mammals)) -> mammals

# Weighted density plot
ggplot(mammals2, aes(x = sleep_total, fill = vore)) +
  geom_density(aes(weight = n), col = NA, alpha = 0.35) +
  scale_x_continuous(limits = c(0, 24)) +
  coord_cartesian(ylim = c(0, 1.5))

# Weighted violin plot
ggplot(mammals2, aes(x = vore, y = sleep_total, fill = vore)) +
  geom_violin(aes(weight = n), col = NA)


# 2D density plots ------------------------------------------------------------
data("faithful")
library(viridisLite)
library(viridis)

p <- ggplot(faithful, aes(x = waiting, y = eruptions)) +
  scale_y_continuous(limits = c(1, 5.5), expand = c(0, 0)) +
  scale_x_continuous(limits = c(40, 100), expand = c(0, 0)) +
  coord_fixed(60 / 4.5)
p + geom_density_2d()
p + stat_density_2d(aes(col = ..level..), h = c(5, 0.5)) # h defines bandwidths

ggplot(faithful, aes(x = waiting, y = eruptions)) +
  scale_y_continuous(limits = c(1, 5.5), expand = c(0,0)) +
  scale_x_continuous(limits = c(40, 100), expand = c(0,0)) +
  coord_fixed(60/4.5) +
  stat_density_2d(geom = "tile", aes(fill = ..density..), h=c(5,.5), contour = FALSE) +
  scale_fill_viridis()


# Pair plots and correlation matrices -----------------------------------------
pairs(iris[1:4]) # it only accepts continuous vars

library(PerformanceAnalytics)
chart.Correlation(iris[1:4]) # it only accepts continuous vars

mtcars_fact <- mtcars
mtcars_fact$vs <- factor(mtcars_fact$vs)
mtcars_fact$am <- factor(mtcars_fact$am)
mtcars_fact$gear <- factor(mtcars_fact$gear)
mtcars_fact$carb <- factor(mtcars_fact$carb)
mtcars_fact$cyl <- factor(mtcars_fact$cyl)

library(GGally)
ggpairs(mtcars_fact[1:3]) # handles all variable types

library(reshape2)
cor_list <- function(x) {
  L <- M <- cor(x)
  
  M[lower.tri(M, diag = TRUE)] <- NA
  M <- melt(M)
  names(M)[3] <- "points"
  
  L[upper.tri(L, diag = TRUE)] <- NA
  L <- melt(L)
  names(L)[3] <- "labels"
  
  merge(M, L)
} 

xx <- iris %>%
  group_by(Species) %>%
  do(cor_list(.[1:4])) 

ggplot(xx, aes(x = Var1, y = Var2)) +
  geom_point(aes(col = points, size = abs(points)), shape = 16) +
  geom_text(aes(col = labels,  size = abs(labels), label = round(labels, 2))) +
  scale_size(range = c(0, 6)) +
  scale_color_gradient2("r", limits = c(-1, 1)) +
  scale_y_discrete("", limits = rev(levels(xx$Var1))) +
  scale_x_discrete("") +
  guides(size = FALSE) +
  geom_abline(slope = -1, intercept = nlevels(xx$Var1) + 1) +
  coord_fixed() +
  facet_grid(. ~ Species) +
  theme(axis.text.y = element_text(angle = 45, hjust = 1),
        axis.text.x = element_text(angle = 45, hjust = 1),
        strip.background = element_blank())


# Ternary plots ---------------------------------------------------------------
# (triangle plots for comopsitional, trivariate data)
africa <- read.csv('C:/Users/Asus/Documents/Knowledge Base/R/ggplot2 in R/africa.csv', sep = ';')
library(ggtern)

# Scatter plot
ggtern(africa, aes(x = Sand, y = Silt, z = Clay)) +
  geom_point(shape = 16, alpha = 0.2)

# Contour density plot
ggtern(africa, aes(x = Sand, y = Silt, z = Clay)) +
  geom_density_tern()

# Filled density plot
ggtern(africa, aes(x = Sand, y = Silt, z = Clay)) +
  stat_density_tern(geom = "polygon", aes(fill = ..level.., alpha = ..level..)) +
  guides(fill = FALSE)


# Network plots ---------------------------------------------------------------
library(geomnet)
data(madmen)

mmnet <- merge(madmen$edges, madmen$vertices,
               by.x = "Name1", by.y = "label",
               all = TRUE)

ggplot(data = mmnet, aes(from_id = Name1, to_id = Name2)) +
  geom_net(aes(col = Gender), 
           size = 6, 
           linewidth = 1, 
           labelon = TRUE, 
           fontsize = 3, 
           labelcolour = "black")

ggplot(data = mmnet, aes(from_id = Name1, to_id = Name2)) +
  geom_net(aes(col = Gender),
           size = 6,
           linewidth = 1,
           labelon = TRUE,
           fontsize = 3,
           labelcolour = "black",
           directed = TRUE) +
  scale_color_manual(values = c("#FF69B4", "#0099ff")) +
  xlim(c(-0.05, 1.05)) +
  ggmap::theme_nothing(legend = TRUE) +
  theme(legend.key = element_blank())


# Diagnostics plots -----------------------------------------------------------
data(trees)

# Autoplot on linear models
res <- lm(Volume ~ Girth, data = trees)
plot(res)

install_version("ggfortify", version = "0.4.1", repos = "http://cran.us.r-project.org")
library(ggfortify)
autoplot(res, ncol = 2)

# Time series
data(Canada, package = 'vars')
plot(Canada)
autoplot(Canada)

# Distance matrices and Multi-Dimensional Scaling
data(eurodist)

autoplot(eurodist) + 
  coord_fixed()

autoplot(cmdscale(eurodist, eig = TRUE), 
         label = TRUE, 
         label.size = 3, 
         size = 0)

# K-means clustering
iris_k <- kmeans(iris[-5], 3)
autoplot(iris_k, data = iris, frame = TRUE, shape = "Species")


# Maps ------------------------------------------------------------------------
library(maps)
library(ggmap)
library(ggthemes)
library(viridis)

nz <- map_data("nz")
ggplot(nz, aes(x = long, y = lat, group = group)) +
  geom_polygon() +
  coord_map() +
  theme_nothing()

usa <- map_data("usa")
cities <- read.csv('C:/Users/Asus/Documents/Knowledge Base/R/ggplot2 in R/cities.csv', sep = ';')

# Draw points of varying sizes, relative to the estimated population
ggplot(usa, aes(x = long, y = lat, group = group)) +
  geom_polygon() +
  geom_point(data = cities, aes(group = State, size = Pop_est),
             col = "red", shape = 16, alpha = 0.6) +
  coord_map() +
  theme_map()

# Use color instead of size, and in this case a nice trick is to order the data frame, 
# so that the largest cities are drawn on top of the smaller cities. This is so that they 
# will stand out against the background, which is particularly effective when using the 
# viridis color palette.
cities_arr <- arrange(cities, Pop_est)
ggplot(usa, aes(x = long, y = lat, group = group)) +
  geom_polygon(fill = "grey90") +
  geom_point(data = cities_arr, aes(group = State, col = Pop_est), 
             shape = 16, alpha = 0.6, size = 2) +
  coord_map() +
  theme_map() + 
  scale_color_viridis()


# Choropleths -----------------------------------------------------------------
pop <- read.csv('C:/Users/Asus/Documents/Knowledge Base/R/ggplot2 in R/pop.csv', sep = ';')
state <- map_data("state")

# Map of states
ggplot(state, aes(x = long, y = lat, fill = region, group = group)) +
  geom_polygon(col = "white") +
  coord_map() +
  theme_nothing()

# Merge state and pop: state2
state2 <- merge(state, pop)

# Map of states with populations
ggplot(state2, aes(x = long, y = lat, fill = Pop_est, group = group)) +
  geom_polygon(col = "white") +
  coord_map() +
  theme_map()


# Map from shapefiles ---------------------------------------------------------
library(rgdal)

# Import shape information: germany
germany <- readOGR(dsn = "shapes", layer = "DEU_adm1")

# Convert the shape object into data frame
bundes <- fortify(germany)

# Plot map of germany
ggplot(bundes, aes(x = long, y = lat, group = group)) +
  geom_polygon(fill = "blue", col = "white") +
  coord_map() +
  theme_nothing()


# Choropleth from shapefiles --------------------------------------------------
unemp <- read.csv('C:/Users/Asus/Documents/Knowledge Base/R/ggplot2 in R/unemp.csv', sep = ';')

# re-add state names to bundes
bundes$state <- factor(as.numeric(bundes$id))
levels(bundes$state) <- germany$NAME_1

# Merge bundes and unemp: bundes_unemp
bundes_unemp <- merge(bundes, unemp)

# Plot German unemployment
ggplot(bundes_unemp, aes(x = long, y = lat, group = group, fill = unemployment)) +
  geom_polygon() +
  coord_map() +
  theme_map()


# Carthographic maps ----------------------------------------------------------
library(ggmap) # google maps access

london_map_13 <- get_map("London, England", zoom = 13)
ggmap(london_map_13)

lond <- get_map("London, England", zoom = 13, maptype = "toner", source = "stamen")
ggmap(lond)

lond1 <- get_map("London, England", zoom = 13, maptype = "hybrid")
ggmap(lond1)

lond2 <- get_map("London, England", zoom = 13, maptype = "watercolor")
ggmap(lond2)

lond3 <- get_map("London, England", zoom = 13, maptype = "satellite")
ggmap(lond3)

# Mapping points onto a cartographic map
london_sites <- c("Tower of London, London",             
                  "Buckingham Palace, London",           
                  "Tower Bridge, London",                
                  "Westminster Abbey, London",           
                  "Queen Elizabeth Olympic Park, London")

xx <- geocode(london_sites)
xx$location <- sub(", London", "", london_sites)

london_ton_13 <- get_map(location = "London, England", zoom = 13,
                         source = "stamen", maptype = "toner")

# Olympic Parc missing :(
ggmap(london_ton_13) +
  geom_point(data = xx, aes(col = location), size = 6)

# Instead of defining your cartographic map based on a general location, 
# you can define a bounding box around specific coordinates.

xx <- geocode(london_sites)
xx$location <- sub(", London", "", london_sites)
xx$location[5] <- "Queen Elizabeth\nOlympic Park"

# Create bounding box
bbox <- make_bbox(lon = xx$lon, lat = xx$lat, f = 0.3)

# Re-run get_map to use bbox
london_ton_13 <- get_map(location = bbox, zoom = 13,
                         source = "stamen", maptype = "toner")

# Map same as previously
ggmap(london_ton_13) +
  geom_point(data = xx, aes(col = location), size = 6)

# Map with labels
ggmap(london_ton_13) +
  geom_label(data = xx, aes(label = location), size = 4, 
             fontface = "bold", fill = "grey90", 
             col = "#E41A1C")


# Combine cartographic and choropleth maps ------------------------------------
germany_06 <- get_map(location = "Germany", zoom = 6)

# Plot map and polygon on top:
ggmap(germany_06) +
  geom_polygon(data = bundes,
               aes(x = long, y = lat, group = group),
               fill = NA, col = "red") +
  coord_map()


# Animations ------------------------------------------------------------------
japan <- read.csv('C:/Users/Asus/Documents/Knowledge Base/R/ggplot2 in R/japan.csv', sep = ';')
library(animation)

saveGIF({
  for (i in unique(japan$time)) {
    data <- subset(japan, time == i)
    p <- ggplot(data, aes(x = AGE, y = POP, fill = SEX, width = 1)) +
      coord_flip() +
      geom_bar(data = data[data$SEX == "Female",], stat = "identity") +
      geom_bar(data = data[data$SEX == "Male",], stat = "identity") +
      ggtitle(i)
    print(p)
  }
}, movie.name = "pyramid.gif", interval = 0.1)

# get gganimate package
library(devtools)
library(RCurl)
library(httr)
set_config(config(ssl_verifypeer = 0L))
devtools::install_github("dgrtwo/gganimate")
library(gganimate)

data(Vocab, package = 'car')
p <- ggplot(Vocab, aes(x = education, y = vocabulary,
                       color = year, group = year,
                       frame = year, cumulative = TRUE)) +
  stat_smooth(method = "lm", se = FALSE, size = 3)

gganimate(p, interval = 1.0)


# Grid graphics ---------------------------------------------------------------
library(grid)

# Draw rectangle in null viewport
grid.rect(gp = gpar(fill = "grey90"))

# Write text in null viewport
grid.text("nullviewport")

# Draw a line
grid.lines(x = c(0, 0.75), y = c(0.25, 1),
           gp = gpar(lty = 2, col = "red"))

# Create new viewport: vp
vp <- viewport(x = 0.5, y = 0.5, width = 0.5, height = 0.5, just = "center")

# Push vp
pushViewport(vp)

# Populate new viewport with rectangle
grid.rect(gp = gpar(fill = "blue"))


# Build a plot from scratch with grid -----------------------------------------

# Create plot viewport: pvp
mar <- c(5, 4, 2, 2)
pvp <- plotViewport(mar)
# Push pvp
pushViewport(pvp)
# Add rectangle
grid.rect(gp = gpar(fill = "grey80"))
# Create data viewport: dvp
dvp <- dataViewport(xData = mtcars$wt, yData = mtcars$mpg)
# Push dvp
pushViewport(dvp)
# Add two axes
grid.xaxis()
grid.yaxis()
# Add text to x axis
grid.text("Weight", y = unit(-3, "lines"), name = "xaxis")
# Add text to y axis
grid.text("MPG", x = unit(-3, "lines"), rot = 90, name = "yaxis")
# Add points
grid.points(x = mtcars$wt, y = mtcars$mpg, pch = 16, name = "datapoints")

# Modify the plot with grid.edit
grid.edit("xaxis", label = "Weight (1000 lbs)")
grid.edit("yaxis", label = "Miles/(US) gallon")
grid.edit("datapoints", gp = gpar(col = "#C3212766", cex = 2))


# Grid Graphics in ggplot2 ----------------------------------------------------
library(gtable)
# A simple plot p
p <- ggplot(mtcars, aes(x = wt, y = mpg, col = factor(cyl))) + geom_point()
# Create gtab with ggplotGrob()
gtab <- ggplotGrob(p)
# Extract the grobs from gtab: gtab
g <- gtab$grobs
# Draw only the legend
legend_index <- which(vapply(g, inherits, what = "gtable", logical(1)))
grid.draw(g[[legend_index]])
# Show layout of legend grob
gtable_show_layout(g[[legend_index]])
# Create text grob
my_text <- textGrob(label = "Motor Trend, 1974", gp = gpar(fontsize = 7, col = "gray25"))
# Use gtable_add_grob to modify original gtab
new_legend <- gtable_add_grob(gtab$grobs[[legend_index]], my_text, 3, 2)
# Update in gtab
gtab$grobs[[legend_index]] <- new_legend
# Draw gtab
grid.draw(gtab)



# gridExtra -------------------------------------------------------------------
library(gridExtra)
library(grid) # for unit.c()
# one legend for multiple plots
g1 <- ggplot(mtcars, aes(wt, mpg, col = cyl)) +
  geom_point() +
  theme(legend.position = "bottom")
g2 <- ggplot(mtcars, aes(disp, fill = cyl)) +
  geom_histogram(binwidth = 20) +
  theme(legend.position = "none")
my_legend <- ggplotGrob(g1)$grobs[[15]]  
g1_noleg <- g1 + 
  theme(legend.position = "none")
legend_height <- sum(my_legend$height)
grid.arrange(g1_noleg, g2, my_legend,
             layout_matrix = matrix(c(1, 3, 2, 3), ncol = 2),
             heights = unit.c(unit(1, "npc") - legend_height, legend_height))


# Bag plot --------------------------------------------------------------------
# Bivariate boxplot, 50% of all points withing the inner, darker area
test_data <- read.table("test_data.txt", header = T)
library(aplpack)
bagplot(test_data)
bag <- compute.bagplot(test_data)
points(bag$hull.loop, col = "green", pch = 16)
points(bag$hull.bag, col = "orange", pch = 16)
points(bag$pxy.outlier, col = "purple", pch = 16)


# Writing ggplot extetions ----------------------------------------------------
# ggplot layer for bagplots
# ggproto for StatLoop (hull.loop)
StatLoop <- ggproto("StatLoop", Stat,
                    required_aes = c("x", "y"),
                    compute_group = function(data, scales) {
                      bag <- compute.bagplot(x = data$x, y = data$y)
                      data.frame(x = bag$hull.loop[,1], y = bag$hull.loop[,2])
                    })

# ggproto for StatBag (hull.bag)
StatBag <- ggproto("StatBag", Stat,
                   required_aes = c("x", "y"),
                   compute_group = function(data, scales) {
                     bag <- compute.bagplot(x = data$x, y = data$y)
                     data.frame(x = bag$hull.bag[,1], y = bag$hull.bag[,2])
                   })

# ggproto for StatOut (pxy.outlier)
StatOut <- ggproto("StatOut", Stat,
                   required_aes = c("x", "y"),
                   compute_group = function(data, scales) {
                     bag <- compute.bagplot(x = data$x, y = data$y)
                     data.frame(x = bag$pxy.outlier[,1], y = bag$pxy.outlier[,2])
                   })

# Combine ggproto objects in layers to build stat_bag()
stat_bag <- function(mapping = NULL, data = NULL, geom = "polygon",
                     position = "identity", na.rm = FALSE, show.legend = NA,
                     inherit.aes = TRUE, loop = FALSE, ...) {
  list(
    # StatLoop layer
    layer(
      stat = StatLoop, data = data, mapping = mapping, geom = geom, 
      position = position, show.legend = show.legend, inherit.aes = inherit.aes,
      params = list(na.rm = na.rm, alpha = 0.35, col = NA, ...)
    ),
    # StatBag layer
    layer(
      stat = StatBag, data = data, mapping = mapping, geom = geom, 
      position = position, show.legend = show.legend, inherit.aes = inherit.aes,
      params = list(na.rm = na.rm, alpha = 0.35, col = NA, ...)
    ),
    # StatOut layer
    layer(
      stat = StatOut, data = data, mapping = mapping, geom = "point", 
      position = position, show.legend = show.legend, inherit.aes = inherit.aes,
      params = list(na.rm = na.rm, alpha = 0.7, col = NA, shape = 21, ...)
    )
  )
}

# Check the plot
ggplot(test_data, aes(x = x, y = y)) +
  stat_bag(fill = 'black')
test_data2 <- mutate(test_data, treatment = c(rep("A", 30), rep("B", 30)))
ggplot(test_data2, aes(x = x, y = y, fill = treatment)) +
  stat_bag()


# Weather plot ----------------------------------------------------------------
# Import weather data
weather <- read.fwf("NYNEWYOR.txt",
                    header = FALSE,
                    col.names = c("month", "day", "year", "temp"),
                    widths = c(14, 14, 13, 4))

past <- weather %>%
  filter(!(month == 2 & day == 29)) %>%
  filter(year != max(year)) %>%
  group_by(year) %>%
  mutate(yearday = 1:length(day)) %>%
  ungroup() %>%
  filter(temp != -99) %>%
  group_by(yearday) %>%
  mutate(max = max(temp),
         min = min(temp),
         avg = mean(temp),
         CI_lower = Hmisc::smean.cl.normal(temp)[2],
         CI_upper = Hmisc::smean.cl.normal(temp)[3]) %>%
  ungroup()

present <- weather %>%
  filter(!(month == 2 & day == 29)) %>%
  filter(year == max(year)) %>%
  group_by(year) %>%
  mutate(yearday = 1:length(day)) %>%
  ungroup() %>%
  filter(temp != -99)

past_extremes <- past %>%
  group_by(yearday) %>%
  summarise(past_low = min(temp),
            past_high = max(temp))

record_high_low <- present %>%
  left_join(past_extremes) %>%
  mutate(record = ifelse(temp < past_low, 
                         "#0000CD",
                         ifelse(temp > past_high, 
                                "#CD2626", 
                                "#00000000")))

draw_pop_legend <- function(x = 0.6, y = 0.2, width = 0.2, height = 0.2, fontsize = 10) {
  pushViewport(viewport(x = x, y = y, width = width, height = height, just = "center"))
  legend_labels <- c("Past record high",
                     "95% CI range",
                     "Current year",
                     "Past years",
                     "Past record low")
  legend_position <- c(0.9, 0.7, 0.5, 0.2, 0.1)
  grid.text(label = legend_labels, x = 0.12, y = legend_position, 
            just = "left", 
            gp = gpar(fontsize = fontsize, col = "grey20"))
  point_position_y <- c(0.1, 0.2, 0.9)
  point_position_x <- rep(0.06, length(point_position_y))
  grid.points(x = point_position_x, y = point_position_y, pch = 16,
              gp = gpar(col = c("#0000CD", "#EED8AE", "#CD2626")))
  grid.rect(x = 0.06, y = 0.5, width = 0.06, height = 0.4,
            gp = gpar(col = NA, fill = "#8B7E66"))
  grid.lines(x = c(0.03, 0.09), y = c(0.5, 0.5),
             gp = gpar(col = "black", lwd = 3))
  popViewport()
}

ggplot(past, aes(x = yearday, y = temp)) + 
  geom_point(col = "#EED8AE", alpha = 0.3, shape = 16) +
  geom_linerange(aes(ymin = CI_lower, ymax = CI_upper), col = "#8B7E66") +
  geom_line(data = present) +
  geom_point(data = record_high_low, aes(col = record)) +
  scale_color_identity()
draw_pop_legend()


# Crate own stat layers for the weather plot ----------------------------------
clean_weather <- function(file) {
  weather <- read.fwf(file,
                      header = FALSE,
                      col.names = c("month", "day", "year", "temp"),
                      widths = c(14, 14, 13, 4))
  weather %>%
    filter(!(day == 29 & month == 2)) %>%
    group_by(year) %>%
    mutate(yearday = 1:length(day)) %>%
    ungroup() %>%
    filter(temp != -99)
}

StatHistorical <- ggproto("StatHistorical", Stat,
                          compute_group = function(data, scales, params) {
                            data <- data %>%
                              filter(year != max(year)) %>%
                              group_by(x) %>%
                              mutate(ymin = Hmisc::smean.cl.normal(y)[3],
                                     ymax = Hmisc::smean.cl.normal(y)[2]) %>%
                              ungroup()
                          },
                          required_aes = c("x", "y", "year"))

# Create the layer
stat_historical <- function(mapping = NULL, data = NULL, geom = "point",
                            position = "identity", na.rm = FALSE, show.legend = NA, 
                            inherit.aes = TRUE, ...) {
  list(
    layer(
      stat = "identity", data = data, mapping = mapping, geom = geom,
      position = position, show.legend = show.legend, inherit.aes = inherit.aes,
      params = list(na.rm = na.rm, col = "#EED8AE", alpha = 0.3, shape = 16, ...)
    ),
    layer(
      stat = StatHistorical, data = data, mapping = mapping, geom = "linerange",
      position = position, show.legend = show.legend, inherit.aes = inherit.aes,
      params = list(na.rm = na.rm, col = "#8B7E66", ...)
    )
  )
}

StatPresent <- ggproto("StatPresent", Stat,
                       compute_group = function(data, scales, params) {
                         data <- filter(data, year == max(year))
                       },
                       required_aes = c("x", "y", "year"))

# Create the layer
stat_present <- function(mapping = NULL, data = NULL, geom = "line",
                         position = "identity", na.rm = FALSE, show.legend = NA, 
                         inherit.aes = TRUE, ...) {
  layer(
    stat = StatPresent, data = data, mapping = mapping, geom = geom,
    position = position, show.legend = show.legend, inherit.aes = inherit.aes,
    params = list(na.rm = na.rm, ...)
  )
}

StatExtremes <- ggproto("StatExtremes", Stat,
                        compute_group = function(data, scales, params) {
                          
                          present <- data %>%
                            filter(year == max(year)) 
                          
                          past <- data %>%
                            filter(year != max(year)) 
                          
                          past_extremes <- past %>%
                            group_by(x) %>%
                            summarise(past_low = min(y),
                                      past_high = max(y))
                          
                          # transform data to contain extremes
                          data <- present %>%
                            left_join(past_extremes) %>%
                            mutate(record = ifelse(y < past_low, 
                                                   "#0000CD", 
                                                   ifelse(y > past_high, 
                                                          "#CD2626", 
                                                          "#00000000")))
                        },
                        required_aes = c("x", "y", "year"))

# Create the layer
stat_extremes <- function(mapping = NULL, data = NULL, geom = "point",
                          position = "identity", na.rm = FALSE, show.legend = NA, 
                          inherit.aes = TRUE, ...) {
  layer(
    stat = StatExtremes, data = data, mapping = mapping, geom = geom,
    position = position, show.legend = show.legend, inherit.aes = inherit.aes,
    params = list(na.rm = na.rm, ...)
  )
}

# Build the plot
my_data <- clean_weather("NYNEWYOR.txt")
ggplot(my_data, aes(x = yearday, y = temp, year = year)) +
  stat_historical() +
  stat_present() +
  stat_extremes(aes(col = ..record..)) +
  scale_color_identity()

# File paths of all datasets
my_files <- c("NYNEWYOR.txt","FRPARIS.txt", "ILREYKJV.txt", "UKLONDON.txt")

# Build my_data with a for loop
my_data <- NULL
for (file in my_files) {
  temp <- clean_weather(file)
  temp$id <- sub(".txt", "", file)
  my_data <- rbind(my_data, temp)
}

# Build the final plot, from scratch!
ggplot(my_data, aes(x = yearday, y = temp, year = year)) +
  stat_historical() +
  stat_present() +
  stat_extremes(aes(col = ..record..)) +
  scale_color_identity() +  # specify colour here
  facet_wrap(~id, ncol = 2)
