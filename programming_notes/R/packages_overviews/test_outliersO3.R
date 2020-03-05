# OutliersO3 for outlier detection
# source: https://mran.microsoft.com/package/OutliersO3

# Setup -----------------------------------------------------------------------
install.packages(c("OutliersO3", "mbgraphic"))
library(OutliersO3)
library(mbgraphic)
library(gridExtra)

data(Election2005, package = "mbgraphic")
data <- Election2005[, c(6, 10, 17, 28)]


# Basic plot ------------------------------------------------------------------
# An O3 plot using the HDoutliers algorithm
#   - each row in the grey block on the left shows the combination of variables, 
#     according to which outliers are identified in this row in the red block
#   - columns right from the separator (red block) show cases (row numbres) 
#     identified as outliers by one variable or a combination thereof
O3s <- O3prep(data, method = "HDo", tols = 0.05, boxplotLimits = 6)
O3s1 <- O3plotT(O3s)
O3s1$gO3


# Comparing different tolerance levels ----------------------------------------
O3x <- O3prep(data, method = "HDo", tols = c(0.1, 0.05, 0.01), boxplotLimits = c(3, 6, 10))
O3x1 <- O3plotT(O3x)
grid.arrange(O3x1$gO3, O3x1$gpcp, ncol = 1)


# Comparing two methods -------------------------------------------------------
O3m <- O3prep(data, method = c("HDo", "PCS"))
O3m1 <- O3plotM(O3m)
grid.arrange(O3m1$gO3, O3m1$gpcp, ncol=1)


# Using six methods simultaneously --------------------------------------------
O3y <- O3prep(data, method = c("HDo", "PCS", "BAC", "adjOut", "DDC", "MCD"), tols = 0.05, boxplotLimits = 6)
O3y1 <- O3plotM(O3y)
cx <- data.frame(outlier_method = names(O3y1$nOut), number_of_outliers = O3y1$nOut)
knitr::kable(cx, row.names = FALSE)
grid.arrange(O3y1$gO3, O3y1$gpcp, ncol=1)


# Using individual tolerance levels for different methods ---------------------
# in this case this custimisation is far worse than default - too many outliers deteced
# some methods take alpha, some 1-alpha as tolerance level
O3d <- O3prep(data, method = c("HDo", "PCS", "BAC", "adjOut", "DDC", "MCD"), 
              tolHDo = 0.05, tolPCS = 0.5, tolBAC = 0.95, toladj = 0.25, tolDDC = 0.01, tolMCD = 0.5)
O3d1 <- O3plotM(O3d)
O3d1$nOut
O3d1$gO3
