# Dendrograms are plots used to present results of hierarchical clustering. 
# Here we present 7 ways to produce these plots in R.

# prepare hierarchical cluster 
hc <- hclust(dist(mtcars)) 


# A very simple dendrogram ---------------------------------------------------- 
# basic plot
plot(hc) 
# labels at the same level 
plot(hc, hang = -1)


# Tweeking some parameters for plotting a dendrogram --------------------------
# set background color 
par(bg ="#DDE3CA") 
# plot dendrogram 
plot(hc, col = "#487AA1", col.main = "#45ADA8", col.lab = "#7C8071", 
     col.axis = "#F38630", lwd = 3, lty = 3, sub = '', hang = -1, axes = FALSE) 
# add axis 
axis(side = 2, at = seq(0, 400, 100), col = "#F38630", 
     labels = FALSE, lwd = 2) 
# add text in margin 
mtext(seq(0, 400, 100), side = 2, at = seq(0, 400, 100), 
      line = 1, col = "#A38630", las = 2) 


# Alternative dendrogram ------------------------------------------------------
# An alternative way to produce dendrograms is to specifically convert hclust 
# objects intodendrograms objects. 
hcd = as.dendrogram(hc) 
# alternative way to get a dendrogram 
plot(hcd) 
# triangular dendrogram 
plot(hcd, type="triangle") 


# Zooming-in on dendrograms ---------------------------------------------------
# Another very useful option is the ability to inspect selected parts of 
# a given tree. For instance, if we wanted to examine the top partitions 
# of the dendrogram, we could cut it at a height of 75. 
plot(cut(hcd, h=75)$upper, 
     main="Upper tree of cut at h=75") 
plot(cut(hcd, h=75)$lower[[2]], 
     main="Second branch of lower tree with cut at h=75") 


# More customizable dendrograms -----------------------------------------------
# In order to get more customized graphics we need a little bit of more code. 
# A very useful resource is the function dendrapply that can be used to apply 
# a function to all nodes of a dendrgoram. This comes very handy if we want
# to add some color to the labels. 

# vector of colors 
labelColors = c("#CDB380", "#036564", "#EB6841", "#EDC951")
# cut dendrogram in 4 clusters
clusMember = cutree(hc, 4)
# function to get color labels
colLab <- function(n) {
  if (is.leaf(n)) {
    a <- attributes(n)
    labCol <- labelColors[clusMember[which(names(clusMember) == a$label)]]
    attr(n, "nodePar") <- c(a$nodePar, lab.col = labCol)
  }
  n
}
# using dendrapply
clusDendro = dendrapply(hcd, colLab)
# make plot
plot(clusDendro, main = "Cool Dendrogram", type = "triangle")

     
# Phylogenetic trees ----------------------------------------------------------
# Closely related to dendrograms, phylogenetic trees are another option to 
# display tree diagrams showing the relationships among observations based 
# upon their similarities. A very nice tool for displaying more appealing 
# trees is provided by the R package ape. In this case, what we need is to 
# convert the hclust objects into phylo objects with the funtions as.phylo. 
# The plot.phylo function has four more different types for plotting 
# a dendrogram. Here they are: 
install.packages("ape") 
library(ape) 
# plot basic tree 
plot(as.phylo(hc), cex=0.9, label.offset=1) 
# cladogram 
plot(as.phylo(hc), type="cladogram", cex=0.9, label.offset=1) 
# unrooted 
plot(as.phylo(hc), type="unrooted") 
# fan 
plot(as.phylo(hc), type="fan") 
# radial 
plot(as.phylo(hc), type="radial") 

# Customizing phylogenetic trees
# vector of colors 
mypal = c("#556270", "#4ECDC4", "#1B676B", "#FF6B6B", "#C44D58") 
# cutting dendrogram in 5 clusters 
clus5 = cutree(hc, 5) 
# plot 
par(bg = "#E8DDCB") 
# Size reflects miles per gallon 
plot(as.phylo(hc), type = "fan", tip.color = mypal[clus5], label.offset = 1, 
     cex = log(mtcars$mpg,10), col = "red") 


# Colour in leaves ------------------------------------------------------------
install.packages('sparcl')
library(sparcl)
# colors the leaves of a dendrogram
y = cutree(hc, 3)
ColorDendrogram(hc, y = y, labels = names(y), main = "My Simulated Data", 
                branchlength = 80)


# Dendrograms with ggdendro ---------------------------------------------------
# ggplot2 has no functions to plot dendrograms. However, the ad-hoc package 
# ggdendro offers a decent solution. You would expect to have more 
# customization options, but so far they are rather limited. Anyway, for those 
# of us who are ggploters this is another tool in our toolkit. 

install.packages("ggdendro") 
library(ggplot2) 
library(ggdendro) 
# basic option 
ggdendrogram(hc, theme_dendro=FALSE) 
# another option 
ggdendrogram(hc, rotate=TRUE, size=4, theme_dendro=FALSE, color="tomato")
# Triangular lines
ddata <- dendro_data(as.dendrogram(hc), type = "triangle")
ggplot(segment(ddata)) + 
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend)) + 
  ylim(-10, 150) + 
  geom_text(data = label(ddata), aes(x = x, y = y, label = label), 
            angle = 90, lineheight = 0)


# Colored dendrogram ----------------------------------------------------------
# Last but not least, there's one more resource available from Romain 
# Francois's addicted to R graph gallery. 
# (http://gallery.r-enthusiasts.com/RGraphGallery.php?graph=79) 
# The code in R for generating colored dendrograms, which you can download and 
# modify if wanted so, is available here:
# http://addictedtor.free.fr/packages/A2R/lastVersion/R/code.R

# load code of A2R function
source("http://addictedtor.free.fr/packages/A2R/lastVersion/R/code.R")
# colored dendrogram
par(bg = "#EFEFEF")
A2Rplot(hc, k = 3, boxes = FALSE, col.up = "gray50", 
        col.down = c("#FF6B6B", "#4ECDC4", "#556270"))

# another colored dendrogram
par(bg = "gray15")
cols = hsv(c(0.2, 0.57, 0.95), 1, 1, 0.8)
A2Rplot(hc, k = 3, boxes = FALSE, col.up = "gray50", col.down = cols)

