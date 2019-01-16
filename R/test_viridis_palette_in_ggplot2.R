library(ggplot2)
library(dplyr)

# There are two scales for numeric inputs: 
# discrete (scale_color_viridis_d) and continous (scale_color_viridis_c).
ggplot(iris) +
  aes(Sepal.Length, Sepal.Width, color = Species) +
  geom_point() +
  scale_color_viridis_d()

# fill
ggplot(faithfuld) +
  aes(waiting, eruptions, fill = density) + 
  geom_tile() + 
  scale_fill_viridis_c()

# viridis is the new default palette for ordered factors
diamonds %>%
  dplyr::count(cut, color) %>%
  ggplot() +
    aes(cut, n, fill = color) +
    geom_col()

# There are 4 other available colormaps: 
# "magma" (or "A"), 
# "inferno" (or "B"), 
# "plasma" (or "C"), 
# "cividis" (or "E")

ggplot(mtcars) + 
  aes(mpg, disp, color = cyl) + 
  geom_point() + 
  scale_color_viridis_c(option = "A")

ggplot(mtcars) + 
  aes(mpg, disp, color = cyl) + 
  geom_point() + 
  scale_color_viridis_c(option = "B")

ggplot(mtcars) + 
  aes(mpg, disp, color = cyl) + 
  geom_point() + 
  scale_color_viridis_c(option = "C")

ggplot(mtcars) + 
  aes(mpg, disp, color = cyl) + 
  geom_point() + 
  scale_color_viridis_c(option = "E")
