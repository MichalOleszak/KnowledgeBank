library(shiny)
library(ggplot2)
library(DT)
library(dplyr)
load(url("http://s3.amazonaws.com/assets.datacamp.com/production/course_4850/datasets/movies.Rdata"))

# Example app -----------------------------------------------------------------
# Define UI for application that plots features of movies
ui <- fluidPage(
  # Sidebar layout with a input and output definitions
  sidebarLayout(
    # Inputs
    sidebarPanel(
      # Select variable for y-axis
      selectInput(inputId = "y", 
                  label = "Y-axis:",
                  choices = c("IMDB rating" = "imdb_rating", 
                              "IMDB number of votes" = "imdb_num_votes", 
                              "Critics score" = "critics_score", 
                              "Audience score" = "audience_score", 
                              "Runtime" = "runtime"), 
                  selected = "audience_score"),
      # Select variable for x-axis
      selectInput(inputId = "x", 
                  label = "X-axis:",
                  choices = c("IMDB rating" = "imdb_rating", 
                              "IMDB number of votes" = "imdb_num_votes", 
                              "Critics score" = "critics_score", 
                              "Audience score" = "audience_score", 
                              "Runtime" = "runtime"), 
                  selected = "critics_score"),
      # Select variable for color
      selectInput(inputId = "z", 
                  label = "Color by:",
                  choices = c("Title type" = "title_type", 
                              "Genre" = "genre", 
                              "MPAA rating" = "mpaa_rating", 
                              "Critics rating" = "critics_rating", 
                              "Audience rating" = "audience_rating"),
                  selected = "mpaa_rating"),
      # Transparency level
      sliderInput(inputId = "alpha", 
                  label = "Alpha:", 
                  min = 0, max = 1, 
                  value = 0.5)
    ),
    # Output
    mainPanel(
      plotOutput(outputId = "scatterplot"),
      plotOutput(outputId = "densityplot", height = 200)
    )
  )
)

# Define server function required to create the scatterplot
server <- function(input, output) {
  # Create the scatterplot object the plotOutput function is expecting
  output$scatterplot <- renderPlot({
    ggplot(data = movies, aes_string(x = input$x, y = input$y,
                                     color = input$z)) +
      geom_point(alpha = input$alpha)
  })
  # Create densityplot
  output$densityplot <- renderPlot({
    ggplot(data = movies, aes_string(x = input$x)) +
      geom_density()
  })
}

# Create a Shiny app object
shinyApp(ui = ui, server = server)



# Example app 2 ---------------------------------------------------------------
n_total <- nrow(movies)
# Define UI for application that plots features of movies
ui <- fluidPage(
  # Sidebar layout with a input and output definitions
  sidebarLayout(
    # Inputs
    sidebarPanel(
      # Text instructions
      HTML(paste("Enter a value between 1 and", n_total)),
      # Numeric input for sample size
      numericInput(inputId = "n",
                   label = "Sample size:",
                   value = 30,
                   step = 1,
                   min = 1,
                   max = n_total)
      
    ),
    # Output: Show data table
    mainPanel(
      DT::dataTableOutput(outputId = "moviestable")
    )
  )
)

# Define server function required to create the scatterplot
server <- function(input, output) {
  # Create data table
  output$moviestable <- DT::renderDataTable({
    req(input$n) # prevents crashing when no input provided
    movies_sample <- movies %>%
      sample_n(input$n) %>%
      select(title:studio)
    DT::datatable(data = movies_sample, 
                  options = list(pageLength = 10), 
                  rownames = FALSE)
  })
}

# Create a Shiny app object
shinyApp(ui = ui, server = server)


# Inputs ----------------------------------------------------------------------
selectInput(inputId = "y", 
            label = "Y-axis:",
            choices = c("IMDB rating" = "imdb_rating", 
                        "IMDB number of votes" = "imdb_num_votes", 
                        "Critics score" = "critics_score", 
                        "Audience score" = "audience_score", 
                        "Runtime" = "runtime"), 
            selected = "audience_score")

sliderInput(inputId = "alpha", 
            label = "Alpha:", 
            min = 0, 
            max = 1, 
            value = 0.5)

numericInput(inputId = "n",
             label = "Sample size:",
             value = 30,
             step = 1,
             min = 1,
             max = n_total)

dateInput(inputId = "date",
          label = "Select date:",
          value = "2013-01-01",
          min = min_date, max = max_date)

dateRangeInput(inputId = "date",
               label = "Select dates:",
               start = "2013-01-01",
               end = "2014-01-01",
               startview = "year",
               min = min_date, max = max_date)


# Render functions ------------------------------------------------------------
renderPlot({
  ggplot(data = movies, aes_string(x = input$x)) +
    geom_density()
})

DT::renderDataTable({
  req(input$n) # prevents crashing when no input provided
  movies_sample <- movies %>%
    sample_n(input$n) %>%
    select(title:studio)
  DT::datatable(data = movies_sample, 
                options = list(pageLength = 10), 
                rownames = FALSE)
})

renderText({
  r <- round(cor(movies[, input$x], movies[, input$y], use = "pairwise"), 3)
  paste0("Correlation = ", r, ". Note: If the relationship between the two variables is not linear, the correlation coefficient will not be meaningful.")
})

renderUI({
  str_x <- movies %>% pull(input$x) %>% mean() %>% round(2)
  paste("Average", input$x, "=", avg_x)
  str_y <- movies %>% pull(input$y) %>% mean() %>% round(2)
  paste("Average", input$y, "=", avg_y)
  HTML(paste(str_x, str_y, sep = '<br/>'))
})


# Brushing& Hovering ----------------------------------------------------------
# Brushing allows to draw rectangles on the plot, and summarise data in this area
# Hovering summarises datapoints over which the mouse coursor hovers
fluidPage(
  br(),
  ...
)
  
mainPanel(
  # eiter:
  plotOutput(outputId = "scatterplot", brush = "plot_brush"),
  # or:
  plotOutput(outputId = "scatterplot", hover = "plot_hover"),
  dataTableOutput(outputId = "moviestable"),
  br()
)

output$scatterplot <- renderPlot({
  ggplot(data = movies, aes_string(x = input$x, y = input$y)) +
    geom_point()
})
# eiter:
output$moviestable <- DT::renderDataTable({
  brushedPoints(movies, brush = input$plot_brush) %>% 
    select(title, audience_score, critics_score)
})
# or:
output$moviestable <- DT::renderDataTable({
  brushedPoints(movies, coordinfo = input$plot_hover) %>% 
    select(title, audience_score, critics_score)
})


# Outputs ---------------------------------------------------------------------
mainPanel(
  plotOutput(outputId = "scatterplot"),
  plotOutput(outputId = "densityplot", height = 200),
  DT::dataTableOutput(outputId = "moviestable"),
  verbatimTextOutput(outputId = "avg_x"),
  TextOutput(outputId = "text"),
  htmlOutput(outputId = "avgs")
)


# Downloading from a Shiny app ------------------------------------------------
ui <- fluidPage(
  sidebarLayout(
    sidebarPanel(
      # Select filetype
      radioButtons(inputId = "filetype",
                   label = "Select filetype:",
                   choices = c("csv", "tsv"),
                   selected = "csv"),
      # Select variables to download
      checkboxGroupInput(inputId = "selected_var",
                         label = "Select variables:",
                         choices = names(movies),
                         selected = c("title"))
      
    ),
    mainPanel(
      HTML("Select filetype and variables, then hit 'Download data'."),
      downloadButton("download_data", "Download data")
    )
  )
)
server <- function(input, output) {
  # Download file
  output$download_data <- downloadHandler(
    filename = function() {
      paste0("movies.", input$filetype)
    },
    content = function(file) { 
      if(input$filetype == "csv"){ 
        write_csv(movies %>% select(input$selected_var), file) 
      }
      if(input$filetype == "tsv"){ 
        write_tsv(movies %>% select(input$selected_var), file) 
      }
    }
  )
  
}
shinyApp(ui = ui, server = server)


# Reactive programming --------------------------------------------------------
# reactive() produces cache objects that only get re-evaluated if their input changes
server <- function(input, output) {
  # Create reactive data frame
  movies_selected <- reactive({
    req(input$selected_var)
    movies %>% select(input$selected_var)
  })
  # Create data table
  output$moviestable <- DT::renderDataTable({
    DT::datatable(data = movies_selected(), # refer to cache objects with ()
                  options = list(pageLength = 10), 
                  rownames = FALSE)
  })
}


# Isolating reactions --------------------------------------------------------
# Used to stop a reaction:
# Plot will NOT update in plot_title changes, but will if any other input changes
output$scatterplot <- renderPlot({
  ggplot(data = movies_subset(), aes_string(x = input$x, y = input$y)) +
    geom_point() +
    labs(title = isolate({input$plot_title}))
})


# Triggering reactions --------------------------------------------------------
# handler called whenever event is invalidated
observeEvent(eventExpr, handlerExpr, ...)
# Example: Write a CSV of the sampled data when action button is pressed
# ui
actionButton(inputId = "write_csv", label = "Write CSV")
# server
observeEvent(input$write_csv, {
  filename <- paste0("movies.csv")
  write_csv(movies_sample(), path = filename)
  }
)


# Delaying reactions ----------------------------------------------------------
# Example: Update plot title only after clicking the button
# ui
actionButton(inputId = "update_plot_title", label = "Update plot title")
# server
new_plot_title <- eventReactive(input$update_plot_title, {
  toTitleCase(input$plot_title)
  }
)
output$scatterplot <- renderPlot({
  ggplot(data = movies, aes_string(x = input$x, y = input$y, color = input$z)) +
    geom_point(alpha = input$alpha, size = input$size) +
    labs(title = new_plot_title()) # Call new title here
})


# Interface builder functions -------------------------------------------------
names(tags)
tags$b("bolded text in html")

# Add image
h5("Built with",
   img(src = "https://www.rstudio.com/wp-content/uploads/2014/04/shiny.png", height = "30px"),
   "by",
   img(src = "https://www.rstudio.com/wp-content/uploads/2014/07/RStudio-Logo-Blue-Gray.png", height = "30px"),
   ".")


# Layout panels ---------------------------------------------------------------
# FluidRow
ui <- fluidPage(
  "Side", "by", "side", "text",
  fluidRow("Text on row 1"),
  fluidRow("Text on row 2"),
  fluidRow("Text on row 3")
)
server <- function(input, output) {}
shinyApp(ui = ui, server = server)

# Column (width has to sum up to 12)
ui <- fluidPage(
  fluidRow(
    column("R1, C1, width = 3", width = 3),
    column("R1, C2, width = 9", width = 9)
  ),
  fluidRow(
    column("R2, C1, width = 4", width = 4),
    column("R2, C2, width = 4", width = 4),
    column("R2, C3, width = 4", width = 4)
  )
)
server <- function(input, output) {}
shinyApp(ui = ui, server = server)

# WellPanel (useful for grouping related UI widgets into one)
ui <- fluidPage(
  wellPanel( fluidRow("Row 1") ),
  wellPanel( fluidRow("Row 2") ),
  wellPanel( fluidRow("Row 3") )
)
server <- function(input, output) {}
shinyApp(ui = ui, server = server)

# SidebarPanel + MainPanel
ui <- fluidPage(
  sidebarLayout(
    sidebarPanel("Usually inputs go here", width = 6),
mainPanel("Usually outputs go here", width = 6)
  )
)
server <- function(input, output) {}
shinyApp(ui = ui, server = server)

# TitlePanel with windowTitle (on browser's tab)
ui <- fluidPage(
  titlePanel("Movie browser, 1970 to 2014",
             windowTitle = "Movies"),
  sidebarLayout(
    sidebarPanel("Some inputs"),
    mainPanel("Some outputs")
  )
)
server <- function(input, output) {}
shinyApp(ui = ui, server = server)

# ConditionalPanel
# The first argument in this function is the condition, which is a JavaScript expression 
# that will be evaluated repeatedly to determine whether the panel should be displayed. 
# The condition should be stated as input.show_data == true. Note that we use true instead 
# of TRUE because this is a Javascript expression, as opposed to an R expression.
# Example: show header when there is data to be displayed
conditionalPanel("input.show_data == true", h3("Data table"))

# TabsetPanel
mainPanel(
  tabsetPanel(type = "tabs",
              tabPanel(title = "Plot", 
                       plotOutput(outputId = "scatterplot"),
                       br(),
                       h4(uiOutput(outputId = "n"))),
              tabPanel(title = "Data", 
                       br(),
                       DT::dataTableOutput(outputId = "moviestable")),
              tabPanel("Codebook", 
                       br(),
                       DT::dataTableOutput("codebook"))
  )
)

