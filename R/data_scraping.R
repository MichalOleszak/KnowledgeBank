# Downloading Files and Using API Clients -------------------------------------

# Use a wikipedia API client (pageviews package) to 
# check number of visits on a wikipedia page
library(pageviews)
hadley_pageviews <- article_pageviews(project = "en.wikipedia", "Hadley Wickham")

# Using access tokens - some APIs require authorisation - to check
# frequency of the word 'vector' in publications
# (birdnik is an API client for the wordnik dictionary)
library(birdnik)
api_key <- "d8ed66f01da01b0c6a0070d7c1503801993a39c126fbc3382"
vector_frequency <- word_frequency(api_key, "vector")



# Using httr to interact with APIs directly -----------------------------------

# When no API client available - use httr package
library(httr)

# Make a get request and obtain actual content
pageview_response <- GET("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/all-agents/Hadley_Wickham/daily/20170101/20170102")
pageview_data <- content(pageview_response)

# Make post request
post_result <- POST(url = "http://httpbin.org/post", body = "this is a test")

# Check for error while getting data
fake_url <- "http://google.com/fakepagethatdoesnotexist"
request_result <- GET(fake_url)
http_error(request_result)

# Constructing queries
  # Construct a directory-based API URL to `http://swapi.co/api`,
  # looking for person `1` in `people`
directory_url <- paste("http://swapi.co/api", "people", "1", sep = "/")
result <- GET(directory_url)
  # Make parameter-based call to httpbin, with query_params
query_params <- list(nationality = "americans", 
                     country = "antigua")
parameter_response <- GET("https://httpbin.org/get", query = query_params)

# Rate limiting
urls <- c("http://fakeurl.com/api/1.0/", "http://fakeurl.com/api/2.0/")
for(url in urls){
  result <- GET(url)
  Sys.sleep(5)
}



# Handling JSON ---------------------------------------------------------------

# Get revision history for "Hadley Wickham"
resp_json <- readRDS("had_rev_json.rds") # won't work without this file :(

# Check if it's JSON and parse it
http_type(resp_json)
library(jsonlite)
fromJSON(content(resp_json, as = "text"))

# Manipulate parsed JSON with rlist
revs <- content(resp_json)$query$pages$`41916270`$revisions # store revision list
user_time <- list.select(revs, user, timestamp) # extract the user elemen
list.stack(user_time) # transform to data.frame
revs %>%
  bind_rows() %>%           
  select(user, timestamp) # the same with dplyr



# Handling XML ----------------------------------------------------------------

# Get revision history for "Hadley Wickham"
resp_xml <- readRDS("had_rev_xml.rds") # won't work without this file :(

# Check if it's XML and turn it into an XML document
http_type(resp_xml)
library(xml2)
rev_text <- content(resp_xml, as = 'text')
rev_xml <- read_xml(content(resp_xml, as = 'text'))

# Check its structure
xml_structure(rev_xml)

# Extracting XML data with XPATHs:
# /node_name specifies nodes at the current level that have the tag node_name, 
# //node_name specifies nodes at any level below the current level that have the tag node_name
# Find all nodes using XPATH "/api/query/pages/page/revisions/rev"

xml_find_all(rev_xml, "/api/query/pages/page/revisions/rev")
# Find all rev nodes anywhere in document
rev_nodes <- xml_find_all(rev_xml, "//rev")
# Use xml_text() to get text from rev_nodes
xml_text(rev_nodes)

# Extracting XML attributes with XPATHs:
# xml_attrs() takes a nodeset and returns all of the attributes for every node in the nodeset,
# xml_attr() takes a nodeset and an additional argument attr to extract a single named argument from each node in the nodeset.

# Find user attribute for all rev nodes
xml_attr(rev_nodes, "user")



# Web Scraping with XPATH -----------------------------------------------------
library(rvest)
test_url <- "https://en.wikipedia.org/wiki/Hadley_Wickham"
test_xml <- read_html(test_url)
test_node_xpath <- "//*[contains(concat( \" \", @class, \" \" ), concat( \" \", \"vcard\", \" \" ))]"
node <- html_node(x = test_xml, xpath = test_node_xpath)

# Extract name (tag)
html_name(node)  # because this node has tag <table>...</table>

# Extract value (content)
# node containing page title:
second_xpath_val <- "//*[contains(concat( \" \", @class, \" \" ), concat( \" \", \"fn\", \" \" ))]"
page_name <- html_node(x = node, xpath = second_xpath_val)
html_text(page_name)

# Extract tables
wiki_table <- html_table(node)


# Web Scraping with CSS -------------------------------------------------------
# CSS is a way to add design information to HTML affecting browser's display:
#   .class_a {
#     color: black;
#   }
#   .class_b {
#     color: red;
#   }
#   <a class = "class_a" href = "http://en.wikipedia.org/"> This is black </a>
#   <a class = "class_b" href = "http://en.wikipedia.org/"> This is red </a>

# select the all elements with 'table' tag
table_element <- html_nodes(test_xml, css = "table")
# select all elements that have the attribute class = "infobox"
infobox_element <- html_nodes(test_xml, css = ".infobox")
# select all elements that have the attribute id = "firstHeading"
id_element <- html_nodes(test_xml, css = "#firstHeading")

# Extract name
html_name(infobox_element)

# Extract value
page_name <- html_node(x = infobox_element, css = ".fn")
html_text(page_name)

# Getting wikipedia infobox ---------------------------------------------------
library(httr)
library(rvest)
library(xml2)

get_infobox <- function(title){
  base_url <- "https://en.wikipedia.org/w/api.php"
  
  query_params <- list(action = "parse", 
                       page = title, 
                       format = "xml")
  
  resp <- GET(url = base_url, query = query_params)
  resp_xml <- content(resp)
  
  page_html <- read_html(xml_text(resp_xml))
  infobox_element <- html_node(x = page_html, css =".infobox")
  page_name <- html_node(x = infobox_element, css = ".fn")
  page_title <- html_text(page_name)
  
  wiki_table <- html_table(infobox_element)
  colnames(wiki_table) <- c("key", "value")
  cleaned_table <- subset(wiki_table, !wiki_table$key == "")
  name_df <- data.frame(key = "Full name", value = page_title)
  wiki_table <- rbind(name_df, cleaned_table)
  
  wiki_table
}


get_infobox(title = "Hadley Wickham")
get_infobox(title = "Ross Ihaka")
get_infobox(title = "Grace Hopper")
