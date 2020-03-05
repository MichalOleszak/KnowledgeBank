# Prep ------------------------------------------------------------------------
library(tm)
library(qdap)
library(RWeka)
library(wordcloud)
library(plotrix)
amzn <- read.csv('500_amzn.csv')
goog <- read.csv('500_goog.csv')

# Split data into separate text vectors
amzn_pros <- amzn$pros[!is.na(amzn$pros)]
amzn_cons <- amzn$cons[!is.na(amzn$cons)]
goog_pros <- goog$pros[!is.na(goog$pros)]
goog_cons <- goog$cons[!is.na(goog$cons)]


# Cleaning functions using tm and qdap packages -------------------------------
# tm's need to be applied to a corpust
# they have more options, no need to use both, each would do it all
tm_clean_corpus <- function(corpus){
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removeWords, c(stopwords("en"), "google", "amazon", "company"))
  return(corpus)
}
qdap_clean <- function(x){
  x <- replace_abbreviation(x)
  x <- replace_contraction(x)
  x <- replace_number(x)
  x <- replace_ordinal(x)
  x <- replace_ordinal(x)
  x <- replace_symbol(x)
  x <- tolower(x)
  return(x)
}

# Clean Amazon data
amzn_pros <- qdap_clean(amzn_pros)
amzn_cons <- qdap_clean(amzn_cons)
az_p_corp <- VCorpus(VectorSource(amzn_pros))
az_c_corp <- VCorpus(VectorSource(amzn_cons))
amzn_pros_corp <- tm_clean_corpus(az_p_corp)
amzn_cons_corp <- tm_clean_corpus(az_c_corp)

# Clean Google data
goog_pros <- qdap_clean(goog_pros)
goog_cons <- qdap_clean(goog_cons)
goog_p_corp <- VCorpus(VectorSource(goog_pros))
goog_c_corp <- VCorpus(VectorSource(goog_cons))
goog_pros_corp <- tm_clean_corpus(goog_p_corp)
goog_cons_corp <- tm_clean_corpus(goog_c_corp)


# Create a bigram (2 words) wordcloud for Amazon's reviews --------------------
# tokenizer to get pairs of words
tokenizer <- function(x) {
  NGramTokenizer(x, Weka_control(min = 2, max = 2))
}
amzn_p_tdm <- TermDocumentMatrix(amzn_pros_corp, control = list(tokenize = tokenizer))
amzn_c_tdm <- TermDocumentMatrix(amzn_cons_corp, control = list(tokenize = tokenizer))
amzn_p_tdm_m <- as.matrix(amzn_p_tdm)
amzn_c_tdm_m <- as.matrix(amzn_c_tdm)
amzn_p_freq <- rowSums(amzn_p_tdm_m)
amzn_c_freq <- rowSums(amzn_c_tdm_m)
wordcloud(names(amzn_p_freq), amzn_p_freq, max.words = 25, color = "blue")
wordcloud(names(amzn_c_freq), amzn_c_freq, max.words = 25, color = "red")


# Build clustering-based dendrogram to check connections between phrases ------
amzn_c_tdm2 <- removeSparseTerms(amzn_c_tdm, sparse = 0.993)
hc <- hclust(dist(amzn_c_tdm2, method = "euclidean"), method = "complete")
plot(hc)


# Look for associations among top positive phrases ----------------------------
term_frequency <- sort(amzn_p_freq, decreasing = TRUE)
term_frequency[1:5]
findAssocs(amzn_p_tdm, "fast paced", 0.2)


# Create a commonality & comparison clouds of Google's positive ---------------
# and negative reviews 
all_goog_p <- paste(goog_pros, collapse = " ")
all_goog_c <- paste(goog_cons, collapse = " ")
all_goog_corpus  <- VCorpus(VectorSource(c(all_goog_p, all_goog_c)))
all_goog_corp <- tm_clean_corpus(all_goog_corpus)
all_tdm <- TermDocumentMatrix(all_goog_corp)
all_m <- as.matrix(all_tdm)
colnames(all_m) <- c("Goog_Pros", "Goog_Cons")
commonality.cloud(all_m, max.words = 100, colors = "steelblue1")
comparison.cloud(all_m, colors = c("#F44336", "#2196f3"), max.words = 100)


# Make a pyramid plot lining up positive reviews for Amazon and Google so -----
# you can adequately see the differences between any shared bigrams
all_tdm_m <- as.data.frame(read.csv('all_tdm_m.csv', sep = ';', stringsAsFactors = FALSE))
all_tdm_m_names <- as.character(all_tdm_m[,1])
all_tdm_m <- all_tdm_m[,-1]
all_tdm_m[,1] <- as.numeric(all_tdm_m[,1])
all_tdm_m[,2] <- as.numeric(all_tdm_m[,2])
rownames(all_tdm_m) <- all_tdm_m_names
common_words <- subset(all_tdm_m, all_tdm_m[, 1] > 0 & all_tdm_m[, 2] > 0)
difference <- abs(common_words[,1] - common_words[,2])
common_words <- as.data.frame(cbind(common_words, difference))
common_words <- common_words[order(difference, decreasing = TRUE), ]
top15_df <- data.frame(x = common_words[1:15, 1],
                       y = common_words[1:15, 2],
                       labels = rownames(common_words[1:15, ]))
pyramid.plot(top15_df$x, top15_df$y, 
             labels = top15_df$labels, gap = 12, 
             top.labels = c("Amzn", "Pro Words", "Google"), 
             main = "Words in Common", unit = NULL)