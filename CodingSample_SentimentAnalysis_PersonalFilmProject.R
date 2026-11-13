#1. Select two classes from the productreviewsclass.csv dataset. You can also use your dataset, which I prefer.
#We have selected film reviews from 2 movies: Oppenheimer and Barbie. We found them on https://www.kaggle.com/datasets which is a social repository of datasets
#They have one dataset of 84,048 comments for Oppenheimer and another dataset with 23,957 reviews from Barbie. Even though Oppenheimer's ones had the date and user 
#as part of the information, they were part of the comment/review, we decided to eliminate them to have, all together with Barbie's, one standard plain review. 
#After that we replace them on the reviews from Apple and Samsung on the file used in class. We used in that sense the first rows of our datasets, 
#not a sample of them. The other "brand" records on the dataset were deleted. Finally, we got the comma separated values file named "Films Reviews Classfied.csv"

#2. Do the text preprocessing
library(tm)
library(topicmodels)
library(slam)
library(pals)
library(data.table)
library(text2vec)
library(wordcloud)
library(seededlda)
library(quanteda)
options(seededlda_threads = 1)

library(caret)
library(tidyverse)
library(SnowballC)
library(textstem)

# Set global option to prevent automatic conversion of strings to factors
options(stringsAsFactors = FALSE)

# Read film reviews from a CSV file into a dataframe
dt1 <- read.csv("Films Reviews Classified.csv")

# we select two films, Oppenheimer and (/or) Barbie
dt2 <- dt1 %>%
  filter(Brand == "Oppenheimer" | Brand == "Barbie")
dtb <- dt1 %>%
  filter(Brand == "Barbie")
dto <- dt1 %>%
  filter(Brand == "Oppenheimer")

# select the first 300 (to run the code quicker)
dt2 <- dt2[1:300,]
dtb2 <- dtb[1:300,]
dto2 <- dto[1:300,]

# select only the important columns, such as the content and the brand 
dt3 <- dt2 %>% 
  select(Content, Brand, Review_date)
dtb3 <- dtb2 %>% 
  select(Content, Brand)
dto3 <- dto2 %>% 
  select(Content, Brand)

# discard the unnecessary data frames
rm(dt1, dt2)

# put the Content column into a corpus to perform preprocessing operations
corpus <- VCorpus(VectorSource(dt3$Content))
corpusb <- VCorpus(VectorSource(dtb3$Content))
corpuso <- VCorpus(VectorSource(dto3$Content))

# preprocess and save the corpus in a Document Term Matrix
dtm <- corpus %>%
  tm_map(content_transformer(tolower)) %>% # transform into lowercase
  tm_map(content_transformer(removeWords), stopwords("english")) %>% # remove the stopwords  
  tm_map(removePunctuation) %>% # remove punctuation
  tm_map(removeNumbers) %>% # remove numbers
  #tm_map(stemDocument) %>% # stemming
  tm_map(lemmatize_words) %>% # lemmatization
  tm_map(stripWhitespace) %>% # white space removal
  DocumentTermMatrix() %>% # transform into a Document Term Matrix
  #weightTfIdf() %>% # apply the Tf-Idf weighting function
  removeSparseTerms(0.98) # adjust for sparsity
dtmb <- corpusb %>%
  tm_map(content_transformer(tolower)) %>% # transform into lowercase
  tm_map(content_transformer(removeWords), stopwords("english")) %>% # remove the stopwords  
  tm_map(removePunctuation) %>% # remove punctuation
  tm_map(removeNumbers) %>% # remove numbers
  #tm_map(stemDocument) %>% # stemming
  tm_map(lemmatize_words) %>% # lemmatization
  tm_map(stripWhitespace) %>% # white space removal
  DocumentTermMatrix() %>% # transform into a Document Term Matrix
  #weightTfIdf() %>% # apply the Tf-Idf weighting function
  removeSparseTerms(0.98) # adjust for sparsity
dtmo <- corpuso %>%
  tm_map(content_transformer(tolower)) %>% # transform into lowercase
  tm_map(content_transformer(removeWords), stopwords("english")) %>% # remove the stopwords  
  tm_map(removePunctuation) %>% # remove punctuation
  tm_map(removeNumbers) %>% # remove numbers
  #tm_map(stemDocument) %>% # stemming
  tm_map(lemmatize_words) %>% # lemmatization
  tm_map(stripWhitespace) %>% # white space removal
  DocumentTermMatrix() %>% # transform into a Document Term Matrix
  #weightTfIdf() %>% # apply the Tf-Idf weighting function
  removeSparseTerms(0.98) # adjust for sparsity


# Term Document Matrix
tdm <- corpus %>%
  tm_map(content_transformer(tolower)) %>% # transform into lowercase
  tm_map(content_transformer(removeWords), stopwords("english")) %>% # remove the stopwords  
  tm_map(removePunctuation) %>% # remove punctuation
  tm_map(removeNumbers) %>% # remove numbers
  #tm_map(stemDocument) %>% # stemming
  tm_map(lemmatize_words) %>% # lemmatization
  tm_map(stripWhitespace) %>% # white space removal
  TermDocumentMatrix() %>% # transform into a Document Term Matrix
  #weightTfIdf() %>% # apply the Tf-Idf weighting function
  removeSparseTerms(0.98) # adjust for sparsity

#3. Words frequency
library(slam)
v <- sort(row_sums(tdm), decreasing=TRUE)
d <- data.frame(word = names(v), freq=v)
head(d, 10)
barplot(d[1:10,]$freq, las = 2, names.arg = d[1:10,]$word,
        col ="lightblue", main ="Most frequent words",
        ylab = "Word frequencies")

#4. Wordcloud: commonality and comparison using one-gram and bi-gram tokens
library(wordcloud)
library(RWeka)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35,
          colors=brewer.pal(8, "Dark2"))

# bigram function, to make another gram type, you have to change the min and max value
Bigram_Tokenizer <- function(x){
  NGramTokenizer(x, Weka_control(min=2, max=2))
}

# create a bigram matrix
bitdm <- TermDocumentMatrix(corpus, control = list(tokenize = Bigram_Tokenizer))

# remove some sparsity
bitdms <- removeSparseTerms(bitdm, 0.98)

# transform into a regular matrix
bitdmsm <- as.matrix(bitdms)
v <- sort(rowSums(bitdmsm), decreasing=TRUE)
d <- data.frame(word = names(v), freq=v)

# create the bi-gram word cloud
wordcloud(words = d$word, freq = d$freq, random.order = FALSE)

library(magrittr)

# selection of the vectors containing the words oppenheimer and barbie
# through the command grep
# unifying the comments through the command paste making two big documents
sub_op <- paste(grep("Oppenheimer", dto3$Content,  value=TRUE), collapse = " ")
sub_bb <- paste(grep("Barbie", dtb3$Content,  value=TRUE), collapse = " ")

# combining the two vectors and putting them into a corpus data type
movies <- c(sub_op, sub_bb)
corpuscom <- VCorpus(VectorSource(movies))

# make the necessary transformations into a Vector Space
tdmcom <- TermDocumentMatrix(corpuscom)
tdmmatrix <- as.matrix(tdmcom)

# name the columns we want to compare
colnames(tdmmatrix) = c("Oppenheimer", "Barbie")

# commonality cloud
commonality.cloud(tdmmatrix, max.words=200, random.order=FALSE)

# comparison cloud
comparison.cloud(tdmmatrix, colors = c("orange", "pink"), max.words=200, random.order=FALSE)

# create a bigram matrix for each movie
# bigram function, to make another gram type, you have to change the min and max value
Bigram_Tokenizer <- function(x){
  NGramTokenizer(x, Weka_control(min=2, max=2))
}

# Put the Content column of each movie into a corpus to perform preprocessing operations
corpusop <- VCorpus(VectorSource(dto3$Content))
corpusbb <- VCorpus(VectorSource(dtb3$Content))

# Create TermDocumentMatrix for each corpus
bitdmop <- TermDocumentMatrix(corpusop, control = list(tokenize = Bigram_Tokenizer))
bitdmbb <- TermDocumentMatrix(corpusbb, control = list(tokenize = Bigram_Tokenizer))

# Remove some sparsity
bitdmsop <- removeSparseTerms(bitdmop, 0.98)
bitdmsbb <- removeSparseTerms(bitdmbb, 0.98)

# Transform into a regular matrix
bitdmsmop <- as.matrix(bitdmsop)
bitdmsmbb <- as.matrix(bitdmsbb)

# Get word frequencies
vop <- sort(rowSums(bitdmsmop), decreasing = TRUE)
vbb <- sort(rowSums(bitdmsmbb), decreasing = TRUE)

# Create data frames
dop <- data.frame(word = names(vop), freq = vop)
dbb <- data.frame(word = names(vbb), freq = vbb)

# Merge data frames by the 'word' column to create a common matrix
merged_df <- merge(dop, dbb, by = "word", all = TRUE)

# Rename columns for clarity
colnames(merged_df) <- c("word", "freq_op", "freq_bb")

# Replace NA with 0
merged_df[is.na(merged_df)] <- 0

# Convert to matrix
comparison_matrix <- as.matrix(merged_df[, -1])
rownames(comparison_matrix) <- merged_df$word

# Create a commonality cloud
commonality.cloud(comparison_matrix, max.words=200, random.order=FALSE)
# Create a comparison cloud
comparison.cloud(comparison_matrix, colors = c("orange", "pink"), max.words=200, random.order=FALSE, title.size = 1.5)

#5. Wordnetwork for both classes
library(igraph)
d.separated <- d %>% separate(word, c("word1", "word2"), sep = " ")
head(d.separated)

# build a filtered word network from the above created dataframe
word.network <- d.separated %>% filter(freq > 10) %>% graph_from_data_frame()

# visualize the created word network
word.network

# [Step 5 original] Load necessary libraries
library(tidytext)
library(ggraph)
library(stringr)

# Load the data
# Select relevant columns
reviews <- dt3 %>% select(Content)
dates <- dt3 %>% select(Review_date)
brands <- dt3 %>% select(Brand)

# Unnest tokens
tidy_reviews <- reviews %>%
  unnest_tokens(word, Content)

# Remove stop words
data("stop_words")
tidy_reviews <- tidy_reviews %>%
  anti_join(stop_words)

# Create bigrams
bigrams <- reviews %>%
  unnest_tokens(bigram, Content, token = "ngrams", n = 2)

# Separate bigrams into two columns
bigrams_separated <- bigrams %>%
  separate(bigram, c("word1", "word2"), sep = " ")

# Filter out stop words from bigrams
bigrams_filtered <- bigrams_separated %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word)

# Count bigrams
bigram_counts <- bigrams_filtered %>%
  count(word1, word2, sort = TRUE)

# Create a graph from bigrams
bigram_graph <- bigram_counts %>%
  filter(n > 1) %>%
  graph_from_data_frame()

# Plot the word network
set.seed(1234)
ggraph(bigram_graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE) +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), vjust = 1.8, size = 3) +
  theme_void()


#6. Do the sentiment graph for both classes and display it on the same graph
# Load necessary libraries
library(readr)
library(widyr)
library(lubridate)  # For date manipulation if needed
library(rvest)
library(dplyr)
library(ldatuning)
library(ggplot2)
library(textclean)
library(textmineR)
library(syuzhet)
library(nametagger)

reviewso <- as.character(dto3$Content)
dateso <- as.character(dto3$Review_date)
reviewsb <- as.character(dtb3$Content)
datesb <- as.character(dtb3$Review_date)

sentiment_scores_op <- get_nrc_sentiment(reviewso)
sentiment_scores_bb <- get_nrc_sentiment(reviewsb)
sentiment_scores_op$date <- dateso
sentiment_scores_bb$date <- datesb
sentiment_scores_op$date <- as.POSIXct(sentiment_scores_op$date, format="%m/%d/%Y")
sentiment_scores_bb$date <- as.POSIXct(sentiment_scores_bb$date, format="%m/%d/%Y")
sentiment_df_op <- data.frame(date = sentiment_scores_op$date, sentiment_scores_op)
sentiment_df_bb <- data.frame(date = sentiment_scores_bb$date, sentiment_scores_bb)
sentiment_summary_op <- sentiment_df_op %>%
  group_by(date) %>%
  summarise(across(anger:positive, sum))
sentiment_summary_bb <- sentiment_df_bb %>%
  group_by(date) %>%
  summarise(across(anger:positive, sum))

library(scales)

# lims <- as.POSIXct(strptime(c("2023-07-15", "2023-10-15"), format = "%Y-%m-%d"))

ggplot() +
  geom_line(data=sentiment_summary_op, aes(x = date, y = positive, color = "Positive Oppenheimer")) +
  geom_line(data=sentiment_summary_op, aes(x = date, y = negative, color = "Negative Oppenheimer")) +
  geom_line(data=sentiment_summary_bb, aes(x = date, y = positive, color = "Positive Barbie")) +
  geom_line(data=sentiment_summary_bb, aes(x = date, y = negative, color = "Negative Barbie")) +
  labs(title = "Sentiment Over Time", x = "Date", y = "Sentiment Score") +
  # scale_x_datetime(labels = date_format("%Y-%m-%d"), limits = lims, expand = c(0,0)) +
  scale_color_manual(values = c("Positive Oppenheimer" = "orange", "Negative Oppenheimer" = "red", "Positive Barbie" = "pink", "Negative Barbie" = "lightblue"))

#7. Do the topic model of both texts (LDA), choose a reasonable number of topics, and try to make sense of them by giving them a label. You can also use the seed words version, but make sure you use both one-gram and bi-gram tokens. Use text2vec and compare the results with the topics obtained through LDA.
#Load needed libraries
library(factoextra)
library(NbClust)
library(randomForest)
library(qdap)
library(mgsub)
library(plotly)
library(tidyr)
library(ggthemes)

df <- read.csv("Films Reviews Classified.csv")

str(df)
head(df)

na_count <- sum(is.na(df$Content))
print(paste("Number of NA values in Content:", na_count))
empty_count <- sum(df$Content == "")
print(paste("Number of empty strings in Content:", empty_count))

df <- df %>%
  filter(!is.na(Content) & Content != "")
dfb <- df %>%
  filter(Brand == "Barbie")
dfo <- df %>%
  filter(Brand == "Oppenheimer")

# Text preprocessing
corpusb <- Corpus(VectorSource(dfb$Content))
corpusb <- corpusb %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("en")) %>%
  tm_map(stripWhitespace)

dtmb <- DocumentTermMatrix(corpusb)
dtmb <- removeSparseTerms(dtmb, 0.95)

non_empty_docs <- rowSums(as.matrix(dtmb)) > 0
dtmb <- dtmb[non_empty_docs, ]

corpuso <- Corpus(VectorSource(dfo$Content))
corpuso <- corpuso %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("en")) %>%
  tm_map(stripWhitespace)

dtmo <- DocumentTermMatrix(corpuso)
dtmo <- removeSparseTerms(dtmo, 0.95)

non_empty_docs <- rowSums(as.matrix(dtmb)) > 0
dtmo <- dtmo[non_empty_docs, ]


# LDA Application for topic findings
num_topics <- 5  
lda_modelb <- LDA(dtmb, k = num_topics, control = list(seed = 123))
lda_modelo <- LDA(dtmo, k = num_topics, control = list(seed = 123))

simple_lda_topicsb <- tidy(lda_modelb, matrix = "beta")
simple_lda_topicso <- tidy(lda_modelo, matrix = "beta")

simple_lda_top_termsb <- simple_lda_topicsb %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)
simple_lda_top_termso <- simple_lda_topicso %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

print("Simple LDA Top Terms Barbie")
print(simple_lda_top_termsb)
print("Simple LDA Top Terms Oppenheimer")
print(simple_lda_top_termso)

# Seeded LDA

corpb <- corpusb
corpo <- corpuso

toksb <- tokens(corpb, remove_punct = TRUE, remove_symbols = TRUE, 
               remove_numbers = TRUE, remove_url = TRUE)
tokso <- tokens(corpo, remove_punct = TRUE, remove_symbols = TRUE, 
               remove_numbers = TRUE, remove_url = TRUE)

dfmtb <- dfm(toksb) |> 
  dfm_remove(stopwords("en")) |>
  dfm_remove("*@*") |>
  dfm_trim(max_docfreq = 0.1, docfreq_type = "prop")
dfmto <- dfm(tokso) |> 
  dfm_remove(stopwords("en")) |>
  dfm_remove("*@*") |>
  dfm_trim(max_docfreq = 0.1, docfreq_type = "prop")

# Seed words
dict <- dictionary(list(
  topic1 = c("good", "great", "excellent"),
  topic2 = c("bad", "poor", "terrible"),
  topic3 = c("prize", "oscar", "value"),
  topic4 = c("margot", "ken", "actor"),
  topic5 = c("war", "bomb", "downey")
))

set.seed(42)
lda_seedb <- textmodel_seededlda(dfmtb, dict, auto_iter = TRUE)
lda_seedo <- textmodel_seededlda(dfmto, dict, auto_iter = TRUE)

seeded_lda_top_termsb <- terms(lda_seedb, 10) %>%
  as.data.frame() %>%
  pivot_longer(cols = everything(), names_to = "topic", values_to = "term") %>%
  group_by(topic) %>%
  mutate(beta = row_number()) %>%
  ungroup()
seeded_lda_top_termso <- terms(lda_seedo, 10) %>%
  as.data.frame() %>%
  pivot_longer(cols = everything(), names_to = "topic", values_to = "term") %>%
  group_by(topic) %>%
  mutate(beta = row_number()) %>%
  ungroup()

# Mismatching topic column fix
simple_lda_top_termsb <- simple_lda_top_termsb %>%
  mutate(topic = paste0("topic", topic))
simple_lda_top_termso <- simple_lda_top_termso %>%
  mutate(topic = paste0("topic", topic))

comparisonb <- merge(simple_lda_top_termsb, seeded_lda_top_termsb, 
                    by = c("topic", "term"), suffixes = c("_simple", "_seeded"))
comparisono <- merge(simple_lda_top_termso, seeded_lda_top_termsb, 
                    by = c("topic", "term"), suffixes = c("_simple", "_seeded"))

print("Comparison of Simple LDA and Seeded LDA Top Terms for Barbie")
print(comparisonb)
print("Comparison of Simple LDA and Seeded LDA Top Terms for Oppenheimer")
print(comparisono)

# Bigram Seeded LDA

corp_smallb <- corpus(dtb2, text_field = "Content")
corp_smallo <- corpus(dto2, text_field = "Content")

toks_smallb <- tokens(corp_smallb, remove_punct = TRUE, remove_symbols = TRUE, 
                     remove_numbers = TRUE, remove_url = TRUE) %>%
  tokens_ngrams(n = 2)
toks_smallo <- tokens(corp_smallo, remove_punct = TRUE, remove_symbols = TRUE, 
                     remove_numbers = TRUE, remove_url = TRUE) %>%
  tokens_ngrams(n = 2)

dfmt_bigrams_smallb <- dfm(toks_smallb) |> 
  dfm_remove(stopwords("en")) |>
  dfm_trim(max_docfreq = 0.1, docfreq_type = "prop")
dfmt_bigrams_smallo <- dfm(toks_smallo) |> 
  dfm_remove(stopwords("en")) |>
  dfm_trim(max_docfreq = 0.1, docfreq_type = "prop")

lda_seed_bigrams_smallb <- textmodel_seededlda(dfmt_bigrams_smallb, dict, auto_iter = TRUE)
lda_seed_bigrams_smallo <- textmodel_seededlda(dfmt_bigrams_smallo, dict, auto_iter = TRUE)

seeded_lda_bigrams_top_terms_smallb <- terms(lda_seed_bigrams_smallb, 10) %>%
  as.data.frame() %>%
  pivot_longer(cols = everything(), names_to = "topic", values_to = "term") %>%
  group_by(topic) %>%
  mutate(beta = row_number()) %>%
  ungroup()
seeded_lda_bigrams_top_terms_smallo <- terms(lda_seed_bigrams_smallo, 10) %>%
  as.data.frame() %>%
  pivot_longer(cols = everything(), names_to = "topic", values_to = "term") %>%
  group_by(topic) %>%
  mutate(beta = row_number()) %>%
  ungroup()

print("Seeded LDA Bigrams Top Terms for Small Dataset Barbie")
print(seeded_lda_bigrams_top_terms_smallb)
print("Seeded LDA Bigrams Top Terms for Small Dataset Oppenheimer")
print(seeded_lda_bigrams_top_terms_smallo)


#8 Use text embeddings to find topics and words

# First create a TCM using skip grams, we'll use a 5-word window
# most options available on CreateDtm are also available for CreateTcm
tcm <- CreateTcm(doc_vec = dt3$Content,
                 skipgram_window = 10,
                 verbose = FALSE,
                 cpus = 2)

# a TCM is generally larger than a DTM
dim(tcm)

# Fitting a model

# use LDA to get embeddings into probability space
# This will take considerably longer as the TCM matrix has many more rows 
# than your average DTM
embeddings <- FitLdaModel(dtm = tcm,
                          k = 50,
                          iterations = 200,
                          burnin = 180,
                          alpha = 0.1,
                          beta = 0.05,
                          optimize_alpha = TRUE,
                          calc_likelihood = FALSE,
                          calc_coherence = TRUE,
                          calc_r2 = TRUE,
                          cpus = 2)

## Evaluating the model

# Get an R-squared for general goodness of fit
embeddings$r2


# Get coherence (relative to the TCM) for goodness of fit
summary(embeddings$coherence)

# Get top terms, no labels because we don't have bigrams
embeddings$top_terms <- GetTopTerms(phi = embeddings$phi,
                                    M = 5)

embeddings$top_terms

# Create a summary table, similar to the above
embeddings$summary <- data.frame(topic = rownames(embeddings$phi),
                                 coherence = round(embeddings$coherence, 3),
                                 prevalence = round(colSums(embeddings$theta), 2),
                                 top_terms = apply(embeddings$top_terms, 2, function(x){
                                   paste(x, collapse = ", ")
                                 }),
                                 stringsAsFactors = FALSE)

embeddings$summary[ order(embeddings$summary$prevalence, decreasing = TRUE) , ][ 1:10 , ]

embeddings$summary[ order(embeddings$summary$coherence, decreasing = TRUE) , ][ 1:10 , ]

# Make a DTM from our documents
dtm_embed <- CreateDtm(doc_vec = nih_sample$ABSTRACT_TEXT,
                       doc_names = nih_sample$APPLICATION_ID,
                       ngram_window = c(1,1),
                       verbose = FALSE,
                       cpus = 2)

dtm_embed <- dtm_embed[,colSums(dtm_embed) > 2]

# Project the documents into the embedding space
embedding_assignments <- predict(embeddings, dtm_embed, method = "dot",
                                 iterations = 200, burnin = 180)

# get a goodness of fit relative to the DTM
embeddings$r2_dtm <- CalcTopicModelR2(dtm = dtm_embed, 
                                      phi = embeddings$phi[,colnames(dtm_embed)], # line up vocabulary
                                      theta = embedding_assignments,
                                      cpus = 2)

embeddings$r2_dtm

# get coherence relative to DTM
embeddings$coherence_dtm <- CalcProbCoherence(phi = embeddings$phi[,colnames(dtm_embed)], dtm = dtm_embed)
          
summary(embeddings$coherence_dtm)              


#9 Blend the information from the clouds, sentiment graph, topics, embeddings, and word graphs, and explain what the texts/products are about, how they are different, and the insight you get from them

#Both movies were released to the market on the same day, and both came from Hollywood, so they are "somehow" comparable.
#Text analysis for movies can give you some insight into "performance," especially when comparing them, but it is not helpful when making "decisions."
#Word frequency and clouds gave us tiny details since "popular" monograms, bigrams, clouds, and networks were so "obvious" and sometimes dispersed due to common nonsense associations between words when they refer to "user" comments (viewers).
#However, sentiment analysis has valuable insights: Barbie had fewer positive comments than Oppenheimer at the launching date, but comments kept arriving after so many days. Oppenheimer's comments stop.
#Marketing-wise, Oppenheimer impacted the film industry most when it was released, but people stopped talking about it quickly, and Barbie was quite warm at the beginning.
#For the topic analysis, using LDA gave us very poor insights for marketing decisions, as explained before. However, we could obtain important topics using text embeddings, such as music, and inclusion themes, such as ethnicity and gender, among others.

#10. Write a small report giving a DETAILED description of how you would use the information obtained in marketing. Do not forget to add the literature to the text

#It is essential to understand datasets effectively before doing any analysis. We should also include pre "quantitative" analysis for, for example, frequency and volume of comments, so in that sense, we have a big picture of the information to be analyzed. The text preprocessing could also include language filtering since, in big datasets, there are docs that come globally, especially regarding movie insights. The text analysis for the film industry has partially valuable feedback to the producer and the industry in general since, as explained in the previous section, film viewers are dispersed and repetitive when they write their comments. However, this type of analysis and tool in the products industry is potent; it can provide the marketing teams with valuable feedback to make decisions, such as improving the product or even taking it out of the market.


#Be creative!

#Upload your presentation and provide the code you have used. Do not introduce code in the presentation unless it is absolutely necessary. The presentation should last 10 minutes. 