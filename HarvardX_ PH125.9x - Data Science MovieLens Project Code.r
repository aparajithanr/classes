#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
     semi_join(edx, by = "movieId") %>%
     semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

#----------Data Cleaning starts
names(edx)

# Since we want to predict ratings irrespective of timestamp, remove timestamp
edx_subset <- edx %>% select(-timestamp)
names(edx_subset)

# Since genres is type of string and there are only number of genres repeated, we replace the string with numbers
edx_subset <- edx_subset %>% mutate(genres = as.numeric(as.factor(genres)))

# Since the column title that is unique and has a collinearity only with movieId we treat it as overhead
# Before removing, let's capture them for future reference
movie_titles <- unique(edx_subset %>% select(movieId, title))
edx_subset <- edx_subset %>% select(-title)

# Prepare sampling for 5 & 4.5 ratings
train_n <- 100
train_1 <- edx_subset %>% filter(rating == 5) %>% mutate(rating = 'A')
train_2 <- edx_subset %>% filter(rating == 4.5) %>% mutate(rating = 'B')
set.seed(12345)
train_i_1 <- sample(1:nrow(train_1), train_n, replace = FALSE)
train_i_2 <- sample(1:nrow(train_2), train_n, replace = FALSE)
train <- rbind(train_1[train_i_1, ], train_2[train_i_2, ])

# We have 3 features to predict rating
names(train)

# Train model for binomial logistic regression to see how the overall variable importance is between the given features
control <- trainControl(method="repeatedcv", number=10, repeats=3)
model <- train(as.factor(rating)~., data=train, method="glm", preProcess="scale", trControl=control)

# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

# Let's start understanding the variances by each feature - movieId, userId, & genre
#----------Data Cleaning ends

#----------Data Exploration starts
#----------Understanding movies-rating distribution
m_group <- edx_subset %>% select(movieId, rating) %>% group_by(movieId) %>% dplyr::summarise(r_count = n())
mr_group <- m_group %>% group_by(r_count) %>% dplyr::summarise(mr_count = n())
qplot(r_count, mr_count, data = mr_group, color = I("black"))
qplot(r_count, mr_count, data = mr_group[which(mr_group$r_count > 1000 & mr_group$r_count < 30000),], color = I("black"))
hist(mr_group$r_count)
# To understand better relative distribution, let's remove the movies which has less than 1000 ratings & greater than 30000 ratings
hist(mr_group$r_count[which(mr_group$r_count > 1000 & mr_group$r_count < 30000)])
# Most of the movies have recieved less 5000 ratings, given the max number of ratings 30000

#----------Understanding Users-rating distribution
u_group <- edx_subset %>% select(userId, rating) %>% group_by(userId) %>% dplyr::summarise(r_count = n())
ur_group <- u_group %>% group_by(r_count) %>% dplyr::summarise(ur_count = n())
qplot(r_count, ur_count, data = ur_group, color = I("black"))
hist(ur_group$r_count)
# Most of the users have rated less 2000 ratings, given the max number of ratings 4500

#----------Understanding Genres-rating distribution
g_group <- edx_subset %>% select(genres, rating) %>% group_by(genres) %>% dplyr::summarise(r_count = n())
gr_group <- g_group %>% group_by(r_count) %>% dplyr::summarise(gr_count = n())
qplot(r_count, gr_count, data = gr_group, color = I("black"))
qplot(r_count, gr_count, data = gr_group[which(gr_group$r_count > 100 & gr_group$r_count < 60000),], color = I("black"))
hist(gr_group$r_count)
# To understand better relative distribution, let's remove the genres which has less than 100 ratings & greater than 60000 ratings
hist(gr_group$r_count[which(gr_group$r_count > 100 & gr_group$r_count < 60000)])
# Most of the genres have recieved less 5000 ratings, given the max number of ratings 60000
#----------Data Exploration ends

#----------Prediction Modeling starts
# Define function as If RMSE > 1, not a good prediction
RMSE <- function(true_ratings, predicted_ratings) {
	sqrt(mean((true_ratings - predicted_ratings) ^ 2))
}

# We compute this average on the training data.
mu_hat <- mean(edx_subset$rating)

# Initialize results frame to store RMSEs as calculated & improvised through out the modeling process
rmse_results <- data.frame()

#----------1. By Overall Average
# We compute the residual mean squared error (naive_rmse) on the test set data by overall average.
predictions <- mu_hat # This assignment is performed just for readability purposes
rmse_1 <- RMSE(edx_subset$rating, predictions)
rmse_results <- bind_rows(rmse_results, tibble(method = "By overall average", RMSE = round(rmse_1, digits = 3)))


#----------2. By Movie Effects
# Each movie is rated differently -> can be seen by finding the average of ratings for each movie
avg_m_rating <- edx_subset %>% select(movieId, rating) %>% group_by(movieId) %>% dplyr::summarise(avg = mean(rating))
hist(avg_m_rating$avg)
# Most of the average ratings are between 2.5 & 4.0

# Hence, Y_hat u,m = mu_hat + b_hat_m + E u,m -> where b_hat_m (bias) is the average rating bias for the movie m
# b_hat_m = avg(y_hat u,m - mu_hat)
movie_avgs <- edx_subset %>% select(movieId, rating) %>% group_by(movieId) %>% dplyr::summarise(b_hat_m = mean(rating - mu_hat))

# b_hat_i values vary substantially
hist(movie_avgs$b_hat_m)

# Relationship btw mu_hat & b_hat_m -> Y_hat u,m = mu_hat + b_hat_m (3.5 overall mean + 1.5 bias for a 5 star rated movie)
predictions <- mu_hat + (edx_subset %>% left_join(movie_avgs, by = "movieId") %>% .$b_hat_m)
rmse_2 <- RMSE(edx_subset$rating, predictions)
rmse_results <- bind_rows(rmse_results, tibble(method = "By movie based average", RMSE = round(rmse_2, digits = 3)))

# top 10 best movies by movie averages
edx_subset %>% count(movieId) %>% left_join(movie_avgs, by = "movieId") %>% left_join(movie_titles, by = "movieId") %>% arrange(desc(b_hat_m)) %>% select(title, b_hat_m, n) %>% slice(1:10) %>% knitr::kable()

# top 10 worst movies by movie averages
edx_subset %>% count(movieId) %>% left_join(movie_avgs, by = "movieId") %>% left_join(movie_titles, by = "movieId") %>% arrange(b_hat_m) %>% select(title, b_hat_m, n) %>% slice(1:10) %>% knitr::kable()

# Most of the movies, which are chosen as the best and worst, have received very few ratings. Most of them are obscure.
# We should not trust these noisy estimates and so need to use regularization
# However, before we go for regularization, we will try the user effects too


#----------3. By Movie + User Effects
# Each user rated differently -> can be seen by finding the average of ratings for each user
avg_u_rating <- edx_subset %>% select(userId, rating) %>% group_by(userId) %>% dplyr::summarise(avg = mean(rating))
hist(avg_u_rating$avg)
# Most of the average ratings are between 3 & 4.0

# As realized, there are user-specific effect (Happy Users, Cranky Users, Reasonable Users), we introduce b_hat_u
# Y_hat u,m = mu_hat + b_hat_m + b_hat_u + E u,m -> where b_hat_u (bias) is the average rating bias by the user u
user_avgs <- edx_subset %>% left_join(movie_avgs, by="movieId") %>% select(b_hat_m, userId, rating) %>% group_by(userId) %>% dplyr::summarise(b_hat_u = mean(rating - mu_hat - b_hat_m))

# Relationship btw mu_hat, b_hat_m & b_hat_u -> Y_hat u,m = mu_hat + b_hat_m + b_hat_u
predictions <- edx_subset %>% left_join(movie_avgs, by = "movieId") %>% left_join(user_avgs, by = "userId") %>% mutate(pred = mu_hat + b_hat_m + b_hat_u) %>% .$pred
rmse_3 <- RMSE(edx_subset$rating, predictions)
rmse_results <- bind_rows(rmse_results, tibble(method = "By movie & user based average", RMSE = round(rmse_3, digits = 3)))

# top 10 best movies by user averages
user_avgs %>% left_join(edx_subset, by = "userId") %>% left_join(movie_titles, by = "movieId") %>% arrange(desc(b_hat_u)) %>% select(title, b_hat_u) %>% slice(1:10) %>% knitr::kable()

# top 10 worst movies by user averages
user_avgs %>% left_join(edx_subset, by = "userId") %>% left_join(movie_titles, by = "movieId") %>% arrange(b_hat_u) %>% select(title, b_hat_u) %>% slice(1:10) %>% knitr::kable()

# Most of the positively biased movies are popular/blockbuster movies vs. negatively biased ones are less popular around the world


#----------4. By Regularized Movie + Regularized User Effects
lambdas <- seq(0, 15, 0.5)

# Recalculate Lambda
just_sum_m <- edx_subset %>% group_by(movieId)%>% dplyr::summarise(s_m = sum(rating - mu_hat), n_hat_m = n())
just_sum_u <- edx_subset %>% group_by(userId)%>% dplyr::summarise(s_u = sum(rating - mu_hat), n_hat_u = n())
rmses_movie_user <- sapply(lambdas, function(l) {
    predicted_ratings <- edx_subset %>% left_join(just_sum_m, by = "movieId") %>% mutate(b_hat_m = s_m / (l + n_hat_m)) %>% left_join(just_sum_u, by = "userId") %>% mutate(b_hat_u = s_u / (l + n_hat_u)) %>% mutate(pred = (mu_hat + b_hat_m + b_hat_u)) %>% .$pred
    RMSE(edx_subset$rating, predicted_ratings)
}
)
plot(rmses_movie_user, lambdas)
lambda_movie_user <- lambdas[which.min(rmses_movie_user)]

# Revise movie_user_avgs with the regularized biases
movie_reg_avgs <- just_sum_m %>% mutate(b_hat_m_reg = s_m / (lambda_movie_user + n_hat_m))
user_reg_avgs <- edx_subset %>% left_join(movie_reg_avgs, by = "movieId") %>% select(b_hat_m_reg, userId, rating) %>% group_by(userId) %>% dplyr::summarise(b_hat_u_reg = mean(rating - mu_hat - b_hat_m_reg))

# Relationship btw mu_hat, b_hat_m_reg & b_hat_u_reg -> Y_hat u,m = mu_hat + b_hat_m_reg + b_hat_u_reg
predictions <- edx_subset %>% left_join(movie_reg_avgs, by = "movieId") %>% left_join(user_reg_avgs, by = "userId") %>% mutate(pred = mu_hat + b_hat_m_reg + b_hat_u_reg) %>% .$pred
rmse_4 <- RMSE(edx_subset$rating, predictions)
rmse_results <- bind_rows(rmse_results, tibble(method = "By reg. movie & reg. user based average", RMSE = round(rmse_4, digits = 3)))

# top 10 best movies with regularized bias - impact by movie
edx_subset %>% count(movieId) %>% left_join(movie_reg_avgs, by = "movieId") %>% left_join(movie_titles, by = "movieId") %>% arrange(desc(b_hat_m_reg)) %>% select(title, b_hat_m_reg, n) %>% slice(1:10) %>% knitr::kable()

# top 10 worst movies with regularized bias - impact by movie
edx_subset %>% count(movieId) %>% left_join(movie_reg_avgs, by = "movieId") %>% left_join(movie_titles, by = "movieId") %>% arrange(b_hat_m_reg) %>% select(title, b_hat_m_reg, n) %>% slice(1:10) %>% knitr::kable()

# top 10 best movies with regularized bias - impact by user
user_reg_avgs %>% left_join(edx_subset, by = "userId") %>% left_join(movie_titles, by = "movieId") %>% arrange(desc(b_hat_u_reg)) %>% select(title, b_hat_u_reg) %>% slice(1:10) %>% knitr::kable()

# top 10 worst movies with regularized bias - impact by user
user_reg_avgs %>% left_join(edx_subset, by = "userId") %>% left_join(movie_titles, by = "movieId") %>% arrange(b_hat_u_reg) %>% select(title, b_hat_u_reg) %>% slice(1:10) %>% knitr::kable()

# Though it seems regularization didn't make a huge improvements for user effects, it did for movie effects as we see the movies like Godfather, The (1972) and Shawshank Redemption, The (1994) are grouped together with similar ranks

#----------Prediction Modeling ends

#----------Prediction on Validation starts
validation_subset <- validation %>% select(movieId, userId)
validation_ratings <- validation$rating

predictions <- validation_subset %>% left_join(movie_reg_avgs, by = "movieId") %>% left_join(user_reg_avgs, by = "userId") %>% mutate(pred = mu_hat + b_hat_m_reg + b_hat_u_reg) %>% .$pred
rmse_5 <- RMSE(validation_ratings, predictions)
rmse_results <- bind_rows(rmse_results, tibble(method = "Reg. effects on validation set", RMSE = round(rmse_5, digits = 3)))
#----------Prediction on Validation ends

# The summary of RMSE results would show how the RMSE has been reduced through out this process
rmse_results

rm(dl, ratings, movies, test_index, temp, movielens, removed)
