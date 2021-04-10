if (getwd() != "E:/class/statistic/BIO384K/repeat"){
  setwd("E:/class/statistic/BIO384K/repeat")
}
# https://www.kaggle.com/jonathanbouchet/forecasting-beijing-s-housing
getwd()
library(dplyr)
library(tidyverse)
library(reshape2)
library(visdat)
library(lubridate)
library(ggplot2)
library(viridis)
library(psych)
library(caret)
library(groupdata2)
library(superml)
library(randomForest)
library(ggpubr)
theme_set(theme_pubr())



list_to_df = function(missing, columns_name){
  df = data.frame(matrix(unlist(missing), nrow = length(missing), byrow = TRUE))
  names(df) = columns_name
  return(df)
}



outlier_removal = function(df, col_name){
  q = quantile(df[[col_name]], c(0.25, 0.75))
  q1 = q[[1]]
  q3 = q[[2]]
  iqr = q3 - q1
  # df = df %>% filter((!!as.symbol(col_name) < q3 + 1.5 * iqr) & (!!as.symbol(col_name) > q1 - 1.5 * iqr))
  return(c(col_name, q1 - 1.5*iqr, q3 + 1.5 * iqr))
}

RMSE = function(m, o){
  sqrt(mean((m - o)^2))
}

# log1p(df$price)
# expm1(log1p(df$price))



df = read.csv('./resources/house_price.csv', sep = ',', stringsAsFactors = FALSE, encoding = 'UTF-8')
df = replace(df, df=="nan", NA)
df = data.frame(df %>% dplyr::select(-url, -id, -Cid))
num_columns = c('livingRoom', 'bathRoom', 'drawingRoom', 'constructionTime')
for (i in num_columns){
  df[[i]] = as.numeric(df[[i]])
}
df$district = as.factor(df$district)
df$tradeTime = year(as.Date(df$tradeTime, format = "%Y-%m-%d"))
df[["floorHeight"]] = as.numeric(sapply(df$floor, function(x) strsplit(x,' ')[[1]][2]))
df[["floorType"]] = sapply(df$floor, function(x) strsplit(x,' ')[[1]][1])
lbl = LabelEncoder$new()
df$floorType = lbl$fit_transform(df$floorType)
df[['age']] = as.numeric(2019 - df$constructionTime)
df = as_tibble(df)
df = df %>% mutate(distance = sqrt((Lat - 39.9042)^2 + (Lng - 116.4074)^2)) %>% 
  mutate(livingRoom = pmax(1, pmin(livingRoom, 4)))
# vis_miss(df, warn_large_data = FALSE)
drop_columns = c("DOM", "floor", "constructionTime")
df = df[,!(names(df) %in% drop_columns)]
df = df %>% drop_na()

fact = c("district", "buildingType", "renovationCondition", "buildingStructure", "elevator",
         "fiveYearsProperty", 'subway', "renovationCondition")
for (i in fact){
  df[[i]] = as.factor(df[[i]]) 
}
num_factor = c("followers", "square", "age", "floorHeight", "distance", 
               "ladderRatio", "communityAverage", "totalPrice")
col_list = list()

for (i in 1:length(num_factor)){
  col_list[[i]] = outlier_removal(df, num_factor[[i]])
}
for (i in col_list){
  df = df %>% filter((!!as.symbol(i[[1]]) < as.numeric(i[[3]])) & 
                       (!!as.symbol(i[[1]]) > as.numeric(i[[2]])))
}
df = df %>% filter(price >= 14000 )
count(df)[['n']]
drop_columns = c("DOM", "floor", "constructionTime", "totalPrice", "kitchen", "bathRoom", "drawingRoom", "totalPrice")
df = df[,!(names(df) %in% drop_columns)]
df$price_t = log1p(df$price)
length(names(df)) - 2
a = ggplot(df, aes(x = price)) + geom_histogram(binwidth = 30) + ggtitle("Histogram of price")
b = ggplot(df, aes(x = price_t)) + geom_histogram(binwidth = 0.02) + ggtitle("Histogram of log1p transformed price")
ggarrange(a, b, labels = c('a', 'b'), ncol = 2, nrow = 1)

a = ggplot(df, aes(x = Lat, y = Lng, color = price)) + geom_point()+
    scale_color_viridis(option = "D") + ggtitle("Plot of price with location")

b = ggplot(df, aes(x = Lat, y = Lng, color = age)) + geom_point()+
   scale_color_viridis(option = "D") + ggtitle("Plot of location with age")
ggarrange(a, b, labels = c('a', 'b'), ncol = 2, nrow = 1)
a = ggplot(df, aes(x = district, y=price)) +
  geom_boxplot()+ ggtitle("Plot of price with district")
b = ggplot(df, aes(x = buildingType, y=price)) +
  geom_boxplot()+ ggtitle("Plot of price with buildingType")
c = ggplot(df, aes(x = buildingType, y=square)) +
  geom_boxplot()+ ggtitle("Plot of square with buildingType")
ggarrange(a, ggarrange(b,c, ncol = 2, labels = c('b', 'c')), labels = 'a', nrow = 2)

fact = c("district", "buildingType", "renovationCondition", "buildingStructure", "elevator",
         "fiveYearsProperty", 'subway')
num_factor = c("followers", "square", "age", "floorHeight", "distance", "ladderRatio", "communityAverage")
for (i in num_factor){
  df[[i]] = scale(df[[i]])
}
df = as.data.frame(cbind(df %>% select_if(is.numeric),
                         "district" = dummy.code(df$district),
                         "buildingType" = dummy.code(df$buildingType),
                         "renovationCondition" = dummy.code(df$renovationCondition),
                         "elevator" = dummy.code(df$elevator),
                         "subway" = dummy.code(df$subway),
                         "fiveYearsProperty" = dummy.code(df$fiveYearsProperty),
                         "floorType" = dummy.code(df$floorType),
                         "buildingStructure" = dummy.code(df$buildingStructure)))
pr = prcomp(df[,!(names(df) %in% c("price"))], center = TRUE, scale = TRUE)
length(names(df)) - 2
cum_pr <- as.data.frame(cumsum(pr$sdev^2 / sum(pr$sdev^2)))
names(cum_pr) = c("cum_variance")
cum_pr$index = as.numeric(row.names(cum_pr))
ggplot(cum_pr, aes(x = index, y = cum_variance)) + geom_line() + xlab("number of components") + 
  ylab("cumulative explained variance")
df = fold(df, k = 5)
train = df[df$.folds != 5,names(df)[1:length(names(df))-1]]
train = train[,!(names(train) %in% c('price'))]
test = df[df$.folds == 5,names(df)[1:length(names(df))-1]]
test = test[,!(names(test) %in% c('price'))]
rf = randomForest(price_t ~ ., data = train, ntree = 900, maxnodes = 20, nodesize = 10)
testing = df[,names(df)[1:length(names(df))-1]]
testing = testing[,!(names(testing) %in% c('price', "price_t"))]
predictions = predict(rf, newdata = testing)
df_p = as.data.frame(cbind(df, predictions))
names(df_p)[[length(names(df_p))]] = "rf_prediction"

library(Metrics)
RMSE(df_p$price_t, df_p$rf_prediction)
rmsle(df_p[df_p$.folds != 5,]$price, df_p[df_p$.folds != 5,]$rf_prediction)
library(xgboost)
params = list(objective = "reg:linear", booster = "gbtree", max_depth=20, 
              min_child_weight=2,subsample=1, colsample_bytree=0.8, 
              learning_rate = 0.1, reg_lambda = 0.45, reg_alpha = 0, gamma = 0.5)
names(train)
features = train[,!(names(train) %in% c('price_t'))]
t_features = test[,!(names(test) %in% c('price_t'))]
dtrain = xgb.DMatrix(data = as.matrix(features), label= train$price_t)
dtest = xgb.DMatrix(data = as.matrix(t_features), label = test$price_t)
xgb_model = xgb.train(data = dtrain, params=params, nrounds = 2000, early_stopping_rounds = 50, 
                      watchlist = list(train = dtrain, test = dtest))
testing = df[,names(df)[1:length(names(df))-1]]
testing = testing[,!(names(testing) %in% c('price', 'price_t'))]
dpredict = xgb.DMatrix(data = as.matrix(testing))
prediction = predict(xgb_model, dpredict)
df_p = as.data.frame(cbind(df_p, prediction))
names(df_p)[[length(names(df_p))]] = 'xgb_prediction'
names(df_p)
df_test = df_p[df_p$.folds == 5,]
RMSE(df_p$price_t, df_p$xgb_prediction)
RMSE(df_test$price_t, df_test$xgb_prediction)
install.packages("lightgbm", repos = "https://cran.r-project.org")
library(lightgbm)
lgb_train = lgb.Dataset(data = as.matrix(features), label = train$price_t)
lgb_test = lgb.Dataset(data = as.matrix(t_features), label = test$price_t)
params = list(learning_rate = 0.15, boosting_type = 'gbdt', objective = "regression",
               metric = 'rmse', min_child_weight = 2, num_leaves = 36, 
               colsample_bytree = 0.8, reg_lambda = 0.4)
lgb_model = lgb.train(params = params, data = lgb_train, nrounds = 2000, 
                      early_stopping_rounds = 50, eval = list(valid = lgb_test))
ggplot(df_p[df_p$.folds == 5,], aes(x = prediction, y = price)) + geom_point()


df_train = as_tibble(read.csv('./resources/train_result.csv'))
df_test = as_tibble(read.csv('./resources/test_result.csv'))
df_r = as_tibble(read.csv('./resources/rmse_result.csv'))
ele = element_text(size = 10)
a = ggplot(df_train, aes(y = target, x = Random_Forest)) + geom_point() + 
  ggtitle("Training result of random forest") + xlab("predicted price") + ylab("Actual price") + 
  theme(plot.title = ele, axis.text = ele)
b = ggplot(df_test, aes(y = target, x = Random_Forest)) + geom_point() + 
  ggtitle("Test result of random forest") + xlab("predicted price") + 
  ylab("Actual price") +  theme(plot.title = ele, axis.text = ele)
c = ggplot(df_train, aes(y = target, x = Random_Forest_grid_search)) + geom_point() + 
  ggtitle("Training result of grid searched random forest") + xlab("predicted price") + 
  ylab("Actual price") +  theme(plot.title = ele, axis.text = ele)
d = ggplot(df_test, aes(y = target, x = Random_Forest_grid_search)) + geom_point() + 
  ggtitle("Test result of grid searched random forest") + xlab("predicted price") + 
  ylab("Actual price") +  theme(plot.title = ele, axis.text = ele)
ggarrange(a,b,c,d, labels = c('a', 'b', 'c', 'd'), ncol = 2, nrow = 2)
a = ggplot(df_train, aes(y = target, x = lightGBM)) + geom_point() + 
  ggtitle("Training result of lightGBM") + xlab("predicted price") + ylab("Actual price") + 
  theme(plot.title = ele, axis.text = ele)
b = ggplot(df_test, aes(y = target, x = lightGBM)) + geom_point() + 
  ggtitle("Test result of lightGBM") + xlab("predicted price") + 
  ylab("Actual price") +  theme(plot.title = ele, axis.text = ele)
c = ggplot(df_train, aes(y = target, x = LightGBM_grid_search)) + geom_point() + 
  ggtitle("Training result of grid searched lightGBM") + xlab("predicted price") + 
  ylab("Actual price") +  theme(plot.title = ele, axis.text = ele)
d = ggplot(df_test, aes(y = target, x = LightGBM_grid_search)) + geom_point() + 
  ggtitle("Test result of grid searched lightGBM") + xlab("predicted price") + 
  ylab("Actual price") +  theme(plot.title = ele, axis.text = ele)
ggarrange(a,b,c,d, labels = c('a', 'b', 'c', 'd'), ncol = 2, nrow = 2)
a = ggplot(df_train, aes(y = target, x = xgboost)) + geom_point() + 
  ggtitle("Training result of xgboost") + xlab("predicted price") + ylab("Actual price") + 
  theme(plot.title = ele, axis.text = ele)
b = ggplot(df_test, aes(y = target, x = xgboost)) + geom_point() + 
  ggtitle("Test result of xgboost") + xlab("predicted price") + 
  ylab("Actual price") +  theme(plot.title = ele, axis.text = ele)
c = ggplot(df_train, aes(y = target, x = XGboost_grid_search)) + geom_point() + 
  ggtitle("Training result of grid searched xgboost") + xlab("predicted price") + 
  ylab("Actual price") +  theme(plot.title = ele, axis.text = ele)
d = ggplot(df_test, aes(y = target, x = XGboost_grid_search)) + geom_point() + 
  ggtitle("Test result of grid searched xgboost") + xlab("predicted price") + 
  ylab("Actual price") +  theme(plot.title = ele, axis.text = ele)
ggarrange(a,b,c,d, labels = c('a', 'b', 'c', 'd'), ncol = 2, nrow = 2)

a = ggplot(df_train, aes(y = target, x = hybrid_regrission)) + geom_point() + 
  ggtitle("Training result of hybrid_regrission") + xlab("predicted price") + ylab("Actual price") + 
  theme(plot.title = ele, axis.text = ele)
b = ggplot(df_test, aes(y = target, x = hybrid_regrission)) + geom_point() + 
  ggtitle("Test result of hybrid_regrission") + xlab("predicted price") + 
  ylab("Actual price") +  theme(plot.title = ele, axis.text = ele)
c = ggplot(df_train, aes(y = target, x = hybrid_regrission_grid_search)) + geom_point() + 
  ggtitle("Training result of grid searched hybrid_regrission") + xlab("predicted price") + 
  ylab("Actual price") +  theme(plot.title = ele, axis.text = ele)
d = ggplot(df_test, aes(y = target, x = hybrid_regrission_grid_search)) + geom_point() + 
  ggtitle("Test result of grid searched hybrid_regrission") + xlab("predicted price") + 
  ylab("Actual price") +  theme(plot.title = ele, axis.text = ele)
ggarrange(a,b,c,d, labels = c('a', 'b', 'c', 'd'), ncol = 2, nrow = 2)

a = ggplot(df_train, aes(y = target, x = stack_generation)) + geom_point() + 
  ggtitle("Training result of stack_generation") + xlab("predicted price") + ylab("Actual price") + 
  theme(plot.title = ele, axis.text = ele)
b = ggplot(df_test, aes(y = target, x = stack_generation)) + geom_point() + 
  ggtitle("Test result of stack_generation") + xlab("predicted price") + 
  ylab("Actual price") +  theme(plot.title = ele, axis.text = ele)
c = ggplot(df_train, aes(y = target, x = stack_generation_grid_search)) + geom_point() + 
  ggtitle("Training result of grid searched stack_generation") + xlab("predicted price") + 
  ylab("Actual price") +  theme(plot.title = ele, axis.text = ele)
d = ggplot(df_test, aes(y = target, x = stack_generation_grid_search)) + geom_point() + 
  ggtitle("Test result of grid searched stack_generation") + xlab("predicted price") + 
  ylab("Actual price") +  theme(plot.title = ele, axis.text = ele)
ggarrange(a,b,c,d, labels = c('a', 'b', 'c', 'd'), ncol = 2, nrow = 2)


df_r
df_r["test_rmsle_grid_search"] = 0.0
df_r["train_rmsle_grid_search"] = 0.0
df_r[1, "train_rmsle_grid_search"] = df_r[4, "train_rmsle"]
df_r[2, "train_rmsle_grid_search"] = df_r[5, "train_rmsle"]
df_r[3, "train_rmsle_grid_search"] = df_r[6, "train_rmsle"]
df_r[7, "train_rmsle_grid_search"] = df_r[8, "train_rmsle"]
df_r[9, "train_rmsle_grid_search"] = df_r[10, "train_rmsle"]
df_r[1, "test_rmsle_grid_search"] = df_r[4, "test_rmsle"]
df_r[2, "test_rmsle_grid_search"] = df_r[5, "test_rmsle"]
df_r[3, "test_rmsle_grid_search"] = df_r[6, "test_rmsle"]
df_r[7, "test_rmsle_grid_search"] = df_r[8, "test_rmsle"]
df_r[9, "test_rmsle_grid_search"] = df_r[10, "test_rmsle"]
df_r = df_r[c(1,2,3,7,9),]
df_r
