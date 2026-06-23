---
title: "Cross Validation and Tuning"
teaching: 45
exercises: 30
---

:::::::::::::::::::::::::::::::::::::: questions

- How can the fit of an XGBoost model be improved?
- What is cross validation?
- What are some guidelines for tuning parameters in a machine learning algorithm?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Explore the effects of adjusting the XGBoost parameters.
- Practice some coding techniques to tune parameters using grid searching and cross validation.

::::::::::::::::::::::::::::::::::::::::::::::::


## Parameter Tuning

Like many other machine learning algorithms, XGBoost has an assortment of parameters that control the behavior of the training process. (These parameters are sometimes referred to as *hyperparameters*, because they cannot be directly estimated from the data.) To improve the fit of the model, we can adjust, or *tune*, these parameters. According to the [notes on parameter tuning](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html) in the XGBoost documentation, "[p]arameter tuning is a dark art in machine learning," so it is difficult to prescribe an automated process for doing so. In this episode we will develop some coding practices for tuning an XGBoost model, but be advised that the optimal way to tune a model will depend heavily on the given data set.

You can find a complete list of XGBoost parameters in [the documentation](https://xgboost.readthedocs.io/en/stable/parameter.html). Generally speaking, each parameter controls the complexity of the model in some way. More complex models tend to fit the training data more closely, but such models can be very sensitive to small changes in the training set. On the other hand, while less complex models can be more conservative in this respect, they have a harder time modeling intricate relationships. The "art" of parameter tuning lies in finding an appropriately complex model for the problem at hand.

A complete discussion of the issues involved are beyond the scope of this lesson. An excellent resource on the topic is [An Introduction to Statistical Learning](https://www.stat.berkeley.edu/users/rabbee/s154/ISLR_First_Printing.pdf), by James, Witten, Hastie, and Tibshirani. In particular, Section 2.2.2 discusses the *Bias-Variance Trade-Off* inherent in statistical learning methods.

## Cross Validation

How will we be able to tell if an adjustment to a parameter has improved the model? One possible approach would be to test the model before and after the adjustment on the testing set. However, the problem with this method is that it runs the risk of tuning the model to the particular properties of the testing set, rather than to general future cases that we might encounter. It is better practice to save our testing set until the very end of the process, and then use it to test the accuracy of our model. Training set accuracy, as we have seen, tends to underestimate the accuracy of a machine learning model, so tuning to the training set may also fail to make improvements that generalize.

An alternative testing procedure is to use *cross validation* on the training set to judge the effect of tuning adjustments.  In *k*-fold cross validation, the training set is partitioned randomly into *k* subsets. Each of these subsets takes a turn as a testing set, while the model is trained on the remaining data. The accuracy of the model is then measured *k* times, and the results are averaged to obtain an estimate of the overall model performance. In this way we can be more certain that repeated adjustments will be tested in ways that generalize to future observations. It also allows us to save the original testing set for a final test of our tuned model.

## Revisit the Red Wine Quality Model

Let's see if we can improve the previous episode's model for predicting red wine quality.

```r
library(tidyverse)
library(here)
library(xgboost)
wine <- read_csv(here("data", "wine.csv"))
redwine <- wine |> dplyr::slice(1:1599)
trainSize <- round(0.80 * nrow(redwine))
set.seed(1234)
trainIndex <- sample(nrow(redwine), trainSize)
trainDF <- redwine |> dplyr::slice(trainIndex)
testDF <- redwine |> dplyr::slice(-trainIndex)
dtrain <- xgb.DMatrix(data = as.matrix(select(trainDF, -quality)), label = trainDF$quality)
dtest <- xgb.DMatrix(data = as.matrix(select(testDF, -quality)), label = testDF$quality)
```

The `xgb.cv` command handles most of the details of the cross validation process. Since this is a random process, we will set a seed value for reproducibility. We will use 10 folds and the default value of 0.3 for `eta`.

```r
set.seed(524)
rwCV <- xgb.cv(params = list(eta = 0.3),
               data = dtrain,
               nfold = 10,
               nrounds = 500,
               early_stopping_rounds = 10,
               print_every_n = 5)
```

The output appears similar to the `xgb.train` command. Notice that each error estimate now includes a standard deviation, because these estimates are formed by averaging over all ten folds. The function returns a list,  which we have given the name `rwCV`. Its `names` hint at what each list item represents.

```r
names(rwCV)
```

::::::::::::::::::::::::::::::::::::: challenge

## Challenge: Examine the cross validation results

1. Examine the list item `rwCV$folds`.
   What do suppose these numbers represent?
   Are all the folds the same size? Can you explain why/why not?
2. Display the evaluation log with rows sorted by `test_rmse_mean`.
3. How could you display only the row containing the best iteration?

:::::::::::::::::::::::: solution

1. The numbers are the indexes of the rows in each fold. The folds are not
   all the same size, because no row can be used more than once, and there
   are 1279 rows total in the training set, so they don't divide evenly into
   10 partitions.

2.

```r
rwCV$evaluation_log |> 
  arrange(test_rmse_mean)
```

```output
     iter train_rmse_mean train_rmse_std test_rmse_mean test_rmse_std
    <int>           <num>          <num>          <num>         <num>
 1:    28       0.2320170    0.010740120      0.6033746    0.05477015
 2:    34       0.1981800    0.011213834      0.6037715    0.05344829
 3:    27       0.2386391    0.010507210      0.6039262    0.05479807
 4:    33       0.2024958    0.011358906      0.6039311    0.05478204
 5:    32       0.2089829    0.010361427      0.6041730    0.05461273
 6:    30       0.2210709    0.010046552      0.6041879    0.05581148
 7:    29       0.2260435    0.010869279      0.6042728    0.05585093
 8:    25       0.2501837    0.007824473      0.6044555    0.05502287
 9:    31       0.2147514    0.010094178      0.6046705    0.05532650
10:    18       0.2944292    0.011298599      0.6046983    0.05171586
11:    17       0.3024783    0.009436997      0.6047324    0.05171553
12:    26       0.2437953    0.008537872      0.6048376    0.05618227
13:    36       0.1865345    0.010073930      0.6048411    0.05486732
14:    24       0.2572901    0.008176900      0.6049043    0.05452374
15:    23       0.2640588    0.007058633      0.6050176    0.05367270
16:    21       0.2754504    0.008534985      0.6053905    0.05337158
17:    20       0.2831430    0.009797559      0.6053954    0.05321445
18:    22       0.2689525    0.008305594      0.6055627    0.05279062
19:    35       0.1911810    0.010811872      0.6056385    0.05344183
20:    19       0.2892988    0.009632057      0.6057670    0.05296946
21:    16       0.3106484    0.008904877      0.6058135    0.05223975
22:    15       0.3185064    0.008491525      0.6061559    0.05157840
23:    13       0.3374284    0.011892523      0.6061768    0.04929542
24:    14       0.3278956    0.010382923      0.6063122    0.05134600
25:    37       0.1809015    0.010812019      0.6064709    0.05395293
26:    12       0.3477349    0.009835170      0.6065789    0.04917598
27:    38       0.1772849    0.011664037      0.6066433    0.05355217
28:    11       0.3566206    0.007410271      0.6077226    0.04832899
29:    10       0.3701111    0.007290606      0.6084441    0.04625580
30:     9       0.3867212    0.006981635      0.6103006    0.04333710
31:     8       0.4008058    0.006912524      0.6119924    0.04180357
32:     7       0.4200004    0.007091309      0.6151725    0.04006364
33:     6       0.4426349    0.004755571      0.6189888    0.03530737
34:     5       0.4703238    0.005615681      0.6226820    0.03441598
35:     4       0.5009748    0.004684828      0.6295255    0.03169689
36:     3       0.5432452    0.005389668      0.6464929    0.03307930
37:     2       0.6017115    0.005299966      0.6714706    0.03143677
38:     1       0.6839483    0.004493076      0.7217167    0.03310415
     iter train_rmse_mean train_rmse_std test_rmse_mean test_rmse_std
    <int>           <num>          <num>          <num>         <num>
```

3.

```r
rwCV$evaluation_log |> 
  arrange(test_rmse_mean) |> 
  head(1)
```

```output
    iter train_rmse_mean train_rmse_std test_rmse_mean test_rmse_std
   <int>           <num>          <num>          <num>         <num>
1:    28        0.232017     0.01074012      0.6033746    0.05477015
```

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

## Repeat Cross Validation in a Loop

To expedite the tuning process, it helps to design a loop to run repeated cross validations on different parameter values. We can start by choosing a value of `eta` from a list of candidate values.

```r
paramDF <- tibble(eta = c(0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4))
```

The following command converts a data frame to a list of lists. This format is needed because `xgb.cv()` expects model parameters as a list. The `split` function splits `paramDF` into a list of its rows, and then the `lapply` function converts each row to a list. Each item of `paramlist` will be a list giving a valid parameter setting that we can use in the `xgb.cv` function.

```r
paramList <- lapply(split(paramDF, 1:nrow(paramDF)), as.list)
```

Now we will write a loop that will perform a different cross validation for each parameter setting in the `paramList` list. We'll keep track of the best iterations in the `bestResults` tibble. To avoid too much printing, we set `verbose = FALSE` and use a `txtProgressBar` to keep track of our progress. On some systems, it may be necessary to use `gc()` to prevent running out of memory.  

This loop will take some time to run (about 10 minutes depending on your machine).  For a shorter example, use:


```r
paramDF <- tibble(eta = c(0.05, 0.1, 0.3))
paramList <- lapply(split(paramDF, 1:nrow(paramDF)), as.list)
```

```r
bestResults <- tibble()
set.seed(708)
pb <- txtProgressBar(style = 3) 
for(i in seq(length(paramList))) {
  rwCV <- xgb.cv(params = paramList[[i]], 
                 data = dtrain, 
                 nrounds = 500, 
                 nfold = 10,
                 early_stopping_rounds = 10,
                 verbose = FALSE)
  bestResults <- bestResults |>  
    bind_rows(rwCV$evaluation_log |> 
                arrange(test_rmse_mean) |> 
                head(1))
  gc() # Free unused memory after each loop iteration
  setTxtProgressBar(pb, i/length(paramList))
}
close(pb) # done with the progress bar
```

We now have all of the best iterations in the `bestResults` data frame, which we can combine with the data frame of parameter values.

```r
etasearch <- bind_cols(paramDF, bestResults)
```

In RStudio, it is convenient to use `View(etasearch)` to view the results in a separate tab. We can use the RStudio interface to sort by `mean_test_rmse`.

Note that there is not much difference in `mean_test_rmse` among the best three choices. As we have seen in the previous episode, the choice of `eta` typically involves a trade-off between speed and accuracy. A common approach is to pick a reasonable value of `eta` and then stick with it for the rest of the tuning process. Let's use `eta` = 0.1, because it uses about half as many steps as `eta` = 0.05, and the accuracy is comparable.

## Grid Search

Sometimes it helps to tune a pair of related parameters together. A *grid search* runs through all possible combinations of candidate values for a selection of parameters.

We will tune the parameters `max_depth` and `max_leaves` together. These both affect the size the trees that the algorithm grows. Deeper trees with more leaves make the model more complex. We use the `expand.grid` function to store some reasonable candidate values in `paramDF`.

```r
paramDF <- expand.grid(
  max_depth = seq(15, 29, by = 2),
  max_leaves = c(63, 127, 255, 511, 1023, 2047, 4095),
  eta = 0.1)
```

If you `View(paramDF)` you can see that we have 56 different parameter choices to run through. The rest of the code is the same as before, but this loop might take a while to execute.

For an example that takes less time to execute use: 

```r
paramDF <- expand.grid(
  max_depth = seq(15, 21, by = 2),
  max_leaves = c(63, 127, 255),
  eta = 0.1)
```

```r
paramList <- lapply(split(paramDF, 1:nrow(paramDF)), as.list)
bestResults <- tibble()
set.seed(312)
pb <- txtProgressBar(style = 3)
for(i in seq(length(paramList))) {
  rwCV <- xgb.cv(params = paramList[[i]],
                 data = dtrain,
                 nrounds = 500,
                 nfold = 10,
                 early_stopping_rounds = 10,
                 verbose = FALSE)
  bestResults <- bestResults |>  
    bind_rows(rwCV$evaluation_log |> 
                arrange(test_rmse_mean) |> 
                head(1))
  gc()
  setTxtProgressBar(pb, i/length(paramList))
}
close(pb)
depth_leaves <- bind_cols(paramDF, bestResults)
```

When we `View(depth_leaves)` we see that a choice of `max_depth` = 19 and `max_leaves` = 63 results in the best `test_rmse_mean`. One caveat is that cross validation is a random process, so running this code with a different random seed may very well produce a different result. 

::::::::::::::::::::::::::::::::::::: challenge

## Challenge: Write a Grid Search Function

Instead of repeatedly using the above code block, let's package it into
an R function.
Define a function called `GridSearch` that consumes a data frame `paramDF`
of candidate parameter values
and an `xgb.DMatrix` `dtrain` of training data. The function should
return a data frame combining the columns of `paramDF` with the
corresponding results of the best cross validation iteration. The returned
data frame should be sorted in ascending order of `test_rmse_mean`.

:::::::::::::::::::::::: solution

```r
GridSearch <- function(paramDF, dtrain) {
  paramList <- lapply(split(paramDF, 1:nrow(paramDF)), as.list)
  bestResults <- tibble()
  pb <- txtProgressBar(style = 3)
  for(i in seq(length(paramList))) {
    rwCV <- xgb.cv(params = paramList[[i]],
                   data = dtrain,
                   nrounds = 500,
                   nfold = 10,
                   early_stopping_rounds = 10,
                   verbose = FALSE)
  bestResults <- bestResults |>  
    bind_rows(rwCV$evaluation_log |> 
                arrange(test_rmse_mean) |> 
                head(1))
    gc()
    setTxtProgressBar(pb, i/length(paramList))
  }
  close(pb)
  return(bind_cols(paramDF, bestResults) |> arrange(test_rmse_mean))
}
```

Check the function on a small example.

```r
set.seed(630)
GridSearch(tibble(eta = c(0.3, 0.2, 0.1)), dtrain)
```

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

## Adding Random Sampling

Adding random sampling to the training process can help make the model less dependent on the training set, and hopefully more accurate when generalizing to future cases. In XGBoost, the two parameters `subsample` and `colsample_bytree` will grow trees based on a random sample of a specified percentage of rows and columns, respectively. Typical values for these parameters are between 0.5 and 1.0 (where 1.0 implies that no random sampling will be done), but as this will take some time to execute, we'll use a smaller range in this example.

::::::::::::::::::::::::::::::::::::: challenge

## Challenge: Tune Row and Column Sampling

Use a grid search to tune the parameters `subsample` and `colsample_bytree`.
Choose candidate values between 0.7 and 0.9. 

Use our previously chosen values
of `eta`, `max_depth`, and `max_leaves`.

:::::::::::::::::::::::: solution

```r
paramDF <- expand.grid(
  subsample = seq(0.7, 0.9, by = 0.1),
  colsample_bytree = seq(0.7, 0.9, by = 0.1),
  max_depth = 19,
  max_leaves = 63,
  eta = 0.1)
set.seed(848)
randsubsets <- GridSearch(paramDF, dtrain)
```

It appears that some amount of randomization helps.  It looks like `subsample` = 0.8 and `colsample_bytree` = 0.9 results in the lowest `test_rmse_mean`. 

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

## Final Check using the Testing Set

Once a model has been tuned using the training set and cross validation, it can  be tested using the testing set. Note that we have not used the testing set in any of our tuning experiments, so the testing set accuracy should give a fair assessment of the accuracy of our tuned model relative to the other models we have explored.

We give parameters `max_depth`, `max_leaves`, `subsample`, and `colsample_bytree` the values that we chose during the tuning process. Since we only have to do one training run, a smaller learning rate won't incur much of a time penalty, so we set `eta` = 0.05.

```r
set.seed(805)
rwMod <- xgb.train(
  data   = dtrain,
  verbose = FALSE,
  evals  = list(train = dtrain, test = dtest),
  nrounds = 10000,
  early_stopping_rounds = 50,
  params = list(
    max_depth        = 19,
    max_leaves       = 63,
    subsample        = 0.8,
    colsample_bytree = 0.9,
    eta              = 0.05
  )
)

print(rwMod)

elog <- attr(rwMod, "evaluation_log")
elog[which.min(elog$test_rmse), ]

attr(rwMod, "evaluation_log") |> 
  pivot_longer(cols = c(train_rmse, test_rmse), names_to = "RMSE") |>
  ggplot(aes(x = iter, y = value, color = RMSE)) + 
  geom_line()
```

After some tuning, our testing set RMSE is down to 0.60, which is an improvement over the previous episode and over the RMSE we obtained using the random forest model.

::::::::::::::::::::::::::::::::::::: challenge

## Challenge: Improve the White Wine Model

Improve your XGBoost model for the white wine data (rows 1600-6497) of the
`wine` data frame. Use grid searches to tune several parameters, using only
the training set during the tuning process. Can you improve the testing set
RMSE over the white wine challenges from the previous two episodes?

:::::::::::::::::::::::: solution
Results may vary. The proposed solution below will take quite some time
to execute.

```r
whitewine <- wine |> dplyr::slice(1600:6497)
trainSize <- round(0.80 * nrow(whitewine))
set.seed(1234)
trainIndex <- sample(nrow(whitewine), trainSize)
trainDF <- whitewine |> dplyr::slice(trainIndex)
testDF <- whitewine |> dplyr::slice(-trainIndex)
dtrain <- xgb.DMatrix(data = as.matrix(select(trainDF, -quality)),
                      label = trainDF$quality)
dtest <- xgb.DMatrix(data = as.matrix(select(testDF, -quality)),
                     label = testDF$quality)
```

Start by tuning `max_depth` and `max_leaves` together.

```r
paramGrid <- expand.grid(
  max_depth = seq(10, 40, by = 2),
  max_leaves = c(15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191),
  eta = 0.1
)
set.seed(1981)
ww_depth_leaves <- GridSearch(paramGrid, dtrain)
```

There are several options that perform similarly. Let's choose
`max_depth` = 14 along with `max_leaves` = 127. Now we tune the
two random sampling parameters together.

```r
paramGrid <- expand.grid(
  subsample = seq(0.5, 1, by = 0.1),
  colsample_bytree = seq(0.5, 1, by = 0.1),
  max_depth = 14,
  max_leaves = 127,
  eta = 0.1
)
set.seed(867)
ww_randsubsets <- GridSearch(paramGrid, dtrain)
```

Again, some randomization seems to help, but there are several options.
We'll choose `subsample` = 0.8 and `colsample_bytree` = 0.9.
Finally, we train the model with the chosen parameters.

```r
set.seed(5309)
                   
ww_gbmod <- xgb.train(
  data   = dtrain,
  verbose = FALSE,
  evals  = list(train = dtrain, test = dtest),
  nrounds = 10000,
  early_stopping_rounds = 50,
  params = list(
    max_depth        = 14,
    max_leaves       = 127,
    subsample        = 0.8,
    colsample_bytree = 0.9,
    eta              = 0.1
  )
)
```

The tuned XGBoost model has a testing set RMSE of about 0.63, which is
better than the un-tuned model from the last episode (0.66),
and similar to the random forest model (0.63).

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- Parameter tuning can improve the fit of an XGBoost model.
- Cross validation allows us to tune parameters using the training set only, saving the testing set for final model evaluation.

::::::::::::::::::::::::::::::::::::::::::::::::
