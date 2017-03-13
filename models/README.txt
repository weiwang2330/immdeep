## Simple one-way LSTM
Model: LSTM on the first 30K data

File: models/model.lstm128.bn.dense.hdf5

After more than 100 iterations here are results for the first 20 receptors:
```
[[  5.0744682   11.84071827]
 [  5.22786273  11.8524065 ]
 [  5.50443683  11.84071827]
 [  6.25304755  11.89706039]
 [  6.33391     11.79355621]
 [  6.38307337  11.76680374]
 [  6.37860409  11.83020401]
 [  6.5170848   11.84064388]
 [  6.5273809   11.85927868]
 [  6.60817853  11.79740143]
 [  6.61521585  11.70992184]
 [  6.71463049  11.93083572]
 [  6.94914894  11.79218006]
 [  6.92190991  11.87408543]
 [  6.92383114  11.79092789]
 [  7.0442619   11.84865761]
 [  7.17130679  11.81984138]
 [  7.16884676  11.81451893]
 [  7.17130679  11.85754585]
 [  7.21923525  11.82061195]]
```

Despite quite low loss, the proportions are looking bad. It seems that the optimizer
"narrowed" the proportions in order to minimize loss (I checked - all proporions almost the same). 
I'm pretty sure that such narrowing is due to the very heavy tail of receptors with 
almost identical proportions (~ 11.8 in neg-log-scale).