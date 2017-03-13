## One-way LSTM

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


## One-way LSTM with receptors with small proportions

Model: LSTM at each step trained on ~100 receptors with “big” proportions (count >= 4), and ~28 receptors with “small” proportions (count < 3); all of them randomly sampled from the input data.

File: models/model.lstm128.bn.dense.mix.hdf5

Changed loss to MSE, quite helpful.

After ~180 iterations:
```
Predict big proportions:
        real            pred
[[  5.0744682    2.11004472]
 [  5.50443683   2.11004472]
 [  6.25304755  10.26939869]
 [  6.33391      0.        ]
 [  6.38307337   0.        ]
 [  6.37860409  10.14752769]
 [  6.5170848   10.24857044]
 [  6.5273809   10.22775173]
 [  6.60817853  10.14358807]
 [  6.61521585  10.2518549 ]
 [  6.71463049   0.        ]
 [  6.94914894  10.16672897]
 [  6.92190991  10.25177383]
 [  6.92383114  10.26767731]
 [  7.0442619   10.24433613]
 [  7.17130679  10.18964672]
 [  7.16884676  10.24189472]
 [  7.17130679   9.93906689]
 [  7.21923525  10.27545071]] 

Predict small proportions:
        real            pred
[[ 12.07904766  10.27740002]
 [ 12.07904766  10.28021336]
 [ 12.07904766  10.29475117]
 [ 12.07904766  10.25755119]
 [ 12.07904766  10.21598053]
 [ 12.07904766  10.22934055]
 [ 12.07904766  10.28066349]
 [ 12.48451277  10.24413586]
 [ 12.48451277  10.27627373]
 [ 12.07904766  10.24505806]
 [ 12.07904766  10.30801773]
 [ 12.48451277  10.29165173]
 [ 12.07904766  10.26035595]
 [ 12.07904766  10.2427845 ]
 [ 12.48451277  10.28197098]
 [ 12.07904766  10.27839661]
 [ 12.07904766  10.28106499]
 [ 13.17765995  10.29710865]
 [ 12.07904766  10.29539394]
 [ 12.07904766  10.20482826]]
```