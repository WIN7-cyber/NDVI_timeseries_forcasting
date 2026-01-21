1. Run traitement_ndvidata.ipynb: Linearly interpolate the NDVI data to obtain a regular time series with uniform time-step intervals.ndvi_precombine.nc:（time, y, x）

2. Run traitement_weatherdata.ipynb: The weather data are aggregated using averaging or summation and then aligned with the NDVI time steps (e.g., every 3 days, 5 days, or 16 days). If weather conditions vary across locations, the corresponding weather values are assigned to each pixel according to its predefined region. weather_precombine.nc

3. Run Combine.ipynb: Store the aligned data in a single file(combine.nc), then preprocess it into the model’s required input format.(in folder: data/arrays)

4. Run run_model_final



由于实验中数据量非常有限，所以对比并尝试了多种划分数据集方案：

最终版本run_model_final中， 按时间步顺序：

测试集：7 个连续时间步  
训练集：22 个连续时间步（取中间段）  
验证集：8 个连续时间步

这样做的目的是同时保证模型能学到完整生长周期的中段特征

Due to the very limited amount of data in the experiment, several dataset partitioning schemes were compared and tried:

In the final version `run_model_final`, the time step order is as follows:  
Test set: 7 consecutive time steps  
Training set: 22 consecutive time steps (taking the middle segment)  
Validation set: 8 consecutive time steps  

The purpose of this is to ensure that the model can learn the middle segment features of the complete growth cycle simultaneously.

 <img width="573" height="385" alt="image" src="https://github.com/user-attachments/assets/534618f5-570a-4ba7-84ca-469066715039" />

初始版本run_model_origin中， 按时间步顺序：

训练集：22 个连续时间步（取中间段）  
验证集：7 个连续时间步  
测试集：8 个连续时间步

训练集只覆盖了植物生长阶段，而验证集正好在衰退阶段。

In the initial version of run_model_origin, the time steps are arranged as follows:

Training set: 22 consecutive time steps (taking the middle segment)  
Validation set: 7 consecutive time steps  
Test set: 8 consecutive time steps

The training set only covers the plant growth stage, while the validation set covers the decay stage.
 <img width="688" height="460" alt="image" src="https://github.com/user-attachments/assets/1b1c199f-1a9c-4779-8911-faa3f37826d9" />

另一个版本run_model_rolling_train中：

训练集：29 个连续时间步（取中间段）  
验证集：8 个连续时间步

这是一个动态的训练集，第一折（fold）训练集：t0…t28，验证集t29…t36。这一折训练5个epoche之后，进入第二折，训练集变为t1…t29，验证集则是t30…t0继续训练。Fold2训练5个epoche后，进入第三折，训练集变为t2…t30，验证集则是t31…t1。以此类推，总共训练指定的fold数量。
本意是考虑到数据集是一段可以前后衔接起来的植物生长+衰败的过程，想要让模型学习到所有阶段，但是结果不尽如人意。

In another version of `run_model_rolling_train`:

Training set: 29 consecutive time steps (taking the middle segment)  
Validation set: 8 consecutive time steps

This is a dynamic training set. The first fold training set is t0…t28, and the validation set is t29…t36. After training for 5 epochs in this fold, the second fold begins, with the training set becoming t1…t29 and the validation set becoming t30…t0. After training for 5 epochs in Fold2, the third fold begins, with the training set becoming t2…t30 and the validation set becoming t31…t1. This continues for the specified number of folds.

The intention was to consider that the dataset represents a continuous process of plant growth and decay, aiming for the model to learn all stages, but the results were unsatisfactory.
<img width="712" height="393" alt="image" src="https://github.com/user-attachments/assets/935e2f93-1a58-4926-8617-b31e54353e5e" />

 ConvLSTM模型参考：https://github.com/kylalangotsky/ndvi_timeseries_forcasting

