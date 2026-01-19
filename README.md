实验中对比并尝试了多种方案：

最终版本run_model_final中， 按时间步顺序：

测试集：7 个连续时间步

训练集：22 个连续时间步（取中间段）

验证集：8 个连续时间步

这样做的目的是同时保证模型能学到完整生长周期的中段特征

 <img width="573" height="385" alt="image" src="https://github.com/user-attachments/assets/534618f5-570a-4ba7-84ca-469066715039" />

初始版本run_model_origin中， 按时间步顺序：

训练集：22 个连续时间步（取中间段）

验证集：7 个连续时间步

测试集：8 个连续时间步

训练集只覆盖了植物生长阶段，而验证集正好在衰退阶段。
 <img width="688" height="460" alt="image" src="https://github.com/user-attachments/assets/1b1c199f-1a9c-4779-8911-faa3f37826d9" />

另一个版本run_model_rolling_train中：

训练集：29 个连续时间步（取中间段）

验证集：8 个连续时间步

这是一个动态的训练集，第一折（fold）训练集：t0…t28，验证集t29…t36。这一折训练5个epoche之后，进入第二折，训练集变为t1…t29，验证集则是t30…t0继续训练。Fold2训练5个epoche后，进入第三折，训练集变为t2…t30，验证集则是t31…t1。以此类推，总共训练指定的fold数量。
本意是考虑到数据集是一段可以前后衔接起来的植物生长+衰败的过程，想要让模型学习到所有阶段，但是结果不尽如人意。
<img width="712" height="393" alt="image" src="https://github.com/user-attachments/assets/935e2f93-1a58-4926-8617-b31e54353e5e" />

 ConvLSTM模型参考：https://github.com/kylalangotsky/ndvi_timeseries_forcasting

