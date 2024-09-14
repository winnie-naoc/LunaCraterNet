import pandas as pd  
  
# 读取CSV文件  
df = pd.read_csv('/DATA01/yolov5-7.0-train/runs/train/exp20/results.csv')  
  
# 将所有浮点数列保留小数点后3位  
float_cols = df.select_dtypes(include=['float64']).columns  
for col in float_cols:  
    df[col] = df[col].round(3)  
  
# 将处理后的数据保存回CSV文件  
df.to_csv('/DATA01/yolov5-7.0-train/runs/train/exp20/results_dd.csv', index=False)