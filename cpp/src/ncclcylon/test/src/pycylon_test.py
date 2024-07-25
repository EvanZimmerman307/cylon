import pycylon

file = "/u/ewz9kg/cylon_nccl_dev/cpp/src/ncclcylon/basic/4stock_5day.csv"
tb = pycylon.read_csv(file, ",")
print(tb.is_cpu())

tb = tb.to_arrow()
npy = tb.to_pandas()
print(npy)


