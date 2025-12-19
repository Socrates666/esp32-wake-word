import esp_ppq
import numpy as np
from esp_ppq import ESPPQUtils

# 加载 ESPDL 模型
model = ESPPQUtils.load_espdl_model('xiaoa08.espdl')

# 准备输入数据
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 执行推理
output = model.inference()

print("推理结果:", output)