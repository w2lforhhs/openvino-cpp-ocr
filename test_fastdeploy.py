import fastdeploy as fd

# 检查是否支持 Paddle 和 OpenVINO
print("Paddle", fd.RuntimeOption().use_paddle_backend())
print("OpenVINO", fd.RuntimeOption().use_openvino_backend())

# 查看版本
print("FastDeploy", fd.__version__)