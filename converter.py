import numpy as np
import os
import sys

def convert_to_dict_format(file_path, d_2):
    """
    将旧格式的 .npy 文件转换为字典格式并重新保存
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在。")
        return
    
    # 加载旧格式的 .npy 文件
    loaded_data = np.load(file_path, allow_pickle=True)
    
    # 检查文件内容长度，判断是否是旧的格式
    if len(loaded_data) == 9:  # 旧格式（没有 d_2）
        time_str, f_A, f_B, J_back_r, mode_number, zeta_ini, zeta_end, iter_number, power_interval = loaded_data
        data_dict = {
            "time_str": time_str,
            "f_A": f_A,
            "f_B": f_B,
            "J_back_r": J_back_r,
            "mode_number": mode_number,
            "zeta_ini": zeta_ini,
            "zeta_end": zeta_end,
            "iter_number": iter_number,
            "power_interval": power_interval,
            "d_2": d_2  # 从命令行获取的新变量 d_2
        }
        print("旧格式文件已加载并转换。")
    
    elif len(loaded_data) == 10:  # 新格式，包含 d_2
        time_str, f_A, f_B, J_back_r, mode_number, zeta_ini, zeta_end, iter_number, power_interval, _ = loaded_data
        data_dict = {
            "time_str": time_str,
            "f_A": f_A,
            "f_B": f_B,
            "J_back_r": J_back_r,
            "mode_number": mode_number,
            "zeta_ini": zeta_ini,
            "zeta_end": zeta_end,
            "iter_number": iter_number,
            "power_interval": power_interval,
            "d_2": d_2  # 从命令行获取的新变量 d_2
        }
        print("带 d_2 的数据已加载并转换。")
    
    else:
        print(f"文件格式不符合预期，包含 {len(loaded_data)} 个元素。")
        return
    
    # 将字典保存为新的 .npy 文件
    new_file_path = file_path.replace('.npy', '_dict.npy')
    np.save(new_file_path, data_dict)
    print(f"数据已保存为新的字典格式文件：{new_file_path}")

# 从命令行读取文件路径和 d_2 的值
if len(sys.argv) != 3:
    print("用法: python script_name.py <file_path> <d_2_value>")
else:
    file_path = sys.argv[1]
    d_2_value = float(sys.argv[2])  # 假设 d_2 是一个浮点数
    convert_to_dict_format(file_path, d_2_value)
