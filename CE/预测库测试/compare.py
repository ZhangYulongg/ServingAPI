import numpy as np


def parse_data(filename="profile"):
    output_gpu = []
    with open(filename, "r") as f:
        lines = f.readlines()
    for line in lines:
        data = float(line.strip())
        output_gpu.append(data)
    return np.array(output_gpu).astype("float32")


def compare(a, b):
    diff_array = a - b
    total = len(diff_array)
    count_e2 = 0
    count_e3 = 0
    count_e4 = 0
    for number in diff_array:
        diff = abs(number)
        if diff > 1e-2:
            count_e2 += 1
            count_e3 += 1
            count_e4 += 1
        elif diff > 1e-3:
            count_e3 += 1
            count_e4 += 1
        elif diff > 1e-4:
            count_e4 += 1
    print("total:", total)
    print("diff > 1e-2:", count_e2, f"{count_e2 / total * 100:.3f}%")
    print("diff > 1e-3:", count_e3, f"{count_e3 / total * 100:.3f}%")
    print("diff > 1e-4:", count_e4, f"{count_e4 / total * 100:.3f}%")


if __name__ == '__main__':
    gpu_data_fix = parse_data("gpu_profile_fixed_pure")
    fp32_data_fix = parse_data("fp32_profile_fixed_pure")
    fp16_data_fix = parse_data("fp16_profile_fixed_pure")

    gpu_data_212 = parse_data("gpu_profile_212_pure")
    fp32_data_212 = parse_data("fp32_profile_212_pure")
    fp16_data_212 = parse_data("fp16_profile_212_pure")

    gpu_data_dev = parse_data("gpu_profile_dev_pure")
    fp32_data_dev = parse_data("fp32_profile_dev_pure")
    fp16_data_dev = parse_data("fp16_profile_dev_pure")

    fp32_212 = parse_data("fp32_212_pure")
    fp16_212 = parse_data("fp16_212_pure")

    print("fixed：ed6624ab78d70aa51ca25d6759e6d2ca4e9da9cb")
    print("gpu - fp32:")
    compare(gpu_data_fix, fp32_data_fix)
    print("gpu - fp16:")
    compare(gpu_data_fix, fp16_data_fix)
    print("fp32 - fp16:")
    compare(fp32_data_fix, fp16_data_fix)
    print()

    print("2.1.2：e04b66f2d272d68f77dcd94cb2956938475411d8")
    print("gpu - fp32:")
    compare(gpu_data_212, fp32_data_212)
    print("gpu - fp16:")
    compare(gpu_data_212, fp16_data_212)
    print("fp32 - fp16:")
    compare(fp32_data_212, fp16_data_212)
    print()

    print("dev：572bad8a66ea9650d753911af8383785ed671af9")
    print("gpu - fp32:")
    compare(gpu_data_dev, fp32_data_dev)
    print("gpu - fp16:")
    compare(gpu_data_dev, fp16_data_dev)
    print("fp32 - fp16:")
    compare(fp32_data_dev, fp16_data_dev)
    print()

    print("相同数据：")
    print("fp16 - fp16:")
    compare(fp16_212, fp16_212)

