import os
import urllib.request
import zipfile
import pandas as pd

def download_and_prepare_data():
    data_dir = "NLP/data/cls_raw"
    os.makedirs(data_dir, exist_ok=True)
    
    zip_path = os.path.join(data_dir, "sms_spam_collection.zip")
    extracted_path = os.path.join(data_dir, "sms_spam_collection")
    data_file_path = os.path.join(data_dir, "SMSSpamCollection.tsv")

    # 1. 下载数据集
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    print(f"正在下载数据集从 {url} ...")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"下载失败，尝试备用链接... ({e})")
        url_backup = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
        urllib.request.urlretrieve(url_backup, zip_path)

    # 2. 解压数据集
    print("正在解压...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    
    # 提取出的文件默认没有后缀名，重命名为 .tsv
    original_file = os.path.join(extracted_path, "SMSSpamCollection")
    os.rename(original_file, data_file_path)

    # 3. 数据预处理与类别平衡
    print("正在处理数据与划分数据集...")
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    
    # 欠采样使得正负样本平衡 
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    
    # 标签转为数字：ham->0, spam->1
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    
    # 打乱顺序
    balanced_df = balanced_df.sample(frac=1, random_state=123).reset_index(drop=True)
    
    # 4. 划分训练集(70%)、验证集(10%)、测试集(20%)
    train_end = int(len(balanced_df) * 0.7)
    val_end = train_end + int(len(balanced_df) * 0.1)
    
    train_df = balanced_df[:train_end]
    val_df = balanced_df[train_end:val_end]
    
    # 5. 保存到目标位置
    train_csv_path = os.path.join(data_dir, "train.csv")
    val_csv_path = os.path.join(data_dir, "validation.csv")
    
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    print(f"数据准备完毕！")
    print(f"训练集已保存至: {train_csv_path} (共 {len(train_df)} 条)")
    print(f"验证集已保存至: {val_csv_path} (共 {len(val_df)} 条)")

if __name__ == "__main__":
    download_and_prepare_data()