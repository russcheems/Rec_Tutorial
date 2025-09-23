import pandas as pd
import os

def convert_movielens_to_csv(input_path, output_path):
    df = pd.read_csv(input_path, sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')
    df.to_csv(output_path, index=False)
    print(f"转换完成，已保存为: {output_path}")

if __name__ == "__main__":
    input_file = "ml-100k/u.data" 
    output_file = "ml-100k/ratings.csv"
    if not os.path.exists(input_file):
        print(f"找不到文件: {input_file}")
    else:
        convert_movielens_to_csv(input_file, output_file)
