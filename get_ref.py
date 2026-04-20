import os
import numpy as np
from PIL import Image
import torch
import clip  
from sklearn.preprocessing import MinMaxScaler

# 配置路径
cold_items_dir = "t_yuan_10"
hot_items_dir = "tradesy_hot_images"
popularity_file = "tradesy_popular_items_info.txt"
output_file = "reference_selection_results.txt"

# 初始化CLIP
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
except Exception as e:
    raise ImportError("Failed to load CLIP. Please install with: pip install git+https://github.com/openai/CLIP.git") from e

# 读取热门商品流行度数据
def load_popularity_data(file_path):
    popularity = {}
    with open(file_path, 'r') as f:
        next(f)  # 跳过标题行
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    pid, count = parts[0], parts[1]
                    popularity[pid] = int(count)
    return popularity

# 计算CLIP视觉特征
def get_clip_features(image_path):
    try:
        image = Image.open(image_path)
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            return model.encode_image(image_input).cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def main():
    # 加载数据
    popularity_data = load_popularity_data(popularity_file)
    
    # 处理热门商品
    hot_items = []
    for filename in os.listdir(hot_items_dir):
        if filename.endswith('.jpg'):
            pid = os.path.splitext(filename)[0]
            img_path = os.path.join(hot_items_dir, filename)
            features = get_clip_features(img_path)
            if features is not None:
                hot_items.append({
                    'pid': pid,
                    'image_path': img_path,
                    'popularity': popularity_data.get(pid, 0),
                    'features': features
                })

    # 处理冷门商品
    cold_items = []
    for filename in os.listdir(cold_items_dir):
        if filename.endswith('.jpg'):
            pid = os.path.splitext(filename)[0]
            img_path = os.path.join(cold_items_dir, filename)
            features = get_clip_features(img_path)
            if features is not None:
                cold_items.append({
                    'pid': pid,
                    'image_path': img_path,
                    'features': features
                })

    # 归一化流行度
    pop_values = np.array([x['popularity'] for x in hot_items])
    pop_scaler = MinMaxScaler()
    norm_pop = pop_scaler.fit_transform(pop_values.reshape(-1, 1)).flatten()
    for i, item in enumerate(hot_items):
        item['norm_pop'] = norm_pop[i]

    # 选择参考商品 (α=0.5)
    alpha = 0.05
    results = []
    
    for cold in cold_items:
        best_match = None
        best_score = -1
        
        for hot in hot_items:
            # 计算余弦相似度
            cos_sim = np.dot(cold['features'], hot['features']) / (
                np.linalg.norm(cold['features']) * np.linalg.norm(hot['features'])
            )
            # 综合得分
            score = alpha * hot['norm_pop'] + (1 - alpha) * (cos_sim + 1)/2  # 归一化到[0,1]
            
            if score > best_score:
                best_score = score
                best_match = hot
        
        if best_match:
            results.append({
                'cold_pid': cold['pid'],
                'ref_pid': best_match['pid'],
                'ref_pop': best_match['popularity'],
                'similarity': (best_score - alpha*best_match['norm_pop'])/(1-alpha),
                'combined_score': best_score
            })

    # 保存结果
    with open(output_file, 'w') as f:
        f.write("cold_id,ref_id,popularity,similarity,combined_score\n")
        for res in results:
            f.write(f"{res['cold_pid']},{res['ref_pid']},{res['ref_pop']:.0f},{res['similarity']:.4f},{res['combined_score']:.4f}\n")

    print(f"\nGenerated reference selections for {len(results)} cold items")
    print(f"Results saved to {output_file}")
    print("\nSample results:")
    for res in results[:3]:
        print(f"Cold: {res['cold_pid']} -> Hot: {res['ref_pid']} (Pop: {res['ref_pop']}, Sim: {res['similarity']:.2f})")

if __name__ == "__main__":
    main()
