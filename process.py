# Cài đặt thư viện cần thiết
!pip install ftfy chardet openpyxl

import pandas as pd
import json
import os
import glob
import numpy as np
import ftfy
import chardet

# --- HÀM SỬA LỖI FONT MẠNH MẼ ---
def fix_encoding_aggressive(text):
    if pd.isna(text) or text == '':
        return text
    text = str(text)
    text = ftfy.fix_text(text)
    try:
        if any(char in text for char in ['Ð', 'Ñ', 'Ð']):
            temp = text.encode('latin1', errors='ignore')
            try:
                text = temp.decode('utf-8')
            except:
                text = temp.decode('windows-1251', errors='ignore')
    except:
        pass
    text = ftfy.fix_text(text)
    return text.strip()

# --- CẤU HÌNH ---
INPUT_FOLDER = '/kaggle/input/datasets/js042710/3tand1k/data DM/data DM'  # ← sửa đúng path
OUTPUT_FOLDER = '/kaggle/working/cleaned_files/'

# =============================================
# ⚙️ CHỈ CẦN SỬA DÒNG NÀY TRÊN MỖI MÁY
# Máy 1: batch_files = all_files[0:10]
# Máy 2: batch_files = all_files[10:20]
# Máy 3: batch_files = all_files[20:32]
# =============================================

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

cols_to_keep = [
    'user_id', 'timestamp', 'recording_msid',
    'track_metadata.track_name', 
    'track_metadata.artist_name', 
    'track_metadata.release_name'
]

rename_map = {
    'track_metadata.track_name': 'track_name',
    'track_metadata.artist_name': 'artist_name',
    'track_metadata.release_name': 'release_name'
}

# --- HÀM XỬ LÝ 1 FILE ---
def process_single_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except: 
                    continue
    
    if not data: 
        return None
    
    df = pd.json_normalize(data)
    valid_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[valid_cols].rename(columns=rename_map)
    
    for col in ['track_name', 'artist_name', 'release_name']:
        if col in df.columns:
            df[col] = df[col].apply(fix_encoding_aggressive)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    
    if 'release_name' in df.columns:
        df['release_name'] = df['release_name'].replace(['', 'nan', 'None'], np.nan).fillna('Single')
    
    df = df.sort_values(by=['user_id', 'timestamp'])
    df['next_ts'] = df.groupby('user_id')['timestamp'].shift(-1)
    df['duration'] = (df['next_ts'] - df['timestamp']).dt.total_seconds()
    df_clean = df[(df['duration'] >= 30) | (df['duration'].isna())].copy()
    df_clean.drop(columns=['next_ts', 'duration'], inplace=True)
    
    return df_clean

# --- LẤY FILE THEO BATCH ---
all_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, '*.listens')))

batch_files = all_files[20:32]  # ← ĐỔI DÒNG NÀY TRÊN MỖI MÁY

print(f"🔍 Tổng file tìm thấy: {len(all_files)}")
print(f"📦 Batch này xử lý: {len(batch_files)} file")
print(f"   Từ: {os.path.basename(batch_files[0]) if batch_files else 'N/A'}")
print(f"   Đến: {os.path.basename(batch_files[-1]) if batch_files else 'N/A'}\n")

# --- VÒNG LẶP CHÍNH ---
for i, file_path in enumerate(batch_files):
    file_name = os.path.basename(file_path)
    print(f"[{i+1}/{len(batch_files)}] Đang xử lý: {file_name}...")
    
    try:
        df_result = process_single_file(file_path)
        
        if df_result is not None and not df_result.empty:
            num_rows = len(df_result)
            
            if num_rows > 1000000:
                out_name = f"clean_{file_name}.csv"
                out_path = os.path.join(OUTPUT_FOLDER, out_name)
                df_result.to_csv(out_path, index=False, encoding='utf-8-sig')
                print(f"   ⚠️ File lớn ({num_rows:,} dòng) → CSV: {out_name}")
            else:
                out_name = f"clean_{file_name}.xlsx"
                out_path = os.path.join(OUTPUT_FOLDER, out_name)
                df_result.to_excel(out_path, index=False, engine='openpyxl')
                print(f"   ✅ ({num_rows:,} dòng) → Excel: {out_name}")
        else:
            print(f"   ⚠️ File rỗng, bỏ qua.")
            
    except Exception as e:
        print(f"   ❌ LỖI: {e}")

print(f"\n🎉 HOÀN TẤT BATCH!")