import sqlite3
import os

# 检查数据库文件是否存在
if os.path.exists('data/market_data.db'):
    print("Database file exists")
    
    # 使用Python内置sqlite3
    conn = sqlite3.connect('data/market_data.db')
    cursor = conn.cursor()
    
    # 获取所有表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables: {tables}")
    
    # 检查market_data表结构
    if ('market_data',) in tables:
        cursor.execute("PRAGMA table_info(market_data);")
        columns = cursor.fetchall()
        print("Market_data table structure:")
        for col in columns:
            print(f"  {col}")
    else:
        print("market_data table not found")
        
    conn.close()
else:
    print("Database file does not exist")