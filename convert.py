from numbers_parser import Document
import csv

def convert_numbers_to_csv(numbers_path, csv_path):
    # 讀取 .numbers 檔案
    doc = Document(numbers_path)
    
    # 取得第一個工作表 (Sheet) 中的第一個表格 (Table)
    # 註：一個 .numbers 檔案可以有多個 Sheet，每個 Sheet 可以有多個 Table
    sheet = doc.sheets[0]
    table = sheet.tables[0]
    
    # 開啟並寫入 CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in table.rows():
            # 取得每一列中單元格的值 (value)
            writer.writerow([cell.value for cell in row])

    print(f"轉換完成！檔案已儲存至: {csv_path}")

# 使用範例
convert_numbers_to_csv('mirage_architecture_checklist.numbers', 'output.csv')