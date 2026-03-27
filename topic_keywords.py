import csv
import json


def load_keywords_from_csv(csv_file='ms-main-key.csv'):
    """
    Đọc file CSV và convert sang dictionary với format:
    {
        "topic_id": ["keyword1", "keyword2", ...]
    }
    Merge tất cả keywords của cùng một topic_id
    """
    keywords_dict = {}
    
    with open(csv_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            topic_id = row.get('ID Topic MS', '').strip()
            main_keywords = row.get('main_keywords', '').strip()
            
            # Bỏ qua dòng trống hoặc không có topic_id
            if not topic_id or not main_keywords:
                continue
            
            # Tách keywords bằng dấu phẩy và chuyển thành lowercase
            keywords_list = [
                keyword.strip().lower() 
                for keyword in main_keywords.split(',')
                if keyword.strip()
            ]
            
            # Merge keywords nếu topic_id đã tồn tại
            if topic_id in keywords_dict:
                # Thêm keywords mới và loại bỏ trùng lặp
                existing_keywords = set(keywords_dict[topic_id])
                new_keywords = set(keywords_list)
                keywords_dict[topic_id] = list(existing_keywords | new_keywords)
            else:
                keywords_dict[topic_id] = keywords_list
    
    return keywords_dict


def get_main_keywords(topic_id, keywords_dict=None):
    """
    Lấy danh sách main_keywords theo topic_id
    
    Args:
        topic_id (str): ID của topic cần lấy keywords
        keywords_dict (dict, optional): Dictionary chứa keywords. 
                                       Nếu None, sẽ load từ CSV
    
    Returns:
        list: Danh sách keywords dạng lowercase, hoặc [] nếu không tìm thấy
    """
    if keywords_dict is None:
        keywords_dict = load_keywords_from_csv()
    
    return keywords_dict.get(topic_id, [])


def save_to_json(output_file='keywords.json'):
    """
    Lưu dữ liệu keywords vào file JSON
    """
    keywords_dict = load_keywords_from_csv()
    
    # Đếm tổng số keywords
    total_keywords = sum(len(keywords) for keywords in keywords_dict.values())
    
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(keywords_dict, file, ensure_ascii=False, indent=2)
    
    print(f"Đã lưu {len(keywords_dict)} topics (tổng {total_keywords} keywords) vào {output_file}")
    return keywords_dict


# Ví dụ sử dụng
if __name__ == "__main__":
    # Load và lưu vào JSON
    keywords_data = save_to_json()
    
    # Test lấy keywords theo topic_id
    test_topic_id = "5ff52f9c3cb8f0afa747233b"
    keywords = get_main_keywords(test_topic_id, keywords_data)
    
    print(f"\nKeywords cho topic_id '{test_topic_id}':")
    print(json.dumps(keywords, ensure_ascii=False, indent=2))
