import json
import csv
import os

def extract_first(value):
    """If the value is a list, pick the first element. Otherwise return the value."""
    if isinstance(value, list) and len(value) > 0:
        return value[0]
    return value

def create_global_description(item):
    """
    Creates a natural language global description.
    """
    type_val = str(extract_first(item.get('type', ''))).replace('_', ' ') if item.get('type') else 'item'
    style_val = str(extract_first(item.get('style', ''))).replace('_', ' ') if item.get('style') else ''
    color_val = str(extract_first(item.get('color', ''))).replace('_', ' ') if item.get('color') else ''
    pattern_val = str(extract_first(item.get('pattern', ''))).replace('_', ' ') if item.get('pattern') else ''
    material_val = str(extract_first(item.get('material', ''))).replace('_', ' ') if item.get('material') else ''
    fit_val = str(extract_first(item.get('fit', ''))).replace('_', ' ') if item.get('fit') else ''
    gender_val = str(extract_first(item.get('gender', ''))).replace('_', ' ') if item.get('gender') else ''
    age_group_val = str(extract_first(item.get('age_group', ''))).replace('_', ' ') if item.get('age_group') else ''
    season_val = str(extract_first(item.get('season', ''))).replace('_', ' ') if item.get('season') else ''
    
    # Adjectives before the type
    adjectives = [a for a in [style_val, color_val, pattern_val] if a]
    
    if adjectives:
        adj_str = " ".join(adjectives)
        first_char = adj_str[0].lower() if adj_str else 'i'
        article = "An" if first_char in 'aeiou' else "A"
        desc = f"{article} {adj_str} {type_val}"
    else:
        first_char = type_val[0].lower() if type_val else 'i'
        article = "An" if first_char in 'aeiou' else "A"
        desc = f"{article} {type_val}"
        
    if material_val:
        desc += f" made of {material_val}"
        
    if fit_val:
        desc += f" that is {fit_val}"
        
    if gender_val or age_group_val:
        target = " ".join([t for t in [gender_val, age_group_val] if t])
        desc += f", made for a {target}"
        
    if season_val:
        desc += f" for the {season_val} season"
        
    return desc + "."

def main():
    # Base path assuming the script is in temp folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input JSON file path (adjust the filename if needed)
    json_path = os.path.join(base_dir, '..', '20260315_184628.json')
    
    # Output CSV file path
    csv_path = os.path.join(base_dir, 'output_items.csv')
    
    if not os.path.exists(json_path):
        # Fallback to data.json if the specific timestamped one isn't found
        json_path = os.path.join(base_dir, '..', 'data.json')
        
    print(f"Reading from: {json_path}")
    
    # Read JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    headers = [
        'id', 'global_description', 'type', 'color', 'style', 
        'pattern', 'material', 'fit', 'gender', 'age_group', 'season'
    ]
    
    # Write CSV
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for item in data:
            row = {}
            row['id'] = item.get('id', '')
            row['global_description'] = create_global_description(item)
            row['type'] = extract_first(item.get('type', ''))
            row['color'] = extract_first(item.get('color', ''))
            row['style'] = extract_first(item.get('style', ''))
            row['pattern'] = extract_first(item.get('pattern', ''))
            row['material'] = extract_first(item.get('material', ''))
            row['fit'] = extract_first(item.get('fit', ''))
            row['gender'] = extract_first(item.get('gender', ''))
            row['age_group'] = extract_first(item.get('age_group', ''))
            row['season'] = extract_first(item.get('season', ''))
            
            writer.writerow(row)
            
    print(f"Successfully converted {len(data)} items to {csv_path}")

if __name__ == "__main__":
    main()
