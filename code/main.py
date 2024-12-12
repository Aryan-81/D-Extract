import random
import pytesseract
import easyocr
import re
from src.varibles import ImgDataMngr
from src.single_unit import units_dict

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
reader = easyocr.Reader(['en'])

# Generate random UID and Group ID
uid = random.randint(100, 999)
group_id = random.randint(100, 999)
image_path = 'DownloadedImages\\61I9XdN6OFL-example_uid.jpg'
entity_name = 'item_weight'

# Initialize Image Data Manager
img_data_mngr = ImgDataMngr(
    uid=uid,
    image_paths=image_path,
    group_ids=group_id,
    entity_names=entity_name
)

# Preprocess Image
preProcessed_image = img_data_mngr.preProcessedImages[0]

# OCR Function
def ocr(preprocessed_img, ocr_method='e') -> str:
    try:
        if ocr_method == 't':
            text = pytesseract.image_to_string(preprocessed_img, lang='eng')
        else:
            text = ''.join(reader.readtext(preprocessed_img, detail=0))
        return text
    except Exception as e:
        print(f"OCR error: {e}")
        return ""

texts = ocr(preProcessed_image)
print(texts)
# Postprocessing
def postprocessing(text: str) -> list:
    lower_case = text.lower().replace('\n', '')
    symbols = r'[ !@#%^&*()$_\-\+\{\}\[\]\'\|:;"<>,/~?`=\"©™°®¢»«¥“”§—‘’é€]'
    pattern = r'(\d+(\.\d+)?\w{2})'
    cleaned_symbol = re.sub(symbols, ' ', lower_case)
    cleaned_text = re.findall(pattern, cleaned_symbol)
    cleaned_text = [item[0] for item in cleaned_text]  # Extract main match
    return cleaned_text

texts = postprocessing(texts)
print(texts)
# Match Units
def match_units(input_list, units_dict, entity_name):
    abb_dict = {key: value 
                for category, sub_dict in units_dict.items()
                if category == entity_name
                for key, value in sub_dict.items()}
    results = []
    for item in input_list:
        if len(item) >= 2:
            last_two_chars = item[-2:]
            result = abb_dict.get(last_two_chars)
            if result:
                results.append(f"{item[:-2]} {result}")
    return results

matched_units = match_units(texts, units_dict, entity_name)

print('Predicted entity values:', matched_units)
