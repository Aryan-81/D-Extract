import streamlit as st
from PIL import Image
import random
import pytesseract
import easyocr
import re
from src.varibles import ImgDataMngr
from src.single_unit import units_dict

# OCR Configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
reader = easyocr.Reader(['en'])

# Helper Functions
def preprocess_image(image_path, entity_name):
    uid = random.randint(100, 999)
    group_id = random.randint(100, 999)

    # Initialize Image Data Manager
    img_data_mngr = ImgDataMngr(
        uid=uid,
        image_paths=image_path,
        group_ids=group_id,
        entity_names=entity_name
    )

    return img_data_mngr.preProcessedImages[0]

def ocr(preprocessed_img, ocr_method='e'):
    try:
        if ocr_method == 't':
            text = pytesseract.image_to_string(preprocessed_img, lang='eng')
        else:
            text = ''.join(reader.readtext(preprocessed_img, detail=0))
        return text
    except Exception as e:
        return f"OCR error: {e}"

def postprocessing(text):
    lower_case = text.lower().replace('\n', '')
    symbols = r'[ !@#%^&*()$_\-\+\{\}\[\]\'\|:;"<>,/~?`=\"©™°®¢»«¥“”§—‘’é€]'
    pattern = r'(\d+(\.\d+)?\w{2})'
    cleaned_symbol = re.sub(symbols, ' ', lower_case)
    cleaned_text = re.findall(pattern, cleaned_symbol)
    return [item[0] for item in cleaned_text]  # Extract main match

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

# Streamlit App
st.title("Entity Extraction Tool")
st.sidebar.info("Built with Streamlit")

# Image Upload
uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])
entity_name = st.selectbox("Select an entity type:", ["item_weight", "width",'depth',"height","voltage","wattage","item_volume"])

if uploaded_file:
    # Save uploaded file to a temporary location
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load image for display
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze"):
        with st.spinner("Processing..."):
            # Preprocess and analyze
            preprocessed_image = preprocess_image(temp_file_path, entity_name)
            extracted_text = ocr(preprocessed_image)
            cleaned_text = postprocessing(extracted_text)
            matched_units = match_units(cleaned_text, units_dict, entity_name)

            # Display results
            if matched_units:
                st.success(f"Predicted entity values: {matched_units}")
            else:
                st.error("No matching entities found. Try a different image or entity type.")
