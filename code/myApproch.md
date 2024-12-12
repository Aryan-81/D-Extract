This in my another approach(not implemented here) to using two separate pipelines sounds effective, especially for handling a variety of entity types with distinct requirements. Here’s how each pipeline could be implemented and optimized for detecting `entity_value`:

### 1. **Pipeline 1: Physical Dimensions Detection**
   - **Goal**: Use CV-based algorithms (e.g., Hough Transform) to detect physical dimensions like height, width, and depth, which usually correspond to line markers in images.
   - **Steps**:
     1. **Edge Detection**: Use methods like Canny edge detection to identify sharp boundaries where dimension markers might exist.
     2. **Line Detection**: Apply Hough Line Transform to detect straight lines, filtering for horizontal or vertical orientations.
     3. **Dimension Extraction**: After identifying line pairs representing physical dimensions, use OCR to read the measurement values near these lines.
   - **Output**: A list of dimension measurements (e.g., height, width) extracted directly from the image.

### 2. **Pipeline 2: OCR for Entity-Specific Units**
   - **Goal**: For other entities, use OCR to detect numbers alongside relevant units (like grams for weight, liters for volume, etc.).
   - **Steps**:
     1. **OCR Text Detection**: Use an OCR tool (like Tesseract) to scan the image and extract all textual elements, capturing any detected numbers and units.
     2. **Entity-Specific Filtering**:
         - Create a mapping of each `entity_name` to expected units (e.g., "weight" -> "g", "kg"; "volume" -> "ml", "L").
         - For each OCR result, filter based on the `entity_name` to locate the number closest to the relevant unit.
     3. **Error Correction**: Use regex or NLP techniques to handle OCR misreads (e.g., "g" misread as "9").
     4. **Verification**: Cross-check with prior knowledge or constraints (e.g., valid weight ranges for certain products) to improve accuracy.
   - **Output**: Entity values for items like weight, volume, etc., detected through OCR.

### Combining the Pipelines
For each image, you could first classify whether it contains physical dimensions or other types of measurements:
- **Classifier Step**: A lightweight classifier could help route images to the correct pipeline based on visual characteristics or keywords associated with each entity type.

This two-pipeline approach balances the specialized needs of physical dimensions and other entities, making the system more flexible and accurate across different product types.

