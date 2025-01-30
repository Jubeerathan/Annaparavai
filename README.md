# Annaparavai

An AI-generated vs Human-written text detector for Tamil and Malayalam languages

## Installation
```pip install Annaparavai```

## Usage
```python
from Annaparavai import TextDetector

# Initialize detector (choose language: 'Tamil' or 'Malayalam')
detector = TextDetector(language='Tamil')

# Example texts to classify
text = "இது பழுதாக போயிட்டுது."
# text = "ഈ പാൽതന്തു മാവ് ഉപയോഗിച്ചപ്പോൾ എനിക്ക് സംതൃപ്തിയില്ല."
prediction = detector.predict(text)
print(prediction[0])
```