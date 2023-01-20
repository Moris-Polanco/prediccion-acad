import streamlit as st
import pandas as pd
from transformers import AutoModelWithLMHead, AutoTokenizer

# Load GPT-3 model and tokenizer
model = AutoModelWithLMHead.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Get user input data
st.title('Academic Success Predictor')
language_marks = st.slider('Language Marks', 0, 100, 50)
maths_marks = st.slider('Maths Marks', 0, 100, 50)
data = [[language_marks, maths_marks]]
df = pd.DataFrame(data, columns = ['Language Marks', 'Maths Marks'])

# Generate prediction using GPT-3 model
input_ids = tokenizer.encode(df.to_csv(index=False))
outputs = model.generate(input_ids)
prediction = tokenizer.decode(outputs[0])

# Display prediction to user
st.write("The predicted academic success is:", prediction)
