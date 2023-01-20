import streamlit as st
import pandas as pd
from transformers import AutoModelWithLMHead, AutoTokenizer

# Load GPT-3 model and tokenizer
model = AutoModelWithLMHead.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Get user input data
st.title('GPT-3 App')
user_input = st.text_input('Enter your message:')
data = [[user_input]]
df = pd.DataFrame(data, columns = ['User Input'])

# Generate prediction using GPT-3 model
input_ids = tokenizer.encode(df.to_csv(index=False))
outputs = model.generate(input_ids)
prediction = tokenizer.decode(outputs[0])

# Display prediction to user
st.write("GPT-3's response:", prediction)
