import streamlit as st
from transformers import pipeline

# Загрузка модели генерации текста
text_generator = pipeline("text-generation", model="gpt2")

# Интерфейс Streamlit
st.title("Генерация текста с GPT-2")
st.write("Введите начальный текст, и модель сгенерирует продолжение.")

# Поле для ввода текста
user_input = st.text_area("Начальный текст:", "Пример: Искусственный интеллект – это...")

# Параметры генерации
max_length = st.slider("Максимальная длина текста", 50, 500, 150)
temperature = st.slider("Температура (креативность)", 0.5, 1.5, 1.0)

if st.button("Сгенерировать текст"):
    with st.spinner("Генерация..."):
        output = text_generator(user_input, max_length=max_length, temperature=temperature)
        st.subheader("Сгенерированный текст:")
        st.write(output[0]['generated_text'])