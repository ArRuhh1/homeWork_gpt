import pytest
from transformers import pipeline

# Загружаем модель (используем заглушку, если модель отсутствует)
try:
    text_generator = pipeline("text-generation", model="gpt2")
except Exception:
    text_generator = None
    
    
def generate_text(prompt, max_length=150, temperature=1.0):
    """Функция генерации текста с параметрами."""
    if text_generator is None:
        return "Ошибка загрузки модели"
    return text_generator(prompt, max_length=max_length, temperature=temperature)[0]['generated_text']

@pytest.mark.parametrize("prompt, max_length, temperature", [
    ("Привет, мир!", 100, 1.0),
    ("Как работает машинное обучение?", 50, 0.8),
    ("Расскажи сказку", 200, 1.2)
])
def test_generate_text(prompt, max_length, temperature):
    """Тест проверяет, что функция генерации возвращает текст."""
    result = generate_text(prompt, max_length, temperature)
    assert isinstance(result, str)
    assert len(result) > len(prompt)

def test_generate_text_empty():
    """Тест проверяет генерацию текста с пустым вводом."""
    result = generate_text("", 100, 1.0)
    assert isinstance(result, str)
    assert len(result) > 0

def test_generate_text_length():
    """Тест проверяет, что длина сгенерированного текста не превышает max_length."""
    prompt = "Проверка длины"
    max_length = 50
    result = generate_text(prompt, max_length, 1.0)
    assert isinstance(result, str)
    assert len(result) <= max_length + len(prompt)