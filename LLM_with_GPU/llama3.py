# Meta-Llama-3-8B-Instruct.Q4_1.gguf

import datetime as dt
from llama_cpp import Llama
import requests
import os


file_path = 'Meta-Llama-3-8B-Instruct.Q4_1.gguf'

# Функция os.path.isfile возвращает True, если файл существует, иначе False
if os.path.isfile(file_path):
    pass
else:
    print('Loading model ...')
    url = 'https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_1.gguf'
    response = requests.get(url)

    with open('Meta-Llama-3-8B-Instruct.Q4_1.gguf', 'wb') as file:
        file.write(response.content)


# Chat Completion API
llm = Llama(model_path="Meta-Llama-3-8B-Instruct.Q4_1.gguf",
            chat_format="llama-3",
            # n_gpu_layers=100,
            n_gpu_layers=-1,
            # n_gpu_layers=1,  # change for metal
            n_ctx = 2048,
            n_batch = 1024,
            verbose = True
            )


system_message = """
"Вы - лучший GPT бизнес-ассистент!
Ваша задача помогать.
Ответьте на вопрос пользователя НА РУССКОМ ЯЗЫКЕ, используя информацию из информационных источников.
"""

data = """
«Возвращение блудного попугая» — советский и российский мультипликационный сериал, созданный режиссёром Валентином Караваевым и сценаристом Александром Курляндским в 1984 году.

Сюжет рассказывает о приключениях попугая Кеши, «героя нашего времени». Действие сосредоточено в городе и его окрестностях. Кеша живёт в квартире школьника Вовки, однако из-за своего вспыльчивого, заносчивого характера периодически сбегает и попадает в неприятности, в конце концов возвращаясь к Вовке с повинной.
"""

# user_question = """
# Кто такой Кеша?
# """


def answer(user_question):
    timer = dt.datetime.now()
    ans = llm.create_chat_completion(messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "Выдержки текста:\n"+data},
        {"role": "user", "content": "Вопрос пользователя: "+user_question},
    ]
    )
    answer_time = (dt.datetime.now() - timer).seconds
    answer_text = ans['choices'][0]['message']['content']
    return f'{answer_text} \nВремя ответа: {answer_time:.2f} сек'


q = 'привет!'
print(answer(q))