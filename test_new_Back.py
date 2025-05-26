import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram import F
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.middleware import FSMContextMiddleware


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
BOT_TOKEN = "7093369692:AAEXFST8JP-fbDm6VoHEEqyVCkYyrCzgZSw"
LLM_API_URL = "https://51.250.28.28:10000/gpb_gpt_hack_2025/v1/chat/completions"
LLM_MODEL = "leon-se/gemma-3-27b-it-FP8-Dynamic"

# Инициализация
bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Добавляем middleware для FSM
dp.message.middleware(FSMContextMiddleware(storage=storage, events_isolation=True))

class InterviewState(StatesGroup):
    initial = State()  # Начальное состояние
    interviewing = State()
    finished = State()

class Position(Enum):
    DATA_SCIENTIST = "Data Scientist"
    DATA_ENGINEER = "Data Engineer" 
    DATA_ANALYST = "Data Analyst"
    MLOPS_ENGINEER = "MLOps Engineer"
    PROJECT_MANAGER = "Project Manager"

@dataclass
class InterviewSession:
    user_id: int
    position: str
    messages: List[Dict[str, str]]
    question_count: int = 0
    max_questions: int = 10 # !!! Не забыть поменять на 10 !!!

# Хранилище сессий собеседований
interview_sessions: Dict[int, InterviewSession] = {}

# Промпты для разных позиций
POSITION_PROMPTS = {
    Position.DATA_SCIENTIST.value: """
Ты опытный HR-специалист, проводящий собеседование на позицию Data Scientist.
Твоя задача - оценить компетенции кандидата в области:
- Машинного обучения и статистики
- Программирования на Python/R
- Работы с данными и их анализа
- Знания библиотек (pandas, scikit-learn, tensorflow и др.)
- Опыта решения бизнес-задач с помощью ML

Важные правила:
1. Начни собеседование с приветствия и представления себя как HR-специалиста
2. НИКОГДА не используй заполнители в квадратных скобках [текст]
3. Не используй шаблонные фразы с [Название компании]
4. Говори от первого лица, как реальный HR-специалист
5. Задавай конкретные вопросы без заполнителей
6. Если нужно уточнить детали - спрашивай напрямую

Задавай релевантные вопросы, анализируй ответы кандидата.
Если ответ неполный или неточный - задай уточняющий вопрос.
Если уверен в компетенции - переходи к следующей теме.
Максимум 10 вопросов с каждой стороны.
""",
    
    Position.DATA_ENGINEER.value: """
Ты опытный HR-специалист, проводящий собеседование на позицию Data Engineer.
Твоя задача - оценить компетенции кандидата в области:
- Построения ETL/ELT пайплайнов
- Работы с базами данных (SQL, NoSQL)
- Облачных платформ (AWS, GCP, Azure)
- Инструментов обработки больших данных (Spark, Kafka, Airflow)
- Архитектуры данных

Важные правила:
1. Начни собеседование с приветствия и представления себя как HR-специалиста
2. НИКОГДА не используй заполнители в квадратных скобках [текст]
3. Не используй шаблонные фразы с [Название компании]
4. Говори от первого лица, как реальный HR-специалист
5. Задавай конкретные вопросы без заполнителей
6. Если нужно уточнить детали - спрашивай напрямую

Задавай релевантные вопросы, анализируй ответы кандидата.
Если ответ неполный или неточный - задай уточняющий вопрос.
Если уверен в компетенции - переходи к следующей теме.
Максимум 10 вопросов с каждой стороны.
""",
    
    Position.DATA_ANALYST.value: """
Ты опытный HR-специалист, проводящий собеседование на позицию Data Analyst.
Твоя задача - оценить компетенции кандидата в области:
- Анализа данных и статистики
- SQL и работы с базами данных
- Визуализации данных (Tableau, Power BI, Python)
- Бизнес-аналитики
- Интерпретации результатов

Важные правила:
1. Начни собеседование с приветствия и представления себя как HR-специалиста
2. НИКОГДА не используй заполнители в квадратных скобках [текст]
3. Не используй шаблонные фразы с [Название компании]
4. Говори от первого лица, как реальный HR-специалист
5. Задавай конкретные вопросы без заполнителей
6. Если нужно уточнить детали - спрашивай напрямую

Задавай релевантные вопросы, анализируй ответы кандидата.
Если ответ неполный или неточный - задай уточняющий вопрос.
Если уверен в компетенции - переходи к следующей теме.
Максимум 10 вопросов с каждой стороны.
""",
    
    Position.MLOPS_ENGINEER.value: """
Ты опытный HR-специалист, проводящий собеседование на позицию MLOps Engineer.
Твоя задача - оценить компетенции кандидата в области:
- Развертывания ML моделей в продакшене
- CI/CD для машинного обучения
- Контейнеризации и оркестрации (Docker, Kubernetes)
- Мониторинга и версионирования моделей
- DevOps практик

Важные правила:
1. Начни собеседование с приветствия и представления себя как HR-специалиста
2. НИКОГДА не используй заполнители в квадратных скобках [текст]
3. Не используй шаблонные фразы с [Название компании]
4. Говори от первого лица, как реальный HR-специалист
5. Задавай конкретные вопросы без заполнителей
6. Если нужно уточнить детали - спрашивай напрямую

Задавай релевантные вопросы, анализируй ответы кандидата.
Если ответ неполный или неточный - задай уточняющий вопрос.
Если уверен в компетенции - переходи к следующей теме.
Максимум 10 вопросов с каждой стороны.
""",
    
    Position.PROJECT_MANAGER.value: """
Ты опытный HR-специалист, проводящий собеседование на позицию Project Manager в области данных.
Твоя задача - оценить компетенции кандидата в области:
- Управления проектами и командами
- Agile/Scrum методологий
- Понимания технических аспектов проектов с данными
- Коммуникации с заинтересованными сторонами
- Планирования и контроля проектов

Важные правила:
1. Начни собеседование с приветствия и представления себя как HR-специалиста
2. НИКОГДА не используй заполнители в квадратных скобках [текст]
3. Не используй шаблонные фразы с [Название компании]
4. Говори от первого лица, как реальный HR-специалист
5. Задавай конкретные вопросы без заполнителей
6. Если нужно уточнить детали - спрашивай напрямую

Задавай релевантные вопросы, анализируй ответы кандидата.
Если ответ неполный или неточный - задай уточняющий вопрос.
Если уверен в компетенции - переходи к следующей теме.
Максимум 10 вопросов с каждой стороны.
"""
}

async def call_llm(messages: List[Dict[str, str]], system_prompt: str, max_retries: int = 3) -> tuple[str, bool]:
    """
    Вызов LLM модели через API для анализа ответа кандидата
    
    Args:
        messages: Список сообщений для контекста
        system_prompt: Системный промпт
        max_retries: Максимальное количество попыток при ошибке
        
    Returns:
        tuple: (response_text, is_confident)
        - response_text: ответ от LLM
        - is_confident: True если LLM уверена, False если нужно уточнение
    """
    # Формируем полный контекст для LLM один раз
    full_messages = [
        {"role": "system", "content": system_prompt}
    ] + messages
    
    # Добавляем инструкцию для определения уверенности
    full_messages.append({
        "role": "system", 
        "content": """
        ВАЖНО: Строго следуй этим правилам:
        1. НИКОГДА не используй заполнители в квадратных скобках [текст]
        2. Не используй шаблонные фразы с [Название компании]
        3. Говори от первого лица, как реальный HR-специалист
        4. Задавай конкретные вопросы без заполнителей
        5. Если нужно уточнить детали - спрашивай напрямую
        6. Не используй квадратные скобки вообще, кроме меток CONFIDENT/UNCERTAIN
        7. Не используй шаблонные фразы и клише
        8. Пиши естественным языком, как реальный человек
        
        В конце своего ответа добавь метку:
        CONFIDENT - если ты уверен в оценке и готов задать следующий вопрос
        UNCERTAIN - если нужно уточнение или ответ кандидата неполный
        
        Квадратные скобки можно использовать ТОЛЬКО для меток CONFIDENT и UNCERTAIN.
        """
    })
    
    # Подготавливаем запрос к API
    body = {
        "messages": full_messages,
        "model": LLM_MODEL,
        "temperature": 0.8,
        "max_tokens": 500
    }
    
    for attempt in range(max_retries):
        try:
            # Создаем новую сессию для каждой попытки с увеличенным таймаутом
            timeout = aiohttp.ClientTimeout(total=60, connect=30)
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
            
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                # Выполняем запрос к LLM API
                async with session.post(LLM_API_URL, json=body) as response:
                    if response.status == 429:  # Too Many Requests
                        logger.warning("Превышен лимит запросов к API")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)  # Экспоненциальная задержка
                            continue
                        return "Извините, превышен лимит запросов. Пожалуйста, подождите немного и попробуйте снова.", False
                    
                    if response.status == 502:  # Bad Gateway
                        logger.warning(f"Получена ошибка 502 (попытка {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)  # Экспоненциальная задержка
                            continue
                        return "Извините, сервер временно недоступен. Пожалуйста, повторите ваш вопрос через несколько секунд.", False
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ошибка API: {response.status}. Ответ сервера: {error_text}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return "Извините, произошла техническая ошибка при обработке запроса.", False
                    
                    data = await response.json()
                    response_text = data['choices'][0]['message']['content'].strip()
                
                # Определяем уверенность LLM
                is_confident = "CONFIDENT" in response_text
                
                # Убираем метки из финального ответа
                response_text = response_text.replace("CONFIDENT", "").replace("UNCERTAIN", "").strip()
                
                # Проверяем на наличие заполнителей в квадратных скобках
                if "[" in response_text and "]" in response_text and not any(tag in response_text for tag in ["[Data Scientist]", "[Data Engineer]", "[Data Analyst]", "[MLOps Engineer]", "[Project Manager]", "[Некомпетентный соискатель]"]):
                    # Если нашли заполнители (но не финальные метки), очищаем их
                    import re
                    response_text = re.sub(r'\[.*?\]', '', response_text).strip()
                
                return response_text, is_confident
                
        except asyncio.TimeoutError:
            logger.error(f"Таймаут при запросе к LLM API (попытка {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
        except aiohttp.ClientError as e:
            logger.error(f"Ошибка подключения к LLM API: {e} (попытка {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
        except Exception as e:
            logger.error(f"Неизвестная ошибка при обращении к LLM (попытка {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
    
    # Если все попытки исчерпаны
    return "Извините, не удалось получить ответ после нескольких попыток. Пожалуйста, попробуйте позже.", False 

async def start_interview_flow(message: Message, state: FSMContext):
    """Общая функция начала собеседования"""
    user_id = message.from_user.id
    
    # Очищаем предыдущую сессию если есть
    if user_id in interview_sessions:
        del interview_sessions[user_id]
    
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=[
            [types.KeyboardButton(text=Position.DATA_SCIENTIST.value)],
            [types.KeyboardButton(text=Position.DATA_ENGINEER.value)],
            [types.KeyboardButton(text=Position.DATA_ANALYST.value)],
            [types.KeyboardButton(text=Position.MLOPS_ENGINEER.value)],
            [types.KeyboardButton(text=Position.PROJECT_MANAGER.value)]
        ],
        resize_keyboard=True,
        one_time_keyboard=True
    )
    
    await message.answer(
        "Добро пожаловать на HR-собеседование!\n\n"
        "Выберите позицию, на которую вы претендуете:",
        reply_markup=keyboard
    )
    await state.set_state(InterviewState.initial)

def determine_position(text: str) -> Optional[str]:
    """Определяет позицию из текста сообщения"""
    text = text.lower().strip()
    
    # Убираем префикс --0-- если он есть
    if text.startswith("--0--"):
        text = text[5:].strip()
    
    # Проверяем, что сообщение содержит ключевые слова о поиске работы
    job_keywords = [
        "хочу работать", "ищу работу", "ищу позицию", "хочу стать",
        "интересна работа", "интересна позиция", "хочу устроиться",
        "ищу вакансию", "хочу вакансию", "ищу должность", "претендую на вакансию",
        "готов к собеседованию", "хочу пройти собеседование"
    ]
    
    # Проверяем наличие ключевых слов о поиске работы
    has_job_keywords = any(keyword in text for keyword in job_keywords)
    
    # Словарь ключевых слов для каждой позиции
    position_keywords = {
        Position.DATA_SCIENTIST.value: [
            "data scientist", "специалист по данным", "машинное обучение",
            "ml", "искусственный интеллект", "ai", "наука о данных",
            "data science", "аналитик данных", "ml engineer", "специалист по машинному обучению"
        ],
        Position.DATA_ENGINEER.value: [
            "data engineer", "инженер данных", "etl", "пайплайн",
            "data pipeline", "инфраструктура данных", "big data", "инженер по данным"
        ],
        Position.DATA_ANALYST.value: [
            "data analyst", "аналитик данных", "бизнес-аналитик",
            "business analyst", "аналитик", "bi analyst", "специалист по анализу данных"
        ],
        Position.MLOPS_ENGINEER.value: [
            "mlops", "ml ops", "ml-ops", "ml engineer", "инженер ml",
            "devops ml", "ml devops", "ml инженер", "инженер devops для машинного обучения",
            "инженер devops ml", "инженер по машинному обучению", "devops для ml"
        ],
        Position.PROJECT_MANAGER.value: [
            "project manager", "менеджер проектов", "руководитель проектов",
            "проджект менеджер", "pm", "project lead", "менеджер по проектам"
        ]
    }
    
    # Если есть ключевые слова о поиске работы, ищем позицию
    if has_job_keywords:
        # Проверяем совпадение с ключевыми словами
        for position, keywords in position_keywords.items():
            if any(keyword in text for keyword in keywords):
                return position
    
    return None

@dp.message()
async def handle_any_message(message: Message, state: FSMContext):
    """Обработчик для всех сообщений"""
    # Проверяем, что сообщение не из канала
    if message.chat.type == "channel":
        return
        
    # Игнорируем команды /start
    if message.text and message.text.startswith('/start'):
        return
        
    current_state = await state.get_state()
    
    # Если состояние не установлено или мы в начальном состоянии
    if not current_state or current_state == InterviewState.initial.state:
        # Определяем позицию из текста сообщения
        position = determine_position(message.text)
        
        if position:
            # Если позиция определена, начинаем собеседование
            await state.set_state(InterviewState.interviewing)
            
            # Создаем сессию собеседования
            interview_sessions[message.from_user.id] = InterviewSession(
                user_id=message.from_user.id,
                position=position,
                messages=[{"role": "user", "content": message.text}]
            )
            
            # Задаем первый вопрос
            system_prompt = POSITION_PROMPTS[position]
            first_question, _ = await call_llm(
                messages=[{"role": "user", "content": f"Начни собеседование с учетом сообщения кандидата: {message.text}"}],
                system_prompt=system_prompt
            )
            
            await message.answer(first_question)
            
            # Сохраняем в историю
            interview_sessions[message.from_user.id].messages.append({
                "role": "assistant",
                "content": first_question
            })
        else:
            # Если позиция не определена, просто игнорируем сообщение
            return
    elif current_state == InterviewState.interviewing.state:
        # Обрабатываем ответы во время собеседования
        await process_interview_response(message, state)
    else:
        # Для остальных состояний просто игнорируем сообщение
        return

@dp.message(InterviewState.interviewing)
async def process_interview_response(message: Message, state: FSMContext):
    """Обработка ответов кандидата во время собеседования"""
    user_id = message.from_user.id
    user_response = message.text
    
    # Получаем сессию
    session = interview_sessions.get(user_id)
    if not session:
        await message.answer("Сессия собеседования не найдена.")
        return
    
    # Добавляем ответ пользователя в историю
    session.messages.append({"role": "user", "content": user_response})
    session.question_count += 1
    
    # Логируем только счетчик вопросов
    logger.info(f"Текущий счетчик вопросов: {session.question_count}/{session.max_questions}")
    
    # Проверяем лимит вопросов
    if session.question_count >= session.max_questions:
        logger.info("Достигнут лимит вопросов, завершаем собеседование")
        await finish_interview(message, state, session)
        return
    
    # Отправляем в LLM для анализа
    system_prompt = POSITION_PROMPTS[session.position]
    
    # Показываем, что бот думает
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    
    llm_response, is_confident = await call_llm(
        messages=session.messages,
        system_prompt=system_prompt
    )
    
    # Если получили сообщение об ошибке, не добавляем его в историю
    if "Извините, произошла техническая ошибка" in llm_response:
        await message.answer(llm_response)
        return
    
    # Проверяем на наличие метки завершения в ответе LLM
    if "[FINAL]" in llm_response:
        await finish_interview(message, state, session)
        return
    
    # Сохраняем ответ LLM в историю
    session.messages.append({"role": "assistant", "content": llm_response})
    
    # Отправляем ответ пользователю
    await message.answer(llm_response)
    
    # Если LLM не уверена, добавляем уточняющий вопрос в тот же ответ
    if not is_confident:
        clarification = "Не могли бы вы дать более подробный ответ или привести конкретный пример?"
        # Добавляем уточняющий вопрос к основному ответу
        llm_response = f"{llm_response}\n\n{clarification}"
        # Обновляем последнее сообщение в истории
        session.messages[-1]["content"] = llm_response

async def finish_interview(message: Message, state: Optional[FSMContext], session: InterviewSession):
    """Завершение собеседования"""
    user_id = message.chat.id
    
    # Добавляем финальные сообщения в историю
    session.messages.append({
        "role": "assistant",
        "content": "Спасибо за ваши ответы. Теперь я проанализирую ваши ответы и дам финальную оценку."
    })
    session.messages.append({
        "role": "user",
        "content": "Дай финальную оценку кандидата на основе всего собеседования"
    })
    
    # Форматируем историю для LLM
    system_prompt = f"""
    {POSITION_PROMPTS[session.position]}
    
    Собеседование завершено. Проанализируй все ответы кандидата и дай:
    1. Общую оценку компетенций (по шкале 1-10)
    2. Сильные стороны кандидата
    3. Области для развития
    4. Рекомендацию по найму (подходит/не подходит/требует дополнительной оценки)
    
    ВАЖНО: 
    1. Кандидат претендовал на позицию {session.position}
    2. Оценивай соответствие каждой позиции по следующим критериям:
       - Технические навыки (40%)
       - Опыт работы (30%)
       - Мягкие навыки (20%)
       - Мотивация и интерес к позиции (10%)
    3. В конце ответа добавь процентное соотношение для каждой позиции в формате:
       Data Scientist = X%
       Data Engineer = X%
       Data Analyst = X%
       MLOps Engineer = X%
       Project Manager = X%
       Некомпетентный соискатель = X%
    
    Где X - процент соответствия кандидата каждой позиции. Сумма всех процентов должна быть равна 100%.
    Учитывай, что кандидат претендовал на позицию {session.position}, поэтому эта позиция должна иметь приоритет при оценке.
    """
    
    # Форматируем историю для LLM
    formatted_messages = []
    
    # Добавляем историю диалога
    formatted_messages.extend(session.messages)
    
    # Записываем историю в файл
    try:
        with open("History.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Собеседование на позицию: {session.position}\n")
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*50}\n\n")
            
            # Записываем системный промпт
            f.write(f"[SYSTEM]\n{system_prompt}\n\n")
            
            for msg in formatted_messages:
                role = msg['role'].upper()
                content = msg['content'].strip()
                # Убираем метки --N-- из сообщений пользователя
                if role == "USER" and content.startswith("--") and "--" in content[2:]:
                    content = content[content.find("--", 2) + 2:].strip()
                f.write(f"[{role}]\n{content}\n\n")
            
            f.write(f"{'='*50}\n\n")
    except Exception as e:
        logger.error(f"Ошибка при записи истории в файл: {e}")
    
    # Получаем финальную оценку
    final_evaluation, _ = await call_llm(
        messages=formatted_messages,
        system_prompt=system_prompt
    )
    
    # Извлекаем процентное соотношение из ответа
    raw_percentages = {}
    normalized_percentages = {}
    
    for position in [pos.value for pos in Position] + ["Некомпетентный соискатель"]:
        if f"{position} = " in final_evaluation:
            try:
                percent = int(final_evaluation.split(f"{position} = ")[1].split("%")[0])
                raw_percentages[position] = percent
                # Если это желаемая позиция, увеличиваем вес на 20%
                if position == session.position:
                    percent = min(100, int(percent * 1.2))
                normalized_percentages[position] = percent
            except (ValueError, IndexError):
                raw_percentages[position] = 0
                normalized_percentages[position] = 0
    
    # Нормализуем проценты, чтобы сумма была равна 100%
    total = sum(normalized_percentages.values())
    if total > 0:
        normalized_percentages = {k: int((v / total) * 100) for k, v in normalized_percentages.items()}
    
    # Определяем вердикт как позицию с максимальным процентом
    # Изменено: если "Некомпетентный соискатель" >= 15%, выводим только его как вердикт
    if normalized_percentages.get("Некомпетентный соискатель", 0) >= 15:
        verdict = "Некомпетентный соискатель"
        # Очищаем процентные соотношения, чтобы выводить только вердикт
        raw_percentage_str = ""
        normalized_percentage_str = ""
        final_evaluation = "Вердикт: [Некомпетентный соискатель]"
    else:
        verdict = max(normalized_percentages.items(), key=lambda x: x[1])[0]
        # Убираем процентное соотношение из текста оценки
        for position in raw_percentages:
            final_evaluation = final_evaluation.replace(f"{position} = {raw_percentages[position]}%", "").strip()
        # Формируем строки с процентным соотношением
        raw_percentage_str = "\n".join([f"{pos} = {pct}%" for pos, pct in raw_percentages.items()])
        normalized_percentage_str = "\n".join([f"{pos} = {pct}%" for pos, pct in normalized_percentages.items()])
    
    # Логируем процентное соотношение
    logger.info("\nИсходное процентное соотношение позиций:")
    logger.info(raw_percentage_str)
    logger.info("\nНормализованное процентное соотношение:")
    logger.info(normalized_percentage_str)
    logger.info(f"Финальный вердикт: {verdict}")
    logger.info(f"Желаемая позиция: {session.position}")
    
    # Записываем процентное соотношение в файл
    try:
        with open("History.txt", "a", encoding="utf-8") as f:
            f.write("\nИсходное процентное соотношение позиций:\n")
            f.write(raw_percentage_str + "\n")
            f.write("\nНормализованное процентное соотношение:\n")
            f.write(normalized_percentage_str + "\n")
            f.write(f"Финальный вердикт: {verdict}\n")
            f.write(f"Желаемая позиция: {session.position}\n")
            f.write(f"{'='*50}\n\n")
    except Exception as e:
        logger.error(f"Ошибка при записи процентного соотношения в файл: {e}")
    
    await message.answer(
        "Собеседование завершено!\n\n"
        "Спасибо за ваше время и ответы. Вот результаты оценки:\n\n"
        f"{final_evaluation}\n\n"
        f"Исходное процентное соотношение:\n{raw_percentage_str}\n\n"
        f"Нормализованное процентное соотношение:\n{normalized_percentage_str}\n\n"
        f"Вердикт: [{verdict}]\n\n"
    )
    
    # Очищаем сессию
    del interview_sessions[user_id]
    if state:
        await state.set_state(InterviewState.finished)

@dp.channel_post()
async def handle_channel_message(message: Message):
    """Обработчик для сообщений из каналов"""
    # Сначала проверяем есть ли активная сессия
    session = interview_sessions.get(message.chat.id)
    if session:
        # Добавляем ответ в историю
        session.messages.append({"role": "user", "content": message.text})
        session.question_count += 1
        
        # Проверяем лимит вопросов
        if session.question_count >= session.max_questions:
            await finish_interview(message, None, session)
            return
        
        # Отправляем в LLM для анализа
        system_prompt = POSITION_PROMPTS[session.position]
        llm_response, is_confident = await call_llm(
            messages=session.messages,
            system_prompt=system_prompt
        )
        
        # Сохраняем ответ LLM в историю
        session.messages.append({"role": "assistant", "content": llm_response})
        
        # Отправляем ответ пользователю
        await message.answer(llm_response)
        return
    
    # Если нет активной сессии, проверяем на позицию
    position = determine_position(message.text)
    if position:
        # Создаем сессию собеседования
        interview_sessions[message.chat.id] = InterviewSession(
            user_id=message.chat.id,
            position=position,
            messages=[{"role": "user", "content": message.text}]
        )
        
        # Задаем первый вопрос
        system_prompt = POSITION_PROMPTS[position]
        first_question, _ = await call_llm(
            messages=[{"role": "user", "content": f"Начни собеседование с учетом сообщения кандидата: {message.text}"}],
            system_prompt=system_prompt
        )
        
        await message.answer(first_question)
        
        # Сохраняем в историю
        interview_sessions[message.chat.id].messages.append({
            "role": "assistant",
            "content": first_question
        })

async def main():
    """Запуск бота"""
    logger.info("Запуск HR-бота...")
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {e}")
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())