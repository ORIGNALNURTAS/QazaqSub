"""
Транскрипцияланған сегменттерді қазақ тіліне аударатын модуль.

Бұл нұсқада `deep-translator` кітапханасы арқылы Google Translate-тің тегін,
API кілтін талап етпейтін веб-соңғы нүктесі қолданылады. Бұл нақты жұмыс
істейтін шешім, бірақ екі шектеуі бар:

  1) Интернет байланысы қажет (сервер орналасқан машинада).
  2) Тегін болғандықтан, өте көп сұраныс жіберсеңіз, Google уақытша шектеу
     қоюы мүмкін (rate limit). Өндірістік/ауыр жүктеме үшін ақылы Google
     Cloud Translate API немесе DeepL API-ге көшкен жөн (төменде көрсетілген).

Ақылы/ресми API-ге көшу қажет болса, тек `_translate_text` функциясының
ішін өзгерту жеткілікті — қалған код өзгермейді:

  * Google Cloud Translate:
      from google.cloud import translate_v2 as translate
      client = translate.Client()
      return client.translate(text, target_language="kk")["translatedText"]

  * Ашық үлгі NLLB-200 (Meta), интернетсіз/локалды жұмыс үшін:
      from transformers import pipeline
      translator = pipeline("translation", model="facebook/nllb-200-distilled-600M",
                             src_lang="eng_Latn", tgt_lang="kaz_Cyrl")
      translator(text)[0]["translation_text"]
"""
from typing import List

from deep_translator import GoogleTranslator

from app.services.transcriber import Segment

# Кейбір Whisper тілдік кодтары deep-translator форматымен сәйкес келмейді.
_LANGUAGE_CODE_MAP = {
    "auto": "auto",
}


def translate_segments(
    segments: List[Segment],
    source_language: str = "en",
    target_language: str = "kk",
) -> List[Segment]:
    """
    Әр сегменттің мәтінін қазақ тіліне аударады.

    Егер бастапқы тіл қазақша болса ("kk"), аударма керек емес — сол
    қалпында қайтарылады.

    :param segments: transcriber.py модулінен алынған сегменттер тізімі
    :param source_language: Бастапқы мәтін тілі (Whisper анықтаған тіл коды)
    :param target_language: Мақсатты тіл (әдепкі — "kk", қазақша)
    :return: Аударылған мәтіні бар сегменттер тізімі
    """
    if source_language == target_language:
        return segments

    mapped_source = _LANGUAGE_CODE_MAP.get(source_language, source_language)
    translator = GoogleTranslator(source=mapped_source, target=target_language)

    translated: List[Segment] = []
    for seg in segments:
        translated.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": _translate_text(seg["text"], translator),
            }
        )
    return translated


def _translate_text(text: str, translator: GoogleTranslator) -> str:
    """
    Бір сегмент мәтінін нақты аударма қызметі арқылы аударады.
    Аударма қызметінде уақытша қате болса, түпнұсқа мәтінді қайтарамыз —
    осылайша бүкіл процесс бір сегменттің қатесінен тоқтап қалмайды.
    """
    text = text.strip()
    if not text:
        return text
    try:
        result = translator.translate(text)
        return result if result else text
    except Exception:
        # Желі/API уақытша қолжетімсіз болса, түпнұсқа мәтінмен жалғастырамыз
        return text
