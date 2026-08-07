import json
import csv
import time
import argparse

from openai import OpenAI, RateLimitError, APIError, APIConnectionError

# ── Argumentos de linha de comando ────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Avalia respostas via API da OpenAI.")
parser.add_argument("--rounds", "-r", type=int, default=1)
args = parser.parse_args()
ROUNDS = args.rounds

# ══════════════════════════════════════════════════════════════════════════════
TARGET_QUESTION_ID     = "acidsBase_300"
DELAY_BETWEEN_REQUESTS = 3   # segundos entre requisições normais
MAX_RETRIES            = 5   # tentativas máximas por resposta em caso de rate limit

DEFAULT_MODEL          = "gpt-5.6"   # modelo da OpenAI a ser usado
TEMPERATURE             = 0.0
MAX_TOKENS              = 500        # usado apenas se o modelo não for de raciocínio
# ══════════════════════════════════════════════════════════════════════════════

with open("llm-q_and_a/questions.json", encoding="utf-8") as f:
    questions_raw = json.load(f)

with open("llm-q_and_a/simulatedAnswers.json", encoding="utf-8") as f:
    answers_raw = json.load(f)

question_data = next(
    (q for q in questions_raw["questions"] if q["globalId"] == TARGET_QUESTION_ID), None
)
if question_data is None:
    ids = [q["globalId"] for q in questions_raw["questions"]]
    raise ValueError(f"Questão '{TARGET_QUESTION_ID}' não encontrada. IDs disponíveis: {ids}")

answer_data = next(
    (a for a in answers_raw["responsesByQuestion"] if a["questionGlobalId"] == TARGET_QUESTION_ID), None
)
if answer_data is None:
    raise ValueError(f"Respostas para '{TARGET_QUESTION_ID}' não encontradas em simulatedAnswers.json.")

# ── Monta rubrica ──────────────────────────────────────────────────────────────
def build_rubric_block(rubric: dict) -> str:
    lines = []
    for c in rubric["requiredConcepts"]:
        lines.append(f"[REQ {c['id']} w={c['weight']}] {c['description']}")
    for c in rubric["partialConcepts"]:
        lines.append(f"[PAR {c['id']} w={c['weight']}] {c['description']}")
    for c in rubric["misconceptions"]:
        lines.append(f"[MIS {c['id']} w={c['weight']}] {c['description']}")
    return "\n".join(lines)

RUBRIC_TEXT   = build_rubric_block(question_data["openAnswerRubric"])
QUESTION_TEXT = question_data["questionText"]
MAX_SCORE     = (
    sum(c["weight"] for c in question_data["openAnswerRubric"]["requiredConcepts"]) +
    sum(c["weight"] for c in question_data["openAnswerRubric"]["partialConcepts"])
)

SYSTEM_PROMPT = (
    "You are a strict academic evaluator. "
    "Score the student answer on a 0-10 scale using ONLY the rubric provided. "
    "Apply weights exactly: sum matched required/partial weights, subtract matched misconception weights, "
    f"then normalize to 0-10 (max positive weight = {MAX_SCORE:.2f}). "
    "Respond with a single JSON object and nothing else: {\"score\": <number 0-10>}"
)

def build_user_message(student_answer: str) -> str:
    return (
        f"QUESTION: {QUESTION_TEXT}\n\n"
        f"RUBRIC:\n{RUBRIC_TEXT}\n\n"
        f"STUDENT ANSWER: {student_answer}"
    )

# ── Detecção de reasoning model ───────────────────────────────────────────────
# Modelos de raciocínio da OpenAI (o1, o3, o4, série gpt-5, etc.) não aceitam
# o parâmetro "temperature" e costumam precisar de mais tokens de saída.
REASONING_KEYWORDS = ("o1", "o3", "o4", "o5", "gpt-5", "reasoning", "thinking")
is_reasoning_model = any(k in DEFAULT_MODEL.lower() for k in REASONING_KEYWORDS)
effective_max_tokens = 4000 if is_reasoning_model else MAX_TOKENS

# ── Cliente OpenAI ─────────────────────────────────────────────────────────────
# A API key é lida automaticamente da variável de ambiente OPENAI_API_KEY
# (comportamento padrão do SDK).
# client = OpenAI(api_key="your-api-here")

# ── Info inicial ───────────────────────────────────────────────────────────────
print(f"{'='*60}")
print(f"  Questão  : {TARGET_QUESTION_ID}")
print(f"  Modelo   : {DEFAULT_MODEL}")
print(f"  Rodadas  : {ROUNDS}")
print(f"  Respostas: {len(answer_data['responses'])}")
print(f"  Max tok  : {effective_max_tokens}")
print(f"  Delay    : {DELAY_BETWEEN_REQUESTS}s | Retries: {MAX_RETRIES}")
if is_reasoning_model:
    print(f"  [INFO] Reasoning model detectado — temperature omitido")
print(f"{'='*60}\n")

# ── Chamada à API com retry automático em rate limit ───────────────────────────
def call_openai(student_answer: str, resp_id: str) -> tuple[float | None, float]:
    total_elapsed = 0.0

    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.perf_counter()
        try:
            request_kwargs = {
                "model": DEFAULT_MODEL,
                "instructions": SYSTEM_PROMPT,
                "input": build_user_message(student_answer),
                "max_output_tokens": effective_max_tokens,
            }
            if not is_reasoning_model:
                request_kwargs["temperature"] = TEMPERATURE

            response = client.responses.create(**request_kwargs)
            elapsed = time.perf_counter() - t0
            total_elapsed += elapsed

            if getattr(response, "status", None) == "incomplete":
                reason = getattr(response, "incomplete_details", None)
                print(f"  [AVISO] {resp_id}: resposta incompleta ({reason}) — aumente max_output_tokens além de {effective_max_tokens}")

            raw = (response.output_text or "").strip().strip("`").strip()
            if not raw:
                print(f"  [AVISO] {resp_id}: resposta vazia.")
                return None, round(total_elapsed, 3)

            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

            score = float(json.loads(raw)["score"])
            return round(score, 2), round(total_elapsed, 3)

        except RateLimitError as e:
            elapsed = time.perf_counter() - t0
            total_elapsed += elapsed
            wait = 10
            try:
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    wait = int(float(retry_after)) + 1
            except Exception:
                pass
            print(f"  [429] {resp_id} — tentativa {attempt}/{MAX_RETRIES}, aguardando {wait}s...")
            time.sleep(wait)
            continue

        except (APIError, APIConnectionError) as e:
            elapsed = time.perf_counter() - t0
            total_elapsed += elapsed
            print(f"  [ERRO API] {resp_id}: {e}")
            return None, round(total_elapsed, 3)

        except Exception as e:
            elapsed = time.perf_counter() - t0
            total_elapsed += elapsed
            print(f"  [ERRO PARSE/CONEXÃO] {resp_id}: {e}")
            return None, round(total_elapsed, 3)

    print(f"  [FALHA] {resp_id}: esgotou {MAX_RETRIES} tentativas.")
    return None, round(total_elapsed, 3)

# ── Loop de avaliação ──────────────────────────────────────────────────────────
OUTPUT_CSV = f"evaluation_{TARGET_QUESTION_ID}.csv"
total_errors = 0

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["round", "responseId", "targetScore", "modelScore", "processingTime_s"])

    for round_num in range(1, ROUNDS + 1):
        print(f"--- Rodada {round_num}/{ROUNDS} ---")
        for resp in answer_data["responses"]:
            rid          = resp["responseId"]
            target_score = round(resp["targetScorePercent"] / 10, 1)
            print(f"  Avaliando {rid} (alvo={target_score})...", end=" ", flush=True)

            model_score, elapsed = call_openai(resp["studentAnswer"], rid)

            if model_score is not None:
                print(f"→ {model_score} ({elapsed}s)")
            else:
                total_errors += 1
                print(f"→ ERROR ({elapsed}s)")

            writer.writerow([
                round_num, rid, target_score,
                model_score if model_score is not None else "ERROR",
                elapsed,
            ])
            time.sleep(DELAY_BETWEEN_REQUESTS)
        print()

print(f"{'='*60}")
print(f"  Concluído. CSV salvo em: {OUTPUT_CSV}")
print(f"  Erros: {total_errors}/{ROUNDS * len(answer_data['responses'])}")
print(f"{'='*60}")