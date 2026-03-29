import time
import logging
import mlflow

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "rag-knowledge-assistant"


def get_or_create_experiment() -> str:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


def log_query(question: str, answer: str, sources: list[dict], latency_seconds: float) -> str:
    """Log a single query run to MLflow. Returns run_id."""
    experiment_id = get_or_create_experiment()

    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.log_param("question", question[:500])  # param value limit
        mlflow.log_param("num_sources", len(sources))
        mlflow.log_param("source_files", ", ".join({s["source"] for s in sources}))

        mlflow.log_metric("latency_seconds", latency_seconds)
        mlflow.log_metric("answer_length_chars", len(answer))
        mlflow.log_metric("sources_returned", len(sources))

        # write full answer as artifact for manual inspection
        with open("answer.txt", "w") as f:
            f.write(f"Question:\n{question}\n\nAnswer:\n{answer}\n\nSources:\n")
            for s in sources:
                f.write(f"  - {s['source']} (page {s['page']})\n    {s['snippet']}\n\n")
        mlflow.log_artifact("answer.txt")

        run_id = run.info.run_id
        logger.info(f"MLflow run logged: {run_id} | latency={latency_seconds}s")
        return run_id


def ask_and_log(chain, question: str) -> dict:
    """Run ask() and log the result to MLflow. Returns ask() dict + run_id and latency."""
    from chains.qa_chain import ask

    start = time.time()
    result = ask(chain, question)
    latency = round(time.time() - start, 3)

    run_id = log_query(
        question=question,
        answer=result["answer"],
        sources=result["sources"],
        latency_seconds=latency,
    )
    result["run_id"] = run_id
    result["latency_seconds"] = latency
    return result


if __name__ == "__main__":
    from chains.qa_chain import build_chain

    print("Loading chain...")
    chain = build_chain()

    result = ask_and_log(chain, "What was Capital One's net income in 2024?")
    print(f"\nAnswer: {result['answer']}")
    print(f"Latency: {result['latency_seconds']}s")
    print(f"MLflow run_id: {result['run_id']}")
    print(f"\nRun: mlflow ui  (then open http://localhost:5000)")
