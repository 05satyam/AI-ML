# Interview Prep

A growing collection of interview preparation material for AI/ML engineering roles.
Each sub-folder is one interview cycle, named `prep-N-<company>` so they stack up
and stay easy to reference across job searches.

## Structure

```
interview-prep/
  prep-1-ai-engineering/  ← Staff AI Engineer panel (RAG, agents, k8s, evals)
    README.md           overview of the role + panel format
    cheatsheets/        per-round guides + deep technical themes
    behavioral/         STAR story bank
    drills/             rapid-fire Q&A with model answers
    mock_round3/        runnable buggy codebase + TASK.md to practice on
    rag_service/        production-style RAG reference service (runnable)

  interview-expereince/ ← existing general Q&A notes
    AI-ML-QnA.md

  comparison_of_major_llms_architectures(2017-2025)/
    comparison_of_major_llms_architectures(2017-2025).md
    comparison_of_major_llms_architectures(2017-2025).pdf
    comparative_thumbnail.png
    images/             logos + architecture diagram for the comparison doc
```

## Reusable across any AI engineering interview

The themes covered in `prep-1-ai-engineering` — RAG, conversational memory, agentic systems,
context engineering, Kubernetes scalability, evaluation, observability — come up in
virtually every Staff/Senior AI engineer panel. The cheatsheets and drill doc are
worth reviewing before any role at that level.

## Adding a new prep cycle

Copy the `prep-1-ai-engineering` folder as `prep-2-<company>`, update the README inside it,
and tailor the `behavioral/star_stories.md` and `cheatsheets/` to the new JD.
