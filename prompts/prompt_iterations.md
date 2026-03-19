# Prompt iteration log

## Iteration 1
**Prompt idea:** “Answer the question from the retrieved SEC chunks.”

**What went wrong:**
- Too generic.
- The model sometimes answered in a broad business-summary style instead of grounding itself in the chunks.
- It also tended to compress multi-company comparisons into one blended paragraph.

## Iteration 2
**Change:** Added hard grounding instructions:
- “Use ONLY the retrieved context.”
- “If the context is insufficient, say so.”

**Why:**
- Reduced unsupported claims.
- Made the model more honest when retrieval missed something.

**Tradeoff:**
- The answer became safer, but sometimes too short or overly cautious.

## Iteration 3
**Change:** Added stronger anti-hallucination and company-boundary rules:
- “Do not use outside knowledge.”
- “Do not invent facts, numbers, or trends.”
- “Never use evidence from one company to make a claim about a different company.”
- “Use the exact company labels from the retrieved context.”

**Why:**
- The model sometimes filled gaps with plausible business language.
- In multi-company answers, it occasionally mixed evidence across companies or renamed companies incorrectly.

**Tradeoff:**
- Improved trustworthiness and grounding.
- Slightly more rigid tone.

## Iteration 4
**Change:** Added response/mitigation constraint and explicit trend handling:
- “Only describe mitigation or response if the retrieved text explicitly supports it.”
- “If evidence is weak or generic, keep the claim narrow and conservative.”
- Prefer recent filings for trend questions.

**Why:**
- The model was stronger on identifying risks than on accurately describing how companies were responding.
- Temporal questions needed change-over-time framing rather than static summaries.

**Tradeoff:**
- More conservative answers.
- Less speculative detail, especially when evidence was thin.

## Iteration 5 (final)
**Change:** Added explicit markdown structure:
- `## Direct answer`
- `## Comparison / breakdown`
- `## Evidence from filings`

Also added:
- “Do not ask for the question or context again.”
- “Use the actual bracketed chunk labels instead of Context 1 / Context 2.”

**Why:**
- Best balance of groundedness, readability, and demo-friendliness.
- Keeps answers auditable and easy to follow live.
- Works reasonably well across the main query types:
  - named-company comparisons
  - category/regulatory questions
  - single-company trend questions