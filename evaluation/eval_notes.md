# Evaluation notes

This project is intentionally scoped for a ~4 hour build, so evaluation is lightweight and manual, including retrieval relevance,groundedness, citation faithfulness, and readability via chatgpt 5.4. If I had more time, I would have used less generic metrics and added an evaluation set.

## What I checked
1. **Grounding**
   - Does the answer stay inside the retrieved filing context?
   - Does each factual claim carry a citation?

2. **Retrieval relevance**
   - For risk-factor questions, do top chunks come from Item 1A or nearby discussion?
   - For revenue / outlook questions, do top chunks come from MD&A or financial statement discussion?
   - For regulatory questions, do top chunks surface Legal Proceedings, Risk Factors, or industry-specific sections?

3. **Comparison quality**
   - If the question names multiple companies, do retrieved chunks cover more than one company?
   - Does the final answer separate companies instead of blending them together?

4. **Time-awareness**
   - For change-over-time questions, does retrieval surface multiple filing dates?
   - Does the answer clearly distinguish older vs newer filings?

## Failure modes to watch
- Retrieval returns too many chunks from one company.
- The filing text contains a table of contents or repeated headers that pollute chunking.
- The question needs a section that wasn't retrieved because the query is phrased indirectly.

## Small improvements if I had more time
- use full sec parser
- add a local reranker
- tighten section parsing
- add company-name normalization
- add an evaluation set
- reduce latency via optimal indexing/ranking
- experiment with different chunk sizes, overlap sizes, and k values
