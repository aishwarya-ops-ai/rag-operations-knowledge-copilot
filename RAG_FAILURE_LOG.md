# RAG Failure Log

This document is a transparent diagnostic record for the Operations Knowledge
Copilot. It is not a formal production benchmark and it does not claim that each
suggested fix will improve the system.

## Evaluation snapshot

- Dataset: `data/evaluation_questions.csv`
- Cases: 25 total — 19 answerable and 6 unanswerable
- Retrieval configuration: cosine similarity, top-k 4
- Answerability rule: best similarity must be at least `0.50`
- Canonical evaluation result: 100.0% retrieval hit rate, 88.0% answerability
  accuracy, 50.0% unsupported-answer risk, and 94.7% top-source match rate
- Paired robustness result: 100.0% canonical query accuracy, 73.3% paraphrase
  retrieval success, and 50.0% unsupported-question refusal accuracy
- Chunk indexes below are zero-based, matching the CLI and stored metadata.

The log includes every failed case plus a deliberately defined set of weak
passes. A weak pass is a case where the expected source ranked below first, or a
correct answerability decision was within `0.05` of the `0.50` threshold. Under
that definition, this run has five failures and two weak passes.

The paraphrase extension changed only evaluation data and reporting. No SOP,
chunking, embedding, top-k, or threshold change was made to improve these scores.

## Paraphrase robustness failures

The 30-case paraphrase set contains eight failures. Seven retrieved the expected
source within top-k but remained below the `0.50` evidence threshold: six at
rank 1 and one at rank 2. One trainee refund question crossed the threshold
using refund-policy evidence but did not retrieve the expected training source
within top-k.

| Case | Expected source | Expected-source rank | Top score | Failure summary |
| --- | --- | ---: | ---: | --- |
| `q-07-p3` | Customer Escalation Policy | 1 | 0.4239 | Correct source, weak similarity for “route a case” wording. |
| `q-10-p2` | Customer Escalation Policy | 1 | 0.4748 | Correct source, ownership phrasing fell below the gate. |
| `q-10-p3` | Customer Escalation Policy | 1 | 0.4632 | Correct source, responsibility wording fell below the gate. |
| `q-12-p1` | Quality Review Guidelines | 1 | 0.4878 | “QA reviews” and “selection mix” ranked correctly but weakly. |
| `q-12-p2` | Quality Review Guidelines | 1 | 0.4420 | “Quality audits” wording weakened semantic similarity. |
| `q-14-p2` | Quality Review Guidelines | 1 | 0.4456 | “QA appeal” retrieved correctly but did not pass the gate. |
| `q-22-p2` | New Employee Training SOP | 2 | 0.4639 | Training source appeared, but the entire result stayed below the gate. |
| `q-22-p3` | New Employee Training SOP | Not in top 4 | 0.5761 | Refund authority outranked the decisive trainee restriction. |

Likely causes include short or compressed paraphrases, vocabulary shifts such
as “audit” versus “review,” and cross-document questions where refund language
outweighs trainee status. Possible future fixes include a local reranker,
hybrid lexical-semantic retrieval, or narrowly expanded aliases. These are
recorded as candidates only; no automatic retrieval tuning was applied.

## Case q-11 — Failed: unsupported phone-number question passed the gate

- **Question:** What phone number should employees call when the designated
  Incident Response contact cannot be reached?
- **Expected source:** None — the five SOPs do not provide a phone number.
- **Retrieved sources:**
  1. `01_Attendance_Shift_CheckIn_SOP.txt`, chunk 4 — `0.5234`
  2. `01_Attendance_Shift_CheckIn_SOP.txt`, chunk 5 — `0.4709`
  3. `02_Customer_Escalation_Policy.txt`, chunk 2 — `0.4341`
  4. `01_Attendance_Shift_CheckIn_SOP.txt`, chunk 3 — `0.4115`
- **What went wrong:** The top result exceeded the answerability threshold, so
  the system classified the question as answerable. The highest-ranked source
  was also the attendance SOP rather than the escalation policy that mentions
  the Incident Response contact. Neither document contains the requested phone
  number.
- **Likely cause:** Ambiguous operational language such as “employees,” “call,”
  “contact,” and “cannot be reached” is semantically close to attendance and
  emergency-contact passages. This is mainly a missing-detail problem masked by
  embedding similarity; a single global threshold cannot verify that a phone
  number is present.
- **Possible fix:** Add a post-retrieval support check that verifies the requested
  attribute exists before generation. A hybrid keyword check for phone-number
  patterns or a local reranker could help. Raising the global threshold might
  reject this case, but should be tested against answerable borderline cases
  before adoption.

## Case q-13 — Failed: correct quality evidence fell below the gate

- **Question:** An employee receives a numerical quality score of 94%, but the
  review found that they disclosed protected customer information without
  authorization. Does the case pass?
- **Expected source:** `03_Quality_Review_Guidelines.txt` — Quality Review
  Guidelines.
- **Retrieved sources:**
  1. `03_Quality_Review_Guidelines.txt`, chunk 5 — `0.4661`
  2. `03_Quality_Review_Guidelines.txt`, chunk 2 — `0.4478`
  3. `03_Quality_Review_Guidelines.txt`, chunk 3 — `0.4416`
  4. `03_Quality_Review_Guidelines.txt`, chunk 4 — `0.4391`
- **What went wrong:** Retrieval selected the correct document in all four
  positions, including the passage explaining that protected-information
  disclosure is a Critical Error. However, the best score was below `0.50`, so
  the system incorrectly refused an answerable question.
- **Likely cause:** Weak embedding similarity on a compound scenario. The query
  combines a passing numerical score, protected-information disclosure, and an
  exception that overrides the score. The relevant rule may also be diluted by
  surrounding scoring content inside a relatively large chunk.
- **Possible fix:** Evaluate smaller, section-aware chunks around “Critical
  Errors,” or add hybrid lexical retrieval for distinctive phrases such as
  “protected customer information.” A local cross-encoder reranker could score
  the query-rule relationship more accurately. Lowering the global threshold
  alone would worsen unsupported-answer risk elsewhere.

## Case q-16 — Failed: missing bonus detail looked answerable

- **Question:** What exact monetary bonus does a Quality Analyst receive for
  maintaining more than 95% gold-label agreement?
- **Expected source:** None — compensation and bonuses are not covered.
- **Retrieved sources:**
  1. `03_Quality_Review_Guidelines.txt`, chunk 4 — `0.5346`
  2. `03_Quality_Review_Guidelines.txt`, chunk 5 — `0.4575`
  3. `03_Quality_Review_Guidelines.txt`, chunk 1 — `0.4043`
  4. `03_Quality_Review_Guidelines.txt`, chunk 0 — `0.3712`
- **What went wrong:** The evaluator correctly found the document's
  gold-standard calibration section, but that section defines accuracy targets,
  not a monetary bonus. Topical similarity pushed the case above the threshold
  and produced an unsupported-answer classification.
- **Likely cause:** Missing source content combined with strong lexical and
  semantic overlap: “Quality Analyst,” percentage, and “gold-label agreement”
  are all genuinely present. Embeddings detect subject relevance, not whether
  the requested compensation field exists.
- **Possible fix:** Add a support/entailment check after retrieval that can
  distinguish “the topic is present” from “the requested fact is present.” Keep
  this negative case in the evaluation set. Chunking changes are unlikely to
  solve it by themselves.

## Case q-20 — Failed: partially related refund evidence looked highly answerable

- **Question:** After Northstar submits an approved card refund to the payment
  processor, which bank will return the money to the customer fastest?
- **Expected source:** None — the refund policy gives a timing range but does not
  rank or name banks.
- **Retrieved sources:**
  1. `04_Refund_Exception_Policy.txt`, chunk 4 — `0.7002`
  2. `04_Refund_Exception_Policy.txt`, chunk 1 — `0.6075`
  3. `04_Refund_Exception_Policy.txt`, chunk 2 — `0.5638`
  4. `04_Refund_Exception_Policy.txt`, chunk 0 — `0.5490`
- **What went wrong:** Retrieval strongly matched the refund-timing passage, but
  the passage only says timing depends on the customer's financial institution.
  It cannot answer which bank is fastest. The `0.7002` score easily passed the
  answerability gate.
- **Likely cause:** This is a partial-support boundary case, not primarily a bad
  embedding result. The document contains nearly every concept in the question
  except the requested comparison. Repeated refund chunks in the top four also
  provide little evidence diversity, although overlap is not proven to be the
  root cause.
- **Possible fix:** Require a post-retrieval support verifier or grounded model
  refusal for the exact requested claim. A higher similarity threshold would not
  fix this case. Diversity-aware retrieval could reduce repetitive evidence but
  would not create the missing bank comparison.

## Case q-25 — Failed: generic employee-policy chunks crossed the gate

- **Question:** What health-insurance provider does Northstar use for new
  employees, and when does coverage begin?
- **Expected source:** None — insurance providers and coverage dates are absent.
- **Retrieved sources:**
  1. `03_Quality_Review_Guidelines.txt`, chunk 0 — `0.5065`
  2. `05_New_Employee_Training_SOP.txt`, chunk 0 — `0.5014`
  3. `04_Refund_Exception_Policy.txt`, chunk 0 — `0.4635`
  4. `04_Refund_Exception_Policy.txt`, chunk 1 — `0.4319`
- **What went wrong:** Two broad introductory chunks narrowly exceeded `0.50`,
  so the system labeled the question answerable even though none of the chunks
  mentions insurance coverage.
- **Likely cause:** Borderline embedding similarity driven by generic terms such
  as “Northstar,” “employee,” and “new employee.” Introductory boilerplate is
  broad enough to match many organization-level questions. The requested source
  content is missing.
- **Possible fix:** Down-weight document headers and generic introductory text,
  add lexical coverage or support verification for key requested concepts, or
  introduce a narrow uncertainty band around the threshold. Any threshold
  adjustment must be checked against q-13, which is a genuine answerable case
  below `0.50`.

## Case q-22 — Weak pass: expected training source ranked third

- **Question:** During supervised production, a trainee is asked to independently
  process a $500 refund because the queue is understaffed. Is that permitted?
- **Expected source:** `05_New_Employee_Training_SOP.txt` — New Employee Training
  SOP.
- **Retrieved sources:**
  1. `04_Refund_Exception_Policy.txt`, chunk 2 — `0.5367`
  2. `04_Refund_Exception_Policy.txt`, chunk 5 — `0.4999`
  3. `05_New_Employee_Training_SOP.txt`, chunk 3 — `0.4868`
  4. `02_Customer_Escalation_Policy.txt`, chunk 5 — `0.4751`
- **What went wrong:** The case was classified answerable and the expected source
  appeared within top-k, but the most directly applicable training restriction
  ranked behind two refund-policy chunks. A top-k of 2 would have missed the
  expected source entirely, and generation could over-focus on ordinary refund
  approval authority rather than trainee restrictions.
- **Likely cause:** This is a cross-document, multi-intent question. The amount
  and repeated refund language dominate the embedding, while role and training
  context carry the decisive rule. Top-k 4 prevented a retrieval miss; top-k is
  not currently too low, but the ranking is weak.
- **Possible fix:** Add a local reranker, query decomposition, or hybrid retrieval
  that gives more weight to “trainee,” “supervised production,” and
  “understaffed.” Do not simply increase top-k without checking whether extra
  weak context reduces grounding quality.

## Case q-24 — Weak pass: correct refusal was close to the threshold

- **Question:** What brand and model of laptop should a new Northstar employee
  receive during onboarding?
- **Expected source:** None — hardware brands and models are not covered.
- **Retrieved sources:**
  1. `05_New_Employee_Training_SOP.txt`, chunk 0 — `0.4730`
  2. `03_Quality_Review_Guidelines.txt`, chunk 0 — `0.3730`
  3. `01_Attendance_Shift_CheckIn_SOP.txt`, chunk 0 — `0.3351`
  4. `04_Refund_Exception_Policy.txt`, chunk 0 — `0.3125`
- **What went wrong:** The system made the correct unanswerable decision, but the
  top score was only `0.027` below the threshold. A small embedding or corpus
  change could flip the result.
- **Likely cause:** “New employee” and “onboarding” legitimately match the
  training SOP introduction, even though laptop specifications are absent. This
  is another missing-detail case amplified by broad introductory content.
- **Possible fix:** Treat near-threshold decisions as a review/uncertainty zone,
  or verify that requested entities such as brand/model values occur in the
  evidence. This case should remain in regression testing even though it passes
  today.

## Cross-case observations

- **Chunking:** q-13 may benefit from a smaller, section-aware Critical Errors
  chunk. There is not enough evidence to blame chunk size globally.
- **Ambiguous wording:** q-11 and q-22 contain terms that plausibly belong to
  multiple policies, which weakens ranking.
- **Weak embedding similarity:** q-13 is a false negative despite retrieving the
  correct source; q-25 is a borderline false positive on broad policy text.
- **Missing source content:** q-11, q-16, q-20, q-24, and q-25 ask for a detail
  that the corpus does not provide. Topic similarity alone cannot detect this.
- **Overlap and result diversity:** q-16 and q-20 return several chunks from the
  same document. This may crowd out diverse evidence, but the current run does
  not prove that the 240-character overlap caused the failures.
- **Top-k:** top-k 4 rescued q-22 because the expected source ranked third. There
  is no evidence yet that raising top-k further would improve answerability, and
  additional context could make support validation harder.
- **Threshold:** raising `0.50` would fix some borderline false positives but
  would deepen the q-13 false negative; it would not fix q-20 at `0.7002`.

## Next investigation candidates

These are experiments to compare, not changes approved by this log:

1. Add a deterministic or local-model support-verification step after retrieval.
2. Compare current chunking with section-aware chunks on the same 25 cases.
3. Test hybrid semantic plus keyword retrieval, especially for q-13 and q-22.
4. Test a local reranker and measure whether it promotes the expected source for
   q-22 without reducing the existing 100% top-k retrieval hit rate.
5. Add more unanswerable cases before changing the threshold; six negative cases
   are too few for reliable calibration.

Any adopted change should be evaluated one at a time and appended to this log
with its before/after metrics and any regressions.

## Experiment 1 — Reduce chunk overlap

### Decision

Reduce `CHUNK_OVERLAP` from 240 to 120 characters. Keep chunk size 1,100,
top-k 4, `all-MiniLM-L6-v2`, cosine similarity, and the `0.50` answerability
threshold unchanged.

This was selected because it delivered the strongest non-regressive aggregate
improvement among the four requested retrieval variables. It fixed q-11 and
q-13 while preserving the 100% top-k source hit rate and 94.7% first-source
match rate.

### Isolated comparison

Each row changes only the named variable from the baseline. These measurements
use the same 25 cases and the same `0.50` answerability threshold.

| Variant | Retrieval hit | Answerability accuracy | Unsupported-answer rate | Source match |
| --- | ---: | ---: | ---: | ---: |
| Baseline: size 1100, overlap 240, top-k 4, all-MiniLM | 100.0% | 80.0% | 66.7% | 94.7% |
| Chunk size 700 | 100.0% | 80.0% | 66.7% | 94.7% |
| Chunk size 900 | 100.0% | 84.0% | 50.0% | 94.7% |
| Chunk size 1400 | 94.7% | 88.0% | 33.3% | 94.7% |
| Overlap 0 | 100.0% | 80.0% | 66.7% | 94.7% |
| **Overlap 120 — selected** | **100.0%** | **88.0%** | **50.0%** | **94.7%** |
| Overlap 400 | 94.7% | 84.0% | 50.0% | 94.7% |
| Top-k 2 | 94.7% | 80.0% | 66.7% | 94.7% |
| Top-k 3 | 100.0% | 80.0% | 66.7% | 94.7% |
| Top-k 6 | 100.0% | 80.0% | 66.7% | 94.7% |
| Top-k 8 | 100.0% | 80.0% | 66.7% | 94.7% |
| Embedding: multi-qa-MiniLM-L6-cos-v1 | 100.0% | 80.0% | 50.0% | 94.7% |

### Why the other variables were not selected

- **Chunk size:** 1,400 reduced unsupported-answer risk further, but it lost one
  expected source from top-k. The 900-character option improved less than the
  selected overlap change.
- **Chunk overlap:** 120 was the only tested overlap that improved answerability
  while preserving both source metrics. Zero overlap did nothing; 400 lost a
  retrieval hit.
- **Top-k:** Values of 3, 6, and 8 produced the same aggregate metrics as the
  baseline. Top-k 2 lost q-22. This is expected because answerability uses the
  first result's similarity, which top-k does not change.
- **Embedding model:** `multi-qa-MiniLM-L6-cos-v1` reduced one false positive but
  introduced an answerable false negative on q-22, leaving overall accuracy at
  80%. It was already cached locally, so no paid API or download was used.

### Before and after

| Metric | Before: overlap 240 | After: overlap 120 | Change |
| --- | ---: | ---: | ---: |
| Retrieval hit rate | 19/19 (100.0%) | 19/19 (100.0%) | No change |
| Answerability accuracy | 20/25 (80.0%) | 22/25 (88.0%) | +8 percentage points |
| Unsupported-answer rate | 4/6 (66.7%) | 3/6 (50.0%) | -16.7 percentage points |
| Source-match rate | 18/19 (94.7%) | 18/19 (94.7%) | No change |

Resolved in this run:

- q-11 moved from `0.5234` (incorrectly answerable) to approximately `0.439`
  (correctly unanswerable).
- q-13 moved from `0.4661` (incorrectly unanswerable) to approximately `0.504`
  (correctly answerable), with the Quality Review Guidelines still ranked first.

Remaining failures:

- q-16: missing bonus detail still classified answerable (`~0.538`).
- q-20: missing fastest-bank comparison still classified answerable (`~0.587`).
- q-25: missing insurance detail still classified answerable (`~0.506`).

Observed tradeoff:

- q-22's expected training source moved from rank 3 to rank 4. It remains a
  top-k retrieval hit, so aggregate hit and source-match metrics did not change,
  but it is closer to being lost if top-k is reduced. This regression should
  remain visible rather than being hidden by the aggregate improvement.

### Interpretation

The smaller overlap appears to reduce distracting repeated context and produces
more focused chunk embeddings for q-11 and q-13. This is a plausible explanation,
not proof of causality from 25 cases. The remaining failures are largely
missing-detail or partial-support problems, so further overlap tuning is unlikely
to solve them. A support-verification layer is a more appropriate next research
direction than continuing to tune retrieval on this small dataset.
