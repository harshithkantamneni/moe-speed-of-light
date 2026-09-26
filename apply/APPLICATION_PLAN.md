# ETH Zurich application plan

## Deadlines and routes (checked 25 Sept 2026)

| Route | How to apply | Deadline |
|---|---|---|
| ETH AI Center Doctoral Fellowship | Online application with two co-supervising PIs from *different fields* (e.g. Hoefler for systems plus an ML/NLP PI). Topic areas include "AI system design" and "edge computing". | **Tue 27 Oct 2026, 16:00 CET** (09:00 in Milwaukee) |
| SPCL (Hoefler) | Email **spcl-hiring@spcl.inf.ethz.ch**. The subject line **must start with `[JOB@SPCL] `**. Attach: full CV; all BSc/MSc transcripts (originals *and* translations); letter of intent (≤500 words, draft in `SPCL_letter_of_intent.md`); a short description of your most important achievement with a link (the paper and repo); publications and code samples; three referees, ideally academics. Incomplete applications are discarded. | Rolling |
| SAFARI (Mutlu) | Online "apply with us" form, reviewed monthly. Frame the project as consumer-scale capacity tiering (links to their FLINT/HBF and PAPI work). | Rolling |
| D-INFK central doctoral pool | Central application, visible to all faculty for 4 months. | Rolling |

## Before applying

1. Push the repo, then post the paper to arXiv (cs.DC, cross-list cs.LG or cs.PF). As an Independent Researcher you may need an endorser for cs.DC. Ask Prof. Sinclair or Prof. Venkataraman, which is also a natural way to start the conversation you already planned with them.
2. **Read the paper line by line until you can derive Eq. 1, the bound and r\* on a whiteboard without notes.** An interview will test this. Rerun `tests/` and `scripts/` yourself.
3. Referees: Prof. Matt Sinclair (ECE 752) and Prof. Shivaram Venkataraman are the obvious academic names. Ask them now: 4 weeks' notice is the minimum.
4. Decide on the AI-disclosure line in the paper's acknowledgments (see the note in the final message).

## Short "most important achievement" text (for the SPCL email)

> Independent paper plus open artifact: *Where Do the Experts Go? A Validated Speed-of-Light Model for MoE Decode on Memory-Constrained Consumer Hardware*. It shows that exact MoE routing traces can be collected without a GPU. It gives a 6-parameter bytes-over-bandwidth decode model that predicts 52 published measurements from 9 independent sources with 16% median held-out error, and a Belady/Jensen speed-of-light bound for any expert-placement policy. Code, traces, measurement set: github.com/harshithkantamneni/moe-speed-of-light
