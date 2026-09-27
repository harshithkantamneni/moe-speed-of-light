# ETH Zurich application plan

## Deadlines and routes (checked 25 Sept 2026)

| Route | How to apply | Deadline |
|---|---|---|
| ETH AI Center Doctoral Fellowship | Online application with two co-supervising PIs from *different fields* (e.g. Hoefler for systems plus an ML/NLP PI). Topic areas include "AI system design" and "edge computing". | **Tue 27 Oct 2026, 16:00 CET** (09:00 in Milwaukee) |
| SPCL (Hoefler) | Email **spcl-hiring@spcl.inf.ethz.ch**. The subject line **must start with `[JOB@SPCL] `**. Attach: full CV; all BSc/MSc transcripts (originals *and* translations); letter of intent (≤500 words, draft in `SPCL_letter_of_intent.md`); a short description of your most important achievement with a link (the paper and repo); publications and code samples; three referees, ideally academics. Incomplete applications are discarded. | Rolling |
| SAFARI (Mutlu) | Online "apply with us" form, reviewed monthly. Frame the project as consumer-scale capacity tiering (links to their FLINT/HBF and PAPI work). | Rolling |
| D-INFK central doctoral pool | Central application, visible to all faculty for 4 months. | Rolling |

## Before applying (in order)

1. **Read the paper line by line** (`paper/paper.pdf`) until you can derive Eq. 1, Proposition 1 and the audit's band
   without notes; rerun `tests/` and `scripts/paper_numbers4.py` yourself. About 6 hours.
2. **arXiv.** Post to cs.DC (cross-list cs.LG, cs.PF). As an Independent Researcher you likely need a cs.DC endorser:
   ask Prof. Matt Sinclair or Prof. Shivaram Venkataraman, which also opens the referee conversation. Post before
   27 Oct so the fellowship form can link it.
3. **Referees** (letters due 2 Nov for the fellowship): ask now; four weeks' notice is the minimum. Send each the PDF,
   the repo link and a 5-line summary.
4. **Audited teams.** The protocol commits to sending every audited team its rows before a second version. Draft one
   short, neutral email per system (rows + how to reproduce with `mosl/calc.py`). About 3 hours for 41 systems.
5. **AI disclosure.** The paper's acknowledgments state that parts of the code, experiments and text were drafted with
   an AI system. Keep, edit or remove it according to the venue's policy; do not leave it inconsistent with how you
   describe the work in interviews.
6. **MLSys 2027** (30 Oct): the paper is two-column, 9 pages including appendix; check the MLSys template and page
   limit, and move the appendix tables to supplementary material if needed.

## Short "most important achievement" text (for the SPCL email)

> *Seconds, Not Blocks: A Validated Speed-of-Light for Offloaded Mixture-of-Experts Decode, and What It Says About
> Published Speed-Ups* (independent paper, open artifact). A lower bound on MoE decode time for any expert-placement
> policy, computed from exact routing traces; a decode model validated on third-party data (16% median error,
> cross-validated over 9 sources) and tested on pre-registered first-party and anchor runs; and an audit of 147
> published measurements from 41 systems: 16 of 20 adjudicable llama.cpp baselines are weak, 8 of 22 claimed gains
> survive, and the median system reaches 24% of the physical speed-of-light. github.com/harshithkantamneni/moe-speed-of-light

## Documents in this folder

- `SPCL_letter_of_intent.md`: the ≤500-word letter for spcl-hiring@spcl.inf.ethz.ch.
- `research_proposal.md`: the fellowship research proposal (aims: bounds beyond one request; offload-aware MoE
  design; scientific benchmarking for sparse inference).
