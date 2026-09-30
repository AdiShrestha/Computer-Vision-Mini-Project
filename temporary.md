I am an ai agent and i have worked on project ................. Sentinel-GL, a scientific software and publication-rehabilitation project for self-supervised, multi-sensor Earth-observation anomaly scoring around glacial lakes, with a retrospective focus on the South Lhonak glacial lake outburst flood and a planned submission path toward IEEE Transactions on Geoscience and Remote Sensing (TGRS). I worked on the project as the assigned Implementor under Software Factory v2.2.0, specifically on Chunk 01, “Legacy Forensics and Control-Plane Rehabilitation.” During that work I also temporarily executed five Architect-owned High-risk contracts because the Human Operator explicitly stated that the Architect was unavailable and directly authorized that exception for this chunk. This document is my detailed account of the project, my experience using Factory v2.2 in a real and unusually difficult scientific-rehabilitation setting, and my recommendations for a backward-compatible v2.3 or, if the structural changes are accepted, a more substantial v3.

# Factory v2.2 Experience Report — Sentinel-GL

## 1. Executive assessment

Factory v2.2 was genuinely useful. It did not merely organize files or provide a checklist. It materially changed how I approached a repository containing scientifically unsafe legacy behavior, untrusted numerical artifacts, mixed public and private concerns, and a manuscript whose claims could not be accepted simply because code and results already existed. The Factory repeatedly forced the correct question: “What evidence is actually admissible, and what has merely been inherited?” That distinction was the central problem in Sentinel-GL.

The strongest parts of v2.2 were its fail-closed posture, explicit contract scope, separation of implementation from scientific claims, preservation of negative results, nested control-plane repository, contract-level self-review, and chunk-level handoff. Those mechanisms helped prevent an AI agent from converting uncertainty into confident but unsupported progress.

The main weaknesses were not philosophical. They were mechanical. Too many important guarantees depended on exact Markdown wording, duplicated prose, manually synchronized state, or assumptions that were not represented as machine-readable state. In several places, the written protocol was stronger than the executable tooling. The most concrete example was verification-command parsing: earlier reports contained commands and literal outputs, but the aggregate verifier extracted zero commands because their formatting did not match the parser’s precise expected syntax. A second example was Frozen File snapshotting: the Factory expected snapshots, but the snapshot artifacts themselves were outside the contracts’ Allowed Files, creating a conflict between two valid controls. A third example was C01-03, where a wildcard audit correctly scanned more acquisition files than the contract had allowed the Implementor to modify, so the verification boundary and the scope boundary disagreed.

My recommendation is:

- release a v2.3 if the goal is to preserve the current document-and-command model while removing fragile syntax, automating state capture, and closing the conflicts observed on Sentinel-GL;
- call it v3 if the Factory is changed from a prose-driven protocol with helper commands into a machine-readable, event-sourced execution system that generates the prose artifacts automatically.

The amount of change alone should not determine the version number. The architectural boundary should. If Markdown remains authoritative and Gatekeeper remains a collection of commands that interpret it, v2.3 is appropriate. If a structured contract and execution journal become authoritative and Markdown becomes a generated human-readable view, that is a genuine v3.

Every recommendation in this report follows one strict constraint from the Human Operator: no upgrade may add a new responsibility to the Human Operator. Improvements should reduce Human effort, preserve it, or transfer mechanical work from the Human to the Factory and its agents. The Factory may request an existing genuinely human-only decision when one is scientifically unavoidable, but it should not create new forms, confirmations, copying steps, review rituals, or bookkeeping duties merely because the software became more sophisticated.

## 2. The Sentinel-GL project I worked on

### 2.1 Scientific purpose

Sentinel-GL is intended to investigate whether a self-supervised temporal representation learned from multi-sensor Earth-observation data can support defensible anomaly scoring for glacial lakes. The research design separates several questions that are easy to conflate:

1. whether a learned representation responds to controlled perturbations on authentic background observations;
2. whether the resulting anomaly scores outperform simpler operational, statistical, classical machine-learning, and recent learned baselines under identical evaluation instances and information budgets;
3. whether a threshold selected without knowledge of the final event produces a useful retrospective signal before the South Lhonak event;
4. whether apparent signals survive sensor ablation, cloud and missingness analysis, calibration constraints, and negative-control testing;
5. whether any result is broad enough to justify language such as “robust,” “precursor detection,” “benchmark,” or “operational.”

The project is therefore not simply a model-training repository. It is an evidence-production system. External source acquisition, provenance, preprocessing, feature semantics, split boundaries, fitted transforms, model lineage, evaluation instances, prediction rows, statistical units, metrics, claims, manuscript text, and release artifacts form one directed chain. The founding architecture expresses this as evidence flowing upward through trust zones: external sources, authenticated raw observations, derived data, fitted model state, evaluation evidence, aggregated evidence, and finally release claims.

That upward-only evidence model was important in my work. It prevented later-stage assets—plots, manuscript numbers, old result JSON, or an apparently good AUC—from being used to retroactively validate an earlier data or provenance layer. If the acquisition boundary was invalid, every downstream artifact inherited that taint regardless of how plausible it looked.

### 2.2 Condition of the inherited repository

Sentinel-GL already contained substantial work: source code, acquisition scripts, preprocessing paths, tests, a TS-MAE implementation, model artifacts, evaluation scripts, figures, result files, claim-evidence maps, and manuscript material. It would have been easy for an implementation agent to assume that this represented a mostly complete project needing cleanup and polish.

The founding Methodology Adversarial Review showed that this interpretation was unsafe. It found acquisition fallback and generation pathways, modification of raw-like data using synthetic physical variations, insufficient provenance, baseline comparability defects, fabricated or reconstructed statistical inputs, pseudo-replication risk, incorrect or weak event-timing semantics, small or degenerate strata, venue-alignment deficiencies, and manuscript claims whose scope exceeded the available independent evidence. The acquisition audit at founding recorded ten hard failures and four warnings. The MAR therefore failed data authenticity, baseline sufficiency, statistical power, venue alignment, and assumption stress testing, while title-claim consistency was only conditionally acceptable.

The central rehabilitation rule was that existing numerical evidence was not publication evidence merely because it existed or because an old test passed. Legacy artifacts had to be classified as source to preserve, items to reverify, quarantined evidence, or material to delete from the eventual public release. Invalid numerical paths were not to be rerun to fill missing outputs. Publication claims had to be rebuilt only after provenance, data, preprocessing, model, evaluation, and statistics were rebuilt in dependency order.

This was exactly the kind of project where an autonomous AI agent can be dangerous if the process rewards visible completion over epistemic correctness. Factory v2.2’s insistence on honest unknowns, stop conditions, and explicit evidence classes was therefore especially valuable.

### 2.3 Intended research and release architecture

The target public repository is designed around a conservative, auditable structure: versioned configurations, source modules, tests, documentation, manifests, immutable release results, and manuscript sources. Large or sensitive artifacts are to be kept outside ordinary git history and referenced through checksums, versioned releases, or archival identifiers. The private Factory control plane lives in a separate nested `project/.git` repository and is ignored by the outer public repository.

The intended pipeline includes:

- fail-closed external-source adapters;
- source/product/version and observation-level provenance;
- a Reality Gate comparing measured data properties with predeclared expectations;
- one canonical feature schema with explicit order, units, physical ranges, missingness, and transformations;
- training-only imputation and normalization;
- a deterministic or fully identified TS-MAE training path;
- complete checkpoint lineage;
- three distinct score definitions, including a Score-C transform that must not fit on the final test trajectory;
- three distinct evaluation tasks: controlled synthetic perturbations, the real South Lhonak retrospective, and negative controls;
- common instance identifiers and equal information budgets across methods;
- calibration isolated from final evaluation;
- an observation-level prediction ledger written before aggregation;
- statistical procedures using defensible independent units rather than overlapping windows as pseudo-replicates;
- explicit `NOT_ESTIMABLE` behavior for empty or single-class strata;
- real ablations that rerun the affected path rather than editing result summaries;
- claim-evidence generation and manuscript metrics computed from admissible artifacts;
- a clean-clone release verification and final Factory release certification.

The project explicitly supports a negative-result path. Sentinel-GL does not define success as “the proposed model wins.” A valid outcome may be a useful method result, a rigorous negative result showing no advantage over simpler methods, or a data-limited feasibility report that narrows the claim. This principle aligned very well with Factory v2.2’s rule against result-directed implementation.

### 2.4 The twelve-chunk rehabilitation plan

The rehabilitation plan is dependency-driven rather than calendar-driven. Its intended progression is:

1. legacy forensics, evidence quarantine, and Factory control-plane rehabilitation;
2. fail-closed acquisition and provenance reconstruction;
3. clean reacquisition and Reality Gate validation;
4. canonical preprocessing and feature reconstruction;
5. clean TS-MAE training and representation validation;
6. unified prediction ledger and fair baseline evaluation;
7. statistical redesign using defensible independent units;
8. actual ablations, sensitivity analysis, and cloud/missingness robustness;
9. claim-evidence registry, figures, tables, and reproducibility package;
10. manuscript reconstruction for TGRS;
11. independent adversarial review and a bounded fix cycle;
12. clean-clone reproduction, release certification, and submission freeze.

Chunk 01 had to come first because later work would otherwise build on a scientifically untrusted substrate. It was not intended to produce new performance numbers. It was intended to establish what could be trusted, prevent known invalid paths from silently executing, create formal contracts for later evidence, and preserve the legacy repository for forensic comparison.

## 3. The exact Chunk 01 work I performed

Chunk 01 contained ten contracts. All ten reached `COMPLETE`, with twenty-three recorded self-review attempts in total. Five were Medium-risk Implementor contracts and five were High-risk Architect contracts. The High-risk contracts were executed only because the Human Operator explicitly overrode the normal role boundary for this chunk while the Architect was unavailable. The reports and telemetry retained their original `Architect` owner metadata and High-risk classification; the temporary execution exception did not rewrite history to make those contracts appear ordinary.

### 3.1 C01-01 — forensic inventory and legacy classification

I established a verifiable forensic inventory of the inherited repository and produced a machine-readable classification of legacy artifacts. The inventory covered 757 paths and recorded enough information to reason about tracked state, publication relevance, and evidence disposition. The work also produced a legacy evidence taint report and preserved an archive/checksum trail rather than allowing cleanup to erase the state that motivated rehabilitation.

This contract demonstrated a strong Factory pattern: preservation before correction. It is much easier to audit a rehabilitation when the original state can be reconstructed. However, the archive’s temporary storage location also exposed a process weakness discussed later: a Factory-generated forensic artifact should not depend on an ephemeral system location and then require the Human to remember to rescue it.

### 3.2 C01-02 — Factory control-plane verification

I verified Factory v2.2’s bootstrap state, nested repository isolation, ignore behavior, and self-check. The nested `project/.git` repository worked as intended: it gave private control-plane artifacts a real history without placing them into the outer public repository. The Factory metadata recorded version 2.2.0 and its source commit.

This was one of the cleanest parts of the experience. The separation made it possible to commit contracts, reports, telemetry, decisions, matrices, and the chunk report independently of the public scientific code. It also revealed that Factory needs a stronger abstraction for combined outer/nested repository status, because a clean control-plane repository does not imply that the implementation tree is clean or committed.

### 3.3 C01-03 — fail-closed acquisition boundary

I replaced unsafe acquisition behavior with explicit fail-closed adapters and compatibility shims. The adapters reject invalid configuration and unavailable providers rather than generating plausible substitute observations. Compatibility paths were maintained where necessary, but they delegate to the same fail-closed semantics rather than preserving legacy fallback behavior. The exact acquisition audit finished with zero hard failures and zero warnings. Ten focused fail-closed tests and four compatibility tests passed.

This was the most instructive contract mechanically. Its verification command used a wildcard over `source/data/acquisition/*.py`, but the original Allowed Files list did not include every legacy module matched by that wildcard. The audit therefore enforced a broader semantic boundary than the contract’s modification boundary. The Factory did the right thing by making scope important, and the audit did the right thing by scanning the whole acquisition surface, but the two controls had not been reconciled during contract drafting. I recorded the correction in decision `DEC-019` rather than silently broadening scope.

The general lesson is that every verification command has an implied read set and sometimes an implied repair set. A future Factory should compute or declare these sets before execution. If a wildcard, recursive test discovery, linter, generator, or formatter spans files outside Allowed Files, the contract should fail preflight before implementation starts—not after the agent discovers the mismatch during the highest-risk work.

### 3.4 C01-04 — provenance and data-manifest schemas

I implemented strict, machine-readable schema validation for acquisition provenance and dataset manifests. The schemas reject missing identity, ambiguous source information, malformed checksums, incomplete temporal/spatial descriptions, undocumented processing steps, and invalid channel metadata. Sixteen focused tests passed.

This work converted important scientific prose into executable boundaries. It showed why schema-first contracts are valuable: a later agent cannot simply write a convincing narrative saying that provenance exists. The actual object must contain required fields and satisfy consistency rules.

### 3.5 C01-05 — Feature Schema v2

I established the canonical Feature Schema v2 contract and tests. It defines feature identity, ordering, units, physical ranges, missingness observability, transformations, source linkage, and validation semantics. Eight focused tests passed.

This was High-risk because an incorrect feature contract contaminates preprocessing, model input, ablations, interpretation, and reproducibility. The most valuable Factory behavior here was forcing the schema to be treated as an evidence contract rather than a convenient Python data structure.

### 3.6 C01-06 — task protocols, calibration, and information budgets

I formalized the evaluation protocols for the three distinct tasks, including calibration isolation, threshold policies, instance identity, method information budgets, and Score-C constraints. Eight focused tests passed.

The Factory’s scientific claim tiers were helpful here. This contract was descriptive rather than a comparative result, but it touched the rules that later comparative claims will depend on. Treating protocol code as scientifically consequential prevented the common mistake of postponing fairness decisions until after seeing results.

### 3.7 C01-07 — prediction ledger

I implemented the row-level Prediction Ledger contract and statistical-input validation. The ledger preserves individual evaluation rows before aggregation, binds methods to common instances, and records the fields needed to recompute metrics and evaluate missingness, event timing, thresholds, and provenance. Six focused tests passed.

This was one of the best architectural decisions in the founding artifacts. A result table is too late in the evidence chain to diagnose many scientific defects. A ledger makes it possible to identify whether methods saw the same cases, whether an event date was handled correctly, whether a stratum was empty, and whether a statistical routine received real observations rather than generated stand-ins.

### 3.8 C01-08 — evidence quarantine

I implemented and validated a machine-readable Evidence Quarantine Registry with eleven entries. The runtime enforcement prevents quarantined legacy artifacts from entering trusted evidence paths. Six focused tests passed.

This contract gave operational meaning to the MAR. Without enforcement, words such as `LEGACY_TAINTED` are only warnings. With a registry and code boundary, the project can preserve historical files for forensic purposes while preventing them from being promoted into release evidence.

### 3.9 C01-09 — release hygiene and public-history strategy

I reconstructed the repository ignore boundary, wrote the release hygiene policy, and documented a public-history strategy. The policy separates public research artifacts from private Factory controls, local paths, credentials, raw/large data, caches, checkpoints, work results, and internal workflow language. It preserves release-bound outputs through explicit allowlisting. It also records a non-destructive strategy: retain the private legacy bundle, then create a clean public release history at the release stage rather than rewriting the forensic source prematurely.

This contract highlighted the usefulness of explicit public/private separation. It also highlighted the need for the Factory to understand dual-repository delivery. Files under `project/` were committed automatically to the nested repository, while outer source and documentation changes remained visible in the outer working tree and had to be reported accurately.

### 3.10 C01-10 — venue literature and MAR coverage

I built a traceable literature matrix covering recent TGRS work, relevant remote-sensing foundation and temporal representation approaches, and regional GLOF literature. I verified publication identifiers and recorded modalities, geography, sample/protocol details where available, baselines, metrics, cloud or missingness handling, limitations, and implications for Sentinel-GL. I also mapped MAR-1 through MAR-7 to future rehabilitation chunks and made clear that no MAR gate had been resolved merely by creating the matrix.

This contract reinforced a valuable Factory discipline: literature presence is not scientific validation. A DOI, baseline name, or venue citation can support design decisions, but it does not close provenance, power, comparability, robustness, or claim-scope gates.

### 3.11 Chunk closure

I compiled the aggregate Chunk 01 report, committed it in the nested project repository, and staged eleven self-contained reports into `TAKE_THIS/`: ten contract reports plus the chunk report. Gatekeeper reported every contract complete and `NEXT: none`.

The aggregate report explicitly preserved limitations rather than smoothing them away:

- the five High-risk contracts used a temporary Human-authorized role override;
- no clean data, performance results, or MAR closure had been produced;
- Frozen File snapshots were unavailable under the contracts’ path permissions;
- the outer repository remained dirty with pre-existing and Chunk 01 changes;
- C01-03 required a scope correction;
- the aggregate verification parser only recognized seven command declarations from C01-09 and C01-10 because earlier reports used a different format.

That honest closure is a success of the Factory’s reporting philosophy even though some of those limitations expose weaknesses in the Factory implementation.

## 4. What Factory v2.2 did particularly well

### 4.1 It made scientific honesty a workflow property

The most important success was that the Factory did not treat scientific validity as a final review checkbox. Authenticity, leakage, comparable instances, calibration, statistical units, claim scope, negative results, and release traceability were present in the founding artifacts, contracts, invariants, tests, reports, and chunk review package.

The rule “never fabricate anything” was not decorative. It was reinforced by acquisition auditing, provenance schemas, quarantine, prediction-ledger requirements, evidence cross-check concepts, recomputation mechanisms, and report requirements. This redundancy is appropriate because scientific fabrication can enter through many paths: generated acquisition values, reconstructed metric inputs, hand-authored manuscript numbers, swallowed exceptions, or a report that claims a command passed without actually running it.

### 4.2 Allowed Files and Frozen Files constrained scope effectively

Explicit scope prevented opportunistic refactoring and protected founding facts and invariants from being edited to fit implementation convenience. On a dirty rehabilitation repository, that discipline is essential. Without it, an agent could “fix” a failing invariant by changing the invariant or edit historical context until a new result appeared consistent.

The weakness was not the control itself; it was the lack of automated preflight reconciliation. The concept should remain in both v2.3 and v3.

### 4.3 Stop conditions were treated as legitimate outcomes

Factory v2.2 correctly describes a stop as the expected response to missing authority, invalid data, impossible verification, or contradictory architecture. This is especially useful for AI agents, which otherwise tend to optimize for continuing. The five-attempt self-review ceiling also prevents endless local patching from becoming a substitute for recognizing a blocked contract.

### 4.4 Negative results were first-class

Sentinel-GL’s roadmap allows the method to lose to a baseline or fail to produce an operationally useful event signal without treating the entire project as a failure. That is excellent scientific process design. It reduces pressure on the agent to tune toward a desired conclusion and makes it possible to publish a rigorous evaluation or feasibility result.

### 4.5 The nested repository was a strong privacy and history boundary

The separate `project/.git` repository worked well for private control-plane history. Contract reports, decisions, telemetry, matrices, and chunk reports received a real commit trail while remaining excluded from the intended public repository. `commit-project` was simple and reliable.

### 4.6 `next` and the execution manifest reduced ambiguity

The dependency-aware `next` command was a useful state query. It prevented me from relying on memory to decide which contract was ready and provided a clear chunk completion condition. This is a good example of what should become even more central in a future structured state machine.

### 4.7 Materialization reduced transcription risk

For Architect-authored source embedded in contracts, deterministic materialization is preferable to manual copying. It preserves byte identity and makes the division between design and implementation explicit. This is particularly useful for High-risk contracts where a subtle transcription change could alter scientific semantics.

### 4.8 Contract reports were self-contained and useful

The requirement to quote objectives, enumerate files, record literal commands and outputs, audit every Definition of Done item, describe self-review attempts, and state remaining risks produced a stronger record than a conventional commit message or brief completion note. The aggregate chunk report then made it possible to hand over a coherent review package.

### 4.9 Release terminology was disciplined

Factory v2.2 distinguished a contract pass, chunk completion, and project release certification. This prevents agents from describing a locally passing task as a certified scientific release. That terminology discipline should remain non-negotiable.

## 5. Friction and failure modes observed in real use

### 5.1 Markdown formatting acted as an undocumented API boundary

The aggregate `verify-contract` behavior was the clearest defect. C01-01 through C01-08 contained verification commands and outputs, but the parser found zero declared commands because those reports used literal `Command:` blocks rather than the parser-recognized bullet form `- **Verification Command**: ...`. C01-09 and C01-10 used the recognized form, so seven commands were independently rerun and passed.

This means a visually understandable report can be mechanically invisible. The danger is not only missed verification. It is false confidence: a command can report “nothing declared” and still exit successfully unless the caller understands what zero extracted commands means in context.

Formatting should never be the sole authority for a safety-critical verification set. At minimum, v2.3 should support backward-compatible parsing and fail when a contract declares verification scripts but the report parser extracts zero. In v3, the authoritative commands should live in structured contract state and the Markdown report should be generated from execution records.

### 5.2 Frozen File snapshots conflicted with Allowed Files

Contracts listed Frozen Files, and Gatekeeper expected snapshot records under its own snapshot location. However, those snapshot paths were not included in the contract Allowed Files. A strict Implementor could not both obey “write only Allowed Files” and create the expected snapshot artifact.

This is a policy-layer conflict. The agent should not solve it by casually expanding its own permissions. Factory-owned metadata should have a reserved write namespace that does not count as project implementation scope. The Factory itself—not the Implementor—should write snapshots there and record them as system artifacts.

### 5.3 Verification reach and modification scope were not reconciled

C01-03’s wildcard audit scanned all acquisition modules, while the original contract did not allow modification of every matched module. Similar problems can occur with recursive linters, test discovery, generated schemas, formatting tools, import graphs, or release scans.

The Factory should calculate a command’s static path reach where possible and support explicit declarations where static analysis is impossible. A contract preflight should detect:

- wildcard paths that include files outside scope;
- tests importing or generating files outside scope;
- report, telemetry, decision-log, snapshot, or stamp artifacts missing from the permitted system-write namespace;
- outputs whose parent directories do not exist or are ignored unexpectedly;
- Frozen Files also listed as outputs or implied repair targets.

This check should run automatically when the chunk arrives. The Human should not be asked to perform or interpret it manually.

### 5.4 The dual-repository model lacked a unified delivery status

The nested `project` repository could be clean and fully committed while the outer repository contained modified and untracked implementation files. This was not inherently wrong; Chunk 01 intentionally changed outer source and documentation. The problem was that “repository clean” could refer to two different repositories, and `commit-project` only addressed one of them.

Future versions should expose a single status object with at least:

- control-plane repository state;
- implementation repository state;
- pre-existing dirty baseline;
- files changed by the current contract;
- files changed by earlier contracts in the chunk;
- files unrelated to the Factory session;
- expected untracked outputs;
- commit identifiers for each relevant repository.

The Factory should not automatically commit the user’s outer repository unless the existing project policy authorizes it. It can, however, calculate and report the distinction automatically. That improves safety without adding a Human step.

### 5.5 Dirty-tree handling was too coarse

`check --allow-dirty` was necessary because Sentinel-GL began as a dirty rehabilitation repository. But a global dirty-tree bypass weakens repository-integrity evidence. The important question is not simply “is anything dirty?” It is “did this contract touch only its allowed delta relative to the captured starting state?”

V2.3 should replace or supplement `--allow-dirty` with scoped dirtiness:

- capture the initial status and hashes at contract start;
- distinguish pre-existing changes from contract changes;
- fail if the contract modifies an unrelated pre-existing file;
- permit expected Allowed File changes;
- record the exact delta in the report and stamp.

This should be automatic and should not require the Human to clean, stash, or classify the working tree before every chunk.

### 5.6 Execution evidence was duplicated manually

The Implementor specification requires literal command and output blocks in checkpoint logs, fresh reruns in self-review, report verification summaries, telemetry, and sometimes stamps. The underlying principle is sound, but manual duplication increases context usage and creates synchronization risk. A command may be run correctly yet copied into the report with a typo, truncation, or incompatible heading.

The Factory should own an append-only execution journal. Every command should receive:

- contract and checkpoint identity;
- timestamp;
- working directory;
- normalized command or argument vector;
- environment allowlist or environment fingerprint without secrets;
- exit code;
- stdout/stderr artifact paths and hashes;
- duration;
- files changed before and after;
- whether the run was implementation-time or independent self-review;
- the agent/model identity.

Reports should render these records rather than asking the agent to manually reproduce them. The full raw output can live in evidence files while the report contains a bounded excerpt and a hash/link. This preserves evidence while reducing prompt and report bloat.

### 5.7 Telemetry expectations were difficult to apply consistently

The specification says to append telemetry after every phase transition, while the minimum completion schema emphasizes a `phase_4` contract-complete event. This creates ambiguity about whether one final line is minimally compliant or whether four or more events are required. Manual JSONL appends are also vulnerable to schema drift, duplicate events, invalid JSON, and inconsistent model identifiers.

Gatekeeper should emit telemetry automatically when phase commands or state transitions occur. The agent should be able to add qualitative notes, but required fields and event identity should be system-generated. A telemetry validator should run before contract completion.

### 5.8 The High-risk outage override had no native protocol

V2.2 correctly says High-risk contracts belong to the Architect and the Implementor must stop. Sentinel-GL then encountered a real operational exception: the Architect was unavailable, and the Human explicitly instructed the Implementor to perform those contracts for this chunk.

The correct response was to preserve risk and owner metadata, record the Human authorization, conduct stronger self-review, and flag independent review. However, the Factory had no native representation for temporary role delegation. I had to encode the exception in reports, telemetry, and a decision entry.

A future Factory should support a bounded delegation record generated automatically from an explicit Human instruction already given in ordinary language. It should contain:

- original owner;
- temporary executor;
- affected contract IDs or chunk boundary;
- authorization source and timestamp;
- expiration condition;
- required later review state;
- prohibition on using the exception as precedent for unrelated chunks.

This must not require the Human to fill out a form, duplicate their instruction, or perform an extra acknowledgment. The Human’s existing explicit order is enough; the agent and Factory should create the record.

### 5.9 `TAKE_THIS` lifecycle assumed a Human retrieval event

The specification acknowledges that `clear-takethis` assumes the Human already retrieved the prior bundle. Because the files are copies, accidental clearing does not destroy the permanent reports, but it can remove the convenient handoff package.

The improved behavior should be automatic rotation rather than deletion. For example, starting a new chunk could move the old staged package to `TAKE_THIS_ARCHIVE/chunkNN/<bundle-hash>/` or regenerate it on demand. This removes the assumption without adding a Human confirmation.

### 5.10 Temporary forensic archive placement was fragile

The C01-01 forensic archive was created in a temporary location and then documented as something the Human should preserve. That creates a new Human memory burden and conflicts with the requirement for no additional Human responsibilities.

Factory-generated forensic bundles should default to a durable workspace-controlled archive location, with an optional configured external store. They should include checksums and a manifest and should be excluded from public release automatically. If storage constraints require temporary placement, the Factory should copy or package the artifact into the durable handoff automatically before declaring the contract complete.

### 5.11 Known-limitations evidence counts can become stale prose

The Implementor specification contains useful statements such as “evidenced by exactly one project” or “zero real-project occurrences.” Sentinel-GL itself changes some of those counts. If the documentation is not updated automatically, a real successful use can occur while the Factory continues to describe the feature as unexercised.

Feature maturity should be recorded in a machine-readable capability registry populated from anonymized or local project telemetry. Documentation can render the current evidence state. A Human should not have to remember to edit version prose after each project.

### 5.12 The protocol imposed a large attention tax

The comprehension gate, ten planning steps, checkpoint reports, multiple audits, self-review, report construction, telemetry, decision logging, nested commits, and chunk packaging are individually defensible. Together, when manually narrated, they consume substantial agent context and increase the probability of a procedural omission.

The solution is not to remove rigor. The solution is to move deterministic rigor into code. The agent should spend its reasoning budget on scientific semantics, implementation design, failure analysis, and adversarial review—not repeatedly transcribing file lists and command outputs that the Factory can derive.

### 5.13 Contract and report schemas were distributed across documents

Authoritative behavior was spread among `implementor_spec.md`, `factory_spec.md`, Gatekeeper help, the manifest, individual contracts, dynamic rules, and report templates. `self-check` helps detect drift, but the possibility of drift exists because the same concept is expressed in multiple places.

V2.3 should introduce a single machine-readable schema for contracts, reports, statuses, evidence declarations, and telemetry. Markdown specifications can explain the semantics, but field names and allowed values should come from generated schema definitions.

### 5.14 Tier inference is valuable but should be explainable and testable

Fail-closed scientific tier inference is a strong idea. However, keyword heuristics can produce false positives or miss claims expressed indirectly. Every inferred tier should include an explanation trace: matched phrase, rule identifier, inferred minimum tier, and what additional gate it activates. Contracts should include tests for named statistical and mathematical operators where applicable.

The Factory should preserve the conservative default while allowing the agent to attach a reasoned “heuristic not applicable” note that remains visible at review. It should not allow the agent to silently downgrade the tier.

### 5.15 Source verification and citation evidence need a structured form

C01-10 required external literature verification. I recorded DOI and source information in a Markdown matrix. That is useful, but a future Factory could represent source claims as structured evidence records containing identifier, resolved URL, title, authors, venue, publication year, retrieval date, fields extracted, and the artifact or paragraph that uses them.

This would make literature and factual-source drift auditable without asking the Human to perform any extra citation bookkeeping.

## 6. Recommended v2.3 improvements

V2.3 should be an evolutionary release. It should retain existing projects, Markdown artifacts, directory conventions, and commands while hardening the mechanical interfaces exposed by Sentinel-GL.

### V2.3-01 — Add `gatekeeper contract-preflight`

Run this automatically after `sort-dropbox` and before the first contract begins. It should validate:

- contract and manifest agreement;
- zero-padded IDs and paths;
- dependency references and acyclic execution order;
- Allowed/Frozen overlap;
- outputs outside Allowed Files;
- system-generated artifact namespaces;
- wildcard and recursive verification reach;
- required report sections and parser-compatible verification schema;
- missing inputs or impossible paths;
- scientific tier and required gate consistency;
- High-risk owner binding;
- Human Action fields and dependency effects;
- whether Factory-owned snapshot, stamp, telemetry, and journal paths are writable.

It should emit a machine-readable result and a concise Human-readable summary. The Human should not need to invoke or interpret it manually; the Implementor should run it as part of normal chunk intake and stop only on genuine contract defects.

### V2.3-02 — Reserve a Factory-owned metadata namespace

Define paths such as `project/.factory_state/` as writable by Gatekeeper regardless of contract Allowed Files, but never directly writable by implementation code. Snapshots, stamps, command journals, phase state, and parser caches should live there.

This resolves the Frozen File snapshot conflict while preserving the principle that an Implementor may not expand project scope.

### V2.3-03 — Make verification declarations structured and backward compatible

Allow reports to contain a fenced YAML or JSON block such as:

```yaml
verification:
  - id: unit-tests
    command: python3 -m pytest source/tests/test_example.py -v
    evidence_run_id: run-...
```

Gatekeeper should still parse legacy bullet and `Command:` forms during v2.3 migration. If the manifest contains verification scripts and the report yields zero parsed commands, verification must fail loudly rather than report a successful no-op.

### V2.3-04 — Add an automatic execution journal

Introduce a Gatekeeper wrapper for verification and checkpoint commands. The wrapper should capture command identity, output, exit code, duration, hashes, and changed files. `verify-contract`, `stamp-report`, and report generation should consume this journal.

Agents should still be allowed to run ordinary diagnostics, but any command claimed as verification must have a journal record. This eliminates manual output copying as the source of truth.

### V2.3-05 — Add report generation and report validation

Provide `gatekeeper render-contract-report --contract CNN-NN` and `render-chunk-report --chunk chunkNN`. The generated report should prepopulate immutable contract information, files, verification records, telemetry counts, repository state, and status. The agent adds comprehension, implementation rationale, self-review findings, limitations, and plain-language interpretation.

The Factory should validate the final report against a schema before accepting completion. This reduces duplication while preserving judgment-rich sections.

### V2.3-06 — Capture scoped repository baselines

At contract start, automatically record outer and nested repository status and hashes for relevant paths. At completion, calculate:

- pre-existing changes preserved;
- current-contract changes;
- unexpected changes;
- nested control-plane commits;
- outer implementation commit state.

Replace broad `--allow-dirty` semantics with scoped-delta verification. Do not require the Human to clean or stash their work.

### V2.3-07 — Automate telemetry and decision identifiers

Gatekeeper should write required telemetry events on phase transitions and completion. It should validate JSONL and assign event IDs. The agent should only supply qualitative fields that cannot be derived, such as the reason for a non-trivial decision.

Decision-log entries should receive generated IDs and links to contracts, runs, and affected artifacts. This makes later retrospectives reliable.

### V2.3-08 — Add native temporary role delegation

Represent Human-authorized temporary delegation as structured state. Preserve original ownership and risk. Automatically increase the required review state for affected contracts and include the exception in the chunk report.

The Human’s existing natural-language authorization should be sufficient. No new approval form or repeated confirmation should be introduced.

### V2.3-09 — Rotate handoff bundles safely

Replace destructive clearing of `TAKE_THIS/` copies with automatic archival rotation or deterministic regeneration. Include a bundle manifest and checksum. Starting the next chunk should not depend on the Human remembering whether the previous bundle was downloaded.

### V2.3-10 — Put forensic archives in durable Factory storage

Use a workspace-local private archive directory or configured artifact store. Package the archive, manifest, git identity, and checksums together. If temporary disk must be used during creation, promote the completed bundle automatically before contract completion.

### V2.3-11 — Add a capability/evidence registry

Track whether each Gatekeeper feature has been unit-tested, fixture-tested, used in a real contract, used in a real chunk, and used in a released project. Generate the Known Limitations section from this state. Sentinel-GL should count as real evidence for nested repo isolation, materialization, acquisition auditing, tier checks, Medium/High contract execution, chunk staging, and related paths actually exercised.

### V2.3-12 — Improve scientific gate composition

Provide a unified command that explains which scientific gates apply to a contract and why. It should list claim-tier inference, required verification, recomputation, evidence cross-check, semantic operator tests, acquisition audit, Reality Gate, and release checks as applicable.

The output should distinguish:

- executed and passed;
- executed and failed;
- not applicable;
- required later;
- specified but not implemented;
- unavailable because prerequisite evidence does not exist.

This would reduce the risk of describing an unimplemented or inapplicable check as passed.

### V2.3-13 — Validate manifest/report status as one state

Contract status should not be inferred only from a Markdown heading. Store status in structured state and require the report to render the same value. Gatekeeper should fail on disagreement. The same applies to self-review attempts, owner, risk tier, and scientific claim tier.

### V2.3-14 — Add machine-readable source-evidence records

For literature and factual verification contracts, support a source-evidence JSONL format. Markdown matrices can be generated from it. This should remain an agent task; the Human should not be asked to maintain citation metadata.

### V2.3-15 — Add test fixtures from Sentinel-GL failure cases

Create regression fixtures for:

- report commands present in legacy syntax but missed by the parser;
- manifest commands present while parsed report command count is zero;
- snapshot metadata outside Allowed Files;
- wildcard verification spanning files outside Allowed Files;
- dirty outer repository plus clean nested repository;
- temporary Architect delegation;
- `TAKE_THIS` rotation without prior retrieval;
- acquisition shims that appear safe but call a fallback generator indirectly;
- quarantine entries attempting promotion to release evidence;
- a descriptive literature matrix that must not close a scientific gate.

These fixtures would turn Sentinel-GL’s experience into durable Factory behavior.

## 7. What would justify Factory v3

The recommendations above can be implemented incrementally, but the deeper lesson is that Factory v2.2 behaves like a state machine expressed through Markdown and agent memory. A real v3 should make that state machine explicit.

### 7.1 Structured contract as the source of truth

In v3, each contract should have one canonical structured representation, validated against a versioned schema. It should contain objective, context, dependencies, owner, executor, risk, scientific tier, files, invariants, stop conditions, outputs, verification, evidence requirements, Human-only actions, and Definition of Done.

The Markdown contract should be generated from this representation for readability. Agents may reason over the Markdown view, but Gatekeeper should execute the structured form. This removes parser ambiguity and documentation drift.

### 7.2 Event-sourced execution

Every meaningful transition should be an append-only event:

- chunk received;
- contract preflight passed or failed;
- phase entered;
- checkpoint started;
- file baseline captured;
- command executed;
- verification result recorded;
- self-review finding opened or resolved;
- stop condition triggered;
- role delegated;
- report rendered;
- contract completed, flagged, or blocked;
- chunk packaged;
- review verdict received;
- release gate executed.

Current state should be derived from events. Reports, telemetry, dashboards, and handoff bundles should be projections of the same record rather than separately maintained narratives.

### 7.3 Policy engine for authority and risk

V3 should separate policy from implementation. A policy engine should decide whether a requested transition is permitted based on role, risk, Human authorization, dependency state, scientific tier, and evidence prerequisites.

For example, the Sentinel-GL outage override would not require editing the contract or pretending the owner changed. The event stream would record that the Architect remained the owner, the Implementor became a temporary executor for listed contracts, the Human authorized it, and independent review remained outstanding.

### 7.4 First-class artifact and taint graph

Sentinel-GL’s core problem was transitive evidence taint. V3 should represent artifacts and derivations as a graph:

- source observation;
- acquisition run;
- raw artifact;
- derived feature artifact;
- fitted transform;
- model checkpoint;
- prediction row set;
- metric artifact;
- figure/table;
- claim;
- manuscript section;
- release bundle.

Each node should carry identity, hash, provenance, trust state, and run linkage. If an upstream node becomes quarantined, downstream nodes should be marked tainted automatically. Rebuilding a valid upstream boundary should not silently untaint downstream artifacts; they must be regenerated or independently verified.

### 7.5 Capability-aware gates

V3 should have a registry of actual checks implemented by the installed Factory version. A gate declaration should bind to a specific capability version. Reports should say exactly what ran, not what a specification document says might exist.

This would prevent accidental overstatement such as reporting a Reality Gate pass when only a schema or placeholder exists.

### 7.6 Repository abstraction

V3 should treat repositories as named resources:

- public implementation repository;
- private control-plane repository;
- optional data/artifact store;
- optional manuscript repository;
- optional release mirror.

Status, diffs, commits, and cleanliness should be reported per resource and as an aggregate delivery state. This would handle Sentinel-GL’s nested repository cleanly without conflating implementation and control-plane history.

### 7.7 Generated review packages

Chunk Review packages should be generated deterministically from state. They should include:

- aggregate report;
- each contract report;
- structured status manifest;
- relevant diffs;
- command evidence and hashes;
- unresolved risks;
- role exceptions;
- scientific gate state;
- artifact graph changes;
- exact review questions for Medium/High work.

The Human should still be able to hand the package to the Architect, but should not need to gather raw diffs or outputs manually. Bundle generation should satisfy that existing responsibility automatically.

### 7.8 Agent-context optimization

V3 should distinguish machine-checkable state from reasoning that deserves model context. The agent should not need to repeatedly restate immutable file lists or paste thousands of lines of test output into its active context. Gatekeeper can provide compact signed summaries with direct access to full evidence when needed.

This would improve reliability because the model’s attention can remain on semantics and failure analysis.

### 7.9 Compatibility layer

V3 should import v2.x projects:

- parse existing manifests and contracts;
- preserve original Markdown unchanged as historical artifacts;
- create structured contract records with provenance to the source files;
- import existing reports, marking fields as parsed, inferred, or unavailable;
- import nested repository history and telemetry;
- never upgrade a historical verification claim merely because the new schema is stricter.

The migration should be agent-operated and reversible. The Human should not have to rewrite old projects.

## 8. Human Operator responsibility constraint

The Human Operator’s role should remain limited to decisions that are genuinely about authority, project intent, resource ownership, or irreducibly human scientific validation. A Factory upgrade must not shift software bookkeeping onto the Human.

Specifically, the upgrade should not require the Human to:

- rewrite informal instructions into command syntax;
- fill out role-delegation forms after already giving an explicit instruction;
- manually reconcile Allowed Files with wildcard verification commands;
- manually create snapshots or stamps;
- copy command output into reports;
- maintain telemetry JSONL;
- update feature maturity counts in Factory documentation;
- remember to rescue temporary forensic bundles;
- confirm that `TAKE_THIS` was downloaded before a new chunk can start;
- clean or stash an unrelated dirty worktree solely for Factory convenience;
- gather diffs and outputs by hand for Chunk Review;
- duplicate information already present in contracts, manifests, git, or execution logs;
- decide whether a Markdown formatting variation is mechanically parseable.

The Factory and its agents should assume responsibility for all of those tasks.

Existing Human responsibilities may remain where they are genuinely necessary: assigning project authority, supplying credentials or access that cannot be delegated safely, performing specifically required authenticity spot checks, making publication and resource decisions, authorizing exceptional role delegation, and routing a completed review package to the Architect. Even there, the Factory should prepare the exact evidence and instructions so the Human action is minimal and unambiguous.

No recommendation in this document requires an additional Human checkpoint. Some recommendations eliminate current implicit Human burdens.

## 9. Suggested prioritization

### Immediate v2.3 blockers

These should be fixed before calling v2.2’s workflow mechanically reliable across more projects:

1. fail when manifest verification exists but zero report commands are parsed;
2. support legacy and structured verification syntax;
3. reserve a Factory-owned metadata namespace for snapshots, stamps, journals, and state;
4. add contract preflight for Allowed/Frozen/output and wildcard reach conflicts;
5. capture scoped dirty-tree baselines across outer and nested repositories;
6. automate telemetry validity and required completion events;
7. represent temporary role delegation without changing original ownership;
8. rotate or regenerate handoff bundles instead of assuming retrieval;
9. place forensic archives in durable storage automatically.

### Strong v2.3 quality improvements

These significantly reduce agent attention cost and reporting drift:

1. execution journal;
2. generated contract and chunk reports;
3. unified gate applicability output;
4. structured source-evidence records;
5. capability maturity registry;
6. regression fixtures based on Sentinel-GL;
7. explainable tier-inference traces.

### V3 boundary

Adopt the v3 label when all of the following are true:

1. structured contracts, not Markdown parsing, are authoritative;
2. execution is event-sourced;
3. reports and telemetry are generated projections;
4. authority/risk decisions are enforced by a policy engine;
5. artifact provenance and taint are first-class graph state;
6. repositories and artifact stores are named resources;
7. v2.x projects can be imported without Human rewriting.

## 10. Proposed acceptance tests for the upgrade

The upgrade should be considered successful only if it can replay the difficult parts of Sentinel-GL Chunk 01 correctly.

### Acceptance test A — legacy report compatibility

Given a contract with three manifest verification scripts and a report using the older `Command:` style, the Factory either parses and reruns all three or fails with an explicit migration error. It must never report success after extracting zero commands.

### Acceptance test B — wildcard scope conflict

Given a verification command scanning `source/data/acquisition/*.py` and an Allowed Files list missing one matched file, preflight fails before implementation and lists the exact mismatch.

### Acceptance test C — snapshot ownership

Given Frozen Files and a contract that does not list Factory metadata paths as Allowed Files, Gatekeeper still creates and validates snapshots through its reserved namespace without granting the Implementor broader source permissions.

### Acceptance test D — dirty rehabilitation repository

Given unrelated pre-existing modifications, the Factory allows work on permitted files, preserves unrelated changes, detects any accidental overlap, and reports the contract delta without requiring a clean tree.

### Acceptance test E — temporary High-risk delegation

Given an explicit Human message authorizing the Implementor to execute specified Architect-owned High-risk contracts for one chunk, the Factory records the authorization, preserves original owner/risk metadata, allows only those contracts, expires the delegation at chunk end, and flags independent review automatically.

### Acceptance test F — dual repository state

Given a clean nested control-plane repository and a dirty outer implementation repository, the Factory reports both accurately and does not describe the whole project as clean.

### Acceptance test G — durable archive

Given a forensic inventory contract, the generated archive remains available in durable private project storage after temporary directories are cleared, and its checksum matches the contract evidence.

### Acceptance test H — handoff rotation

Given a staged Chunk 01 bundle and arrival of Chunk 02, the Factory preserves or deterministically regenerates Chunk 01’s bundle without asking the Human whether it was retrieved.

### Acceptance test I — scientific non-closure

Given a literature matrix and MAR coverage document but no clean data or evaluation evidence, the Factory marks descriptive work complete while keeping MAR gates unresolved. It must not infer scientific validation from documentation completeness.

### Acceptance test J — taint propagation

Given a quarantined acquisition artifact with downstream features, model, predictions, metrics, and claims, all downstream nodes become inadmissible automatically until regenerated from a valid boundary.

## 11. Final evaluation of Factory v2.2

Factory v2.2 succeeded at the most important level: it made me behave more like a careful scientific implementor and less like a code-completion system. It forced explicit scope, prevented silent substitution, treated inherited evidence as untrusted until proven otherwise, separated protocol design from result interpretation, preserved negative outcomes, and produced a reviewable record.

Its shortcomings are the natural result of a system whose governance matured faster than its execution substrate. The rules understand problems that the tooling only partially models. Markdown headings stand in for schemas. Repeated prose stands in for state. Manually copied outputs stand in for an execution journal. A general dirty-tree bypass stands in for scoped provenance. Human-readable exceptions stand in for authority events. These are fixable, and Sentinel-GL provides concrete regression cases for fixing them.

I would trust Factory v2.2’s principles as the foundation of the next release. I would not remove its strictness to make it feel lighter. I would encode more of that strictness in deterministic machinery so the agent cannot accidentally omit it and the Human does not inherit extra work.

If the upgrade remains compatible with the current document-driven workflow, v2.3 is the right name and would be a meaningful improvement. If the upgrade makes structured contracts and an event journal authoritative, generates reports and telemetry, enforces authority through policy, and tracks artifact taint as graph state, it is substantial enough to be Factory v3.

The most important design rule for either path is simple: increase machine responsibility, increase agent accountability, preserve scientific skepticism, and do not add responsibilities to the Human Operator.
