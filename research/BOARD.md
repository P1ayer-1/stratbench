# Research board

One row per idea, rewritten in place, newest activity at top. Dead ideas keep
their row and the number that killed them: a negative result stops the next
person repeating the test. Maintained by the `strategy-lab` agent (see
`.claude/agents/`).

Stages: `sourced` -> `hypothesis` -> `planned` -> `local` -> `remote` (optional)
-> `verdict`, where the verdict is `dead`, `promising` or `needs-more-data`.
An idea can also be `dropped` at any stage, with the reason.

Domain: `prediction` (binary prediction markets) or `perps` (crypto
perpetual futures).

| id | idea | domain | stage | verdict so far | next step | owner | updated (UTC) |
|---|---|---|---|---|---|---|---|

## Layout

Each idea gets an id `R<NNN>-<slug>` (next free number) and its own folder,
`research/R<NNN>-<slug>/`. `card.md` holds the source and the falsifiable
hypothesis (scout, then theorist); `plan.md` the pre-registered test, with kill
criteria and the frozen scoring code recorded before any out-of-sample data is
fetched (test-planner); `local.md` the local results and verdict
(local-tester); `remote.md` the results of any steps run on a remote box near
the venue (remote-tester, optional); and `briefs/WP<n>.md` the work-package
briefs local-tester writes for its test writer, code writer and reviewer.
Agents hand work to each other through these files.

## Explored directions

Every direction the lab sends out, so later rounds look somewhere else.

| date (UTC) | direction | why chosen | agent | ideas produced |
|---|---|---|---|---|
