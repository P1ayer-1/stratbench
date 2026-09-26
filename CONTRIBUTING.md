# Contributing

Thanks for helping. A few rules keep this toolkit honest; please follow them
in every pull request.

## Ground rules

- **Tests pass with no network and no credentials.** Venue adapters take an
  injectable `httpx.Client` (use `httpx.MockTransport`) and the recorder
  takes an injectable stream. Run `python -m pytest -q` before opening a PR.
- **Hand-computed expected values.** Write the arithmetic in a comment
  (`0.07 * 0.5 * 0.5 = 0.0175`) and say in the test's docstring which
  failure it guards against.
- **Refuse with every reason.** A planner, the risk veto and the runner
  return *all* failing gates, each with what to do next, never just the
  first.
- **One currency.** Nothing above `venues/` handles a NO price.
- **Archive first, parse second.** Never make a parse error able to stop
  the raw archive.
- **No live by default.** Nothing may send an order without `--live
  --confirm`, a geofence pass, the user's own key and the risk veto.
- **Keep the veto import-free.** `risk.py` imports nothing from `predkit`.

## Fee rows and venue facts

A new fee tier or venue rule needs a date, a source (URL or document
name), and `verified=False` unless it was checked against a real fill.
Add a new row rather than editing an old one. Say in the docstring what
was read live and what was assumed.

## New strategies and venues

See "Adding a strategy" in the README. New venue adapters implement the
`Venue` protocol in `predkit/venues/__init__.py`, translate to the YES book
inside the adapter, and must add the venue's published jurisdiction
restrictions to `geofence.py` with sources before any live path is wired.

## Conduct and scope

Keep discussion technical and respectful. Do not submit code that helps
anyone evade a venue's geoblocking, KYC or terms of service, and do not
include private keys, wallet addresses of individuals, or analysis of
identifiable traders.
