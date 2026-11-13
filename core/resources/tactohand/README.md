# Versioned TactoHand training resources

The statistics and gigahands curation files preserve prior artifacts
byte-for-byte. They contain aggregate counts, dataset names and
selected object display names, not MANO model files, frames or checkpoints.
`manifest.json` records size, previous path and available provenance.

| File | Consumer | Snapshot meaning |
|---|---|---|
| `contact_statistics_v5.npz` | conditional VCB contact loss | schema 1; 24 datasets; embedded source `labels_v5:labels_v5_preview:24ds`; engaged hand-frame counts |
| `force_statistics_v5.npz` | active/zero force class weights | schema 1; 17 force datasets; generator counts magnitudes above 0.05 N and 0.01 N |
| `gigahands_tracking_good_objects.allowlist.json` | label reader object filtering | exact original dataset-to-object-display-name mapping; shared asset join remains unchanged |

These are the existing snapshots, not statistics recomputed from today's
installed labels. The original full generation invocation and complete input-label version
were not recovered; the curation list's original review provenance
is also unresolved. Copying these files does not establish fresh data validation.

Defaults use `core/resources/tactohand/...`; `resolve_resource_path` anchors
that prefix to the installed code location, independent of the working
directory. Absolute paths and other relative paths retain their previous
meaning, including paths in saved historical run configurations. Old artifact
files remain in place. New snapshots must be deliberately selected in config;
there is no automatic substitution for missing historical resources.

The original contact/force statistics builders were retired during source
cleanup; this checkout does not provide a supported regeneration command.
Before replacing these snapshots, restore or implement a reviewed operation
under `scripts/`, select a labels-v5 tree and write to a separate output path.
Record the actual command, dataset selection, input-label version and consumer
checks. Preserve the current bundled resources until the replacement is
validated. Object curation also requires its own provenance review.

MANO source decisions are not bundled here. The old default registry is absent;
the one discovered review report contains an unresolved decision and is not an
acceptable replacement. Installed `mano_contract` / `mano_source_basis` metadata
continues to take precedence. Legacy data without those must provide a reviewed
registry through `MANO_BASIS_DECISIONS`. Missing
registries fail with an actionable error; dataset names never imply a basis.
