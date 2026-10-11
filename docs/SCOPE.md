# Scope and preservation

This focused checkout contains the original G4a-G4d experiment path from the
DRF-BLE project. G4a implements the flood expert, G4b implements controlled
retransmission, G4c compares complete mode runs with matched initialization,
and G4d applies the frozen confidence-interval gate.

All source, selected tests, scenario definitions, and historical result files
listed in `PROVENANCE.json` are byte-for-byte copies. The new scripts provide
entry points for smoke execution, frozen-result verification, full replay, and
release-file verification. They do not tune parameters, change policies,
change seeds, alter the gate, or replace the stored negative result.

`.gitattributes` disables automatic newline conversion so the recorded bytes
and file hashes survive a Git checkout. This does not establish runtime
portability to an untested platform.

Legacy policy code required by the preserved modules is included to keep the
import graph intact. Its presence is not a claim that the later DRF/CGF/ML
research lines have been validated or released here. Some comments refer to
earlier phase documents that are outside this focused checkout.

Plans C/D/E, the S/N evidence-tooling work, notebooks, model weights, sealed
truth, administrator materials, personal environment records, and large
decision logs are excluded. This checkout does not depend on the original
folder or its machine-specific ns-3 installation.

The `outputs/` CSVs and G4d report are the preserved historical simulation
results. Small artificial fixtures inside tests are software checks, not
additional experimental evidence. New replay files go only into the ignored
`reproduction/` directory and are compared with the preserved results.
