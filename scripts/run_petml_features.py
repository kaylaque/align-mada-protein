import subprocess
from pathlib import Path
import shutil
import os

REPO = Path(__file__).resolve().parents[1]
PETML_ROOT = REPO / "external_models" / "PETML"

seqfile = REPO / "dataset" / "external_activity_sequences.fasta"
outdir = REPO / "output" / "petml_features_external_activity"

for tool in ["hmmsearch", "mafft"]:
    if shutil.which(tool) is None:
        raise RuntimeError(f"Missing required tool in PATH: {tool}")

outdir.mkdir(parents=True, exist_ok=True)

cmd = [
    "python", "-m", "petml.run",
    "--seqfile", str(seqfile),
    "--outdir", str(outdir),
    "--delete_temp_files", "1",
]

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True, cwd=str(PETML_ROOT))
print("Done. Check:", outdir)