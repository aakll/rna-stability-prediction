import RNA
import pandas as pd
from Bio import SeqIO
import glob

# Read all sequences, get max MFE per nt later for normalization
rows = []

def parse_dot_bracket(structure):
    stack = []
    stems = []
    unpaired_regions = []

    paired_positions = set()
    for i, c in enumerate(structure):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                j = stack.pop()
                paired_positions.update([i, j])

    # Paired bases %
    paired_pct = round(len(paired_positions) / len(structure), 3)

    # Detect stems
    stems = []
    stem_len = 0
    for c in structure:
        if c == '(':
            stem_len += 1
        else:
            if stem_len > 0:
                stems.append(stem_len)
                stem_len = 0
    if stem_len > 0:
        stems.append(stem_len)
    num_stems = len(stems)
    avg_stem_len = round(sum(stems)/num_stems, 2) if num_stems else 0

    # Detect loops and bulges
    num_loops = 0
    num_bulges = 0
    i = 0
    while i < len(structure):
        if structure[i] == '.':
            start = i
            while i < len(structure) and structure[i] == '.':
                i += 1
            end = i - 1

            # Check flanking characters
            left = structure[start-1] if start > 0 else None
            right = structure[end+1] if end+1 < len(structure) else None

            if left in ('(',) and right in ('(',):
                num_bulges += 1
            else:
                num_loops += 1
        else:
            i += 1

    return paired_pct, num_stems, num_loops, num_bulges, avg_stem_len

# First pass to compute MFE_per_nt max for scaling
all_mfe_per_nt = []

for file in glob.glob("* 40.txt"):
    for record in SeqIO.parse(file, "fasta"):
        seq = str(record.seq).upper()
        fc = RNA.fold_compound(seq)
        _, mfe = fc.mfe()
        length = len(seq)
        mfe_per_nt = -mfe / length
        all_mfe_per_nt.append(mfe_per_nt)

max_mfe_per_nt = max(all_mfe_per_nt)

# Now process each sequence fully
for file in glob.glob("* 40.txt"):
    rna_type = file.replace(" 40.txt", "")
    for record in SeqIO.parse(file, "fasta"):
        seq = str(record.seq).upper()
        length = len(seq)
        gc = round((seq.count('G') + seq.count('C')) / length, 3)

        fc = RNA.fold_compound(seq)
        structure, mfe = fc.mfe()

        mfe_per_nt = round(-mfe / length, 4)
        stability_score = round(mfe_per_nt / max_mfe_per_nt, 3)

        paired_pct, stems, loops, bulges, avg_stem_len = parse_dot_bracket(structure)

        rows.append({
            "RNA_ID": record.id,
            "RNA_Type": rna_type,
            "Length": length,
            "GC_Content": gc,
            "MFE": round(mfe, 2),
            "MFE_per_nt": mfe_per_nt,
            "Paired_Bases_Pct": paired_pct,
            "Num_Stems": stems,
            "Num_Loops": loops,
            "Num_Bulges": bulges,
            "Avg_Stem_Length": avg_stem_len,
            "Stability_Score": stability_score
        })

df = pd.DataFrame(rows)
df.to_csv("rna_structure_data.csv", index=False)
print("Saved: rna_structure_data.csv")
