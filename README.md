# Plagiarism Detection and Document Similarity Analysis System

**Semantic AI upgrade** — Sentence-Transformer embeddings + FAISS retrieval + explainable sentence-level evidence.

Team:
- T. Rishikesh (2510030166)
- V. Pushkar (2510030104)
- T. Varshith Teja (2510030321)
- K. Praneeth Reddy (2510030211)

Repository: https://github.com/kolanpraneeth-hue/A7_DSA3_G2

---

## What this system does

1. Extracts text from **TXT / PDF / DOCX**
2. Splits documents into sentences
3. Embeds sentences with a **Sentence-Transformer** model
4. Indexes the reference document with **FAISS**
5. Finds nearest semantic matches for each query sentence
6. Produces:
   - Overall similarity percentage
   - Decision class (LOW / MEDIUM / HIGH)
   - Top matched sentence pairs as evidence
   - Optional HTML report

---

## Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

First run will download the Sentence-Transformer model (`all-MiniLM-L6-v2`, ~80 MB).

---

## Usage

```bash
# Basic comparison (paraphrased student essay vs source)
python detect_similarity.py \
  --query samples/student_essay.txt \
  --ref samples/source_paper.txt

# Also save HTML report
python detect_similarity.py \
  -q samples/student_essay.txt \
  -r samples/source_paper.txt \
  --html --out reports/similarity_report.html

# Unrelated documents (should score LOW)
python detect_similarity.py \
  -q samples/unrelated.txt \
  -r samples/source_paper.txt
```

### Optional arguments

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `all-MiniLM-L6-v2` | Any Sentence-Transformers model name |
| `--high` | `0.65` | Threshold for HIGH class |
| `--medium` | `0.40` | Threshold for MEDIUM class |
| `--html` | off | Write HTML report |
| `--out` | `reports/similarity_report.html` | HTML path |

---

## Expected output (example)

```
[INFO] Loading Sentence-Transformer model: all-MiniLM-L6-v2
[INFO] Extracting & chunking documents ...
[INFO] Query sentences : 10
[INFO] Reference sentences : 10
[INFO] Building FAISS index (dim=384) ...
[INFO] Computing semantic nearest-neighbour matches ...

========== SIMILARITY REPORT ==========
Overall Semantic Similarity: 72.4%
Decision Class             : HIGH
...
Top matched sentence pairs:
  [1] score=0.91
      Q: "Machine learning models learn patterns from data..."
      R: "Machine learning is a branch of artificial intelligence..."
=======================================
```

---

## Project structure

```
plagiarism_system/
├── detect_similarity.py   # Main CLI
├── semantic_engine.py     # Embeddings + FAISS + scoring
├── preprocess.py          # File loading & sentence splitting
├── requirements.txt
├── README.md
├── samples/
│   ├── source_paper.txt
│   ├── student_essay.txt  # paraphrased version
│   └── unrelated.txt
└── reports/               # HTML reports (created on demand)
```

---

## Methodology summary

| Stage | Technique |
|-------|-----------|
| Input | TXT, PDF (PyMuPDF), DOCX (python-docx) |
| Chunking | NLTK sentence tokenizer |
| Representation | Sentence-Transformer embeddings |
| Retrieval | FAISS IndexFlatIP (cosine via normalised vectors) |
| Scoring | Mean of best match scores → percentage |
| Evidence | Top sentence pairs above a soft floor |
| Decision | Configurable LOW / MEDIUM / HIGH thresholds |

---

## Important note

**Similarity is an indicator, not proof of plagiarism.**  
Final academic or institutional decisions should always include human review.

---

## Licence

For academic / educational use as part of the PBL project.
