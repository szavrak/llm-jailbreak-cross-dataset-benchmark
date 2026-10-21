"""Download and normalize all jailbreak datasets into uniform CSV format.

Output: data/{name}.csv with columns: text, label (0=benign, 1=jailbreak)

Run: pip install datasets pandas requests
     python data/download_datasets.py
"""
import os
import pandas as pd
import requests
import json
import io

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def save_dataset(name, texts, labels):
    """Save a normalized dataset CSV."""
    df = pd.DataFrame({'text': texts, 'label': labels})
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip().str.len() > 0]
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    path = os.path.join(DATA_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    n_jb = int(df['label'].sum())
    print(f"  ✓ {name}: {len(df)} total ({n_jb} jailbreak, {len(df)-n_jb} benign) -> {path}")
    return df


def download_dan():
    """DAN — 'Do Anything Now' dataset from verazuo/jailbreak_llms (ACM CCS 2024).

    Contains jailbreak prompts collected from Reddit, Discord, and websites.
    """
    print("\n[DAN] Downloading from GitHub...")

    # Jailbreak prompts
    url_jb = "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/jailbreak_prompts_2023_05_07.csv"
    r = requests.get(url_jb)
    r.raise_for_status()
    jb_df = pd.read_csv(io.StringIO(r.text))
    jb_texts = jb_df['prompt'].tolist()

    # Regular/benign prompts
    url_benign = "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/regular_prompts_2023_05_07.csv"
    r = requests.get(url_benign)
    r.raise_for_status()
    benign_df = pd.read_csv(io.StringIO(r.text))
    benign_texts = benign_df['prompt'].tolist()

    texts = jb_texts + benign_texts
    labels = [1] * len(jb_texts) + [0] * len(benign_texts)

    return save_dataset('dan', texts, labels)


def download_markush():
    """markush1/LLM-Jailbreak-Classifier — large public jailbreak dataset from HuggingFace.

    Replaces Hyperion (gated/private). This is a publicly available
    jailbreak classification dataset.
    """
    print("\n[markush] Downloading from HuggingFace...")

    from datasets import load_dataset
    ds = load_dataset('markush1/LLM-Jailbreak-Classifier', split='train')
    df = ds.to_pandas()

    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Shape: {df.shape}")
    print(f"  Sample:\n{df.head(2)}")

    # Map columns — check what's available
    text_col = next((c for c in ['prompt', 'text', 'input'] if c in df.columns), df.columns[0])
    label_col = next((c for c in ['label', 'is_jailbreak', 'class'] if c in df.columns), df.columns[-1])

    texts = df[text_col].tolist()
    raw_labels = df[label_col].tolist()

    # Normalize labels to 0/1
    unique_labels = set(raw_labels)
    print(f"  Unique labels: {unique_labels}")
    if unique_labels <= {0, 1}:
        labels = raw_labels
    else:
        label_map = {}
        for l in unique_labels:
            ls = str(l).lower().strip()
            if 'jailbreak' in ls or 'malicious' in ls or 'harmful' in ls or ls == '1':
                label_map[l] = 1
            else:
                label_map[l] = 0
        labels = [label_map[l] for l in raw_labels]

    return save_dataset('markush', texts, labels)


def download_jackhhao():
    """jackhhao/jailbreak-classification dataset from HuggingFace."""
    print("\n[jackhhao] Downloading from HuggingFace...")

    from datasets import load_dataset
    ds = load_dataset('jackhhao/jailbreak-classification', split='train')
    df = ds.to_pandas()

    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Sample: {df.head(2)}")

    texts = df['prompt'].tolist()
    # Label column is 'type' (benign/jailbreak)
    label_col = 'type' if 'type' in df.columns else 'label'
    label_map = {'jailbreak': 1, 'benign': 0}
    labels = [label_map.get(str(l).lower().strip(), 0) for l in df[label_col]]

    return save_dataset('jackhhao', texts, labels)


def download_advbench():
    """AdvBench — 520 harmful behaviors from Zou et al. 2023.

    All entries are harmful. We pair with benign prompts from Alpaca dataset.
    """
    print("\n[AdvBench] Downloading from GitHub...")

    # Harmful behaviors
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    r = requests.get(url)
    r.raise_for_status()
    harm_df = pd.read_csv(io.StringIO(r.text))
    harm_texts = harm_df['goal'].tolist()

    # Benign prompts — sample from Alpaca to match size
    print("  Downloading benign prompts from Alpaca...")
    url_alpaca = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    r = requests.get(url_alpaca)
    r.raise_for_status()
    alpaca = json.loads(r.text)
    benign_texts = [item['instruction'] for item in alpaca if item['instruction'].strip()]
    # Sample same number as harmful
    import random
    random.seed(42)
    benign_texts = random.sample(benign_texts, min(len(harm_texts), len(benign_texts)))

    texts = harm_texts + benign_texts
    labels = [1] * len(harm_texts) + [0] * len(benign_texts)

    return save_dataset('advbench', texts, labels)


def download_harmbench():
    """HarmBench — 400+ harmful behaviors from Mazeika et al. 2024.

    All entries are harmful. We pair with benign prompts from Alpaca.
    """
    print("\n[HarmBench] Downloading from GitHub...")

    url = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    r = requests.get(url)
    r.raise_for_status()
    harm_df = pd.read_csv(io.StringIO(r.text))
    harm_texts = harm_df['Behavior'].tolist()

    # Benign prompts from Alpaca
    print("  Downloading benign prompts from Alpaca...")
    url_alpaca = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    r = requests.get(url_alpaca)
    r.raise_for_status()
    alpaca = json.loads(r.text)
    benign_texts = [item['instruction'] for item in alpaca if item['instruction'].strip()]
    import random
    random.seed(43)
    benign_texts = random.sample(benign_texts, min(len(harm_texts), len(benign_texts)))

    texts = harm_texts + benign_texts
    labels = [1] * len(harm_texts) + [0] * len(benign_texts)

    return save_dataset('harmbench', texts, labels)


def download_jailbreakbench():
    """JailbreakBench — standardized jailbreak evaluation (NeurIPS 2024).

    Behaviors are embedded in the pip package source code as a CSV.
    We download the source file directly from GitHub.
    """
    print("\n[JailbreakBench] Downloading from GitHub...")

    # Try multiple known paths for the behaviors CSV in the repo
    urls = [
        "https://raw.githubusercontent.com/JailbreakBench/jailbreakbench/main/src/jailbreakbench/data/behaviors.csv",
        "https://raw.githubusercontent.com/JailbreakBench/jailbreakbench/main/jailbreakbench/data/behaviors.csv",
        "https://raw.githubusercontent.com/JailbreakBench/jailbreakbench/main/data/behaviors.csv",
    ]

    behav_df = None
    for url in urls:
        r = requests.get(url)
        if r.status_code == 200:
            behav_df = pd.read_csv(io.StringIO(r.text))
            print(f"  Found behaviors at: {url}")
            break

    if behav_df is None:
        # Fallback: read CSV directly from installed package without importing it
        print("  CSV not found via HTTP. Checking installed package files...")
        import glob as _glob
        candidates = _glob.glob('/opt/anaconda3/**/jailbreakbench/data/behaviors.csv', recursive=True)
        candidates += _glob.glob(os.path.expanduser('~/.local/**/jailbreakbench/data/behaviors.csv'), recursive=True)
        if candidates:
            behav_df = pd.read_csv(candidates[0])
            print(f"  Found local: {candidates[0]}")
        else:
            print("  ERROR: Could not find jailbreakbench behaviors. Skipping.")
            return None

    if behav_df is not None:
        # Find the goal/behavior column
        for col in ['Goal', 'Behavior', 'goal', 'behavior']:
            if col in behav_df.columns:
                jb_texts = behav_df[col].tolist()
                break
        else:
            print(f"  Columns found: {behav_df.columns.tolist()}")
            jb_texts = behav_df.iloc[:, 0].tolist()

    print(f"  Found {len(jb_texts)} jailbreak behaviors")

    # Benign prompts from Alpaca
    print("  Downloading benign prompts from Alpaca...")
    url_alpaca = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    r = requests.get(url_alpaca)
    r.raise_for_status()
    alpaca = json.loads(r.text)
    benign_texts = [item['instruction'] for item in alpaca if item['instruction'].strip()]
    import random
    random.seed(44)
    benign_texts = random.sample(benign_texts, min(len(jb_texts), len(benign_texts)))

    texts = jb_texts + benign_texts
    labels = [1] * len(jb_texts) + [0] * len(benign_texts)

    return save_dataset('jailbreakbench', texts, labels)


def download_hawkins():
    """Hawkins merged dataset — derived from DAN + garak + jackhhao.

    We merge DAN jailbreak prompts with jackhhao benign prompts,
    following the approach described in Hawkins et al. 2025.
    """
    print("\n[Hawkins] Creating merged dataset from DAN + jackhhao...")

    # Load DAN jailbreak prompts
    url_jb = "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/jailbreak_prompts_2023_05_07.csv"
    r = requests.get(url_jb)
    r.raise_for_status()
    jb_df = pd.read_csv(io.StringIO(r.text))
    jb_texts = jb_df['prompt'].tolist()

    # Load jackhhao benign prompts
    from datasets import load_dataset
    ds = load_dataset('jackhhao/jailbreak-classification', split='train')
    jh_df = ds.to_pandas()
    label_col = 'type' if 'type' in jh_df.columns else 'label'
    benign_mask = jh_df[label_col].str.lower().str.strip() == 'benign'
    benign_texts = jh_df[benign_mask]['prompt'].tolist()

    texts = jb_texts + benign_texts
    labels = [1] * len(jb_texts) + [0] * len(benign_texts)

    return save_dataset('hawkins', texts, labels)


def create_zip():
    """Zip all CSVs into data.zip for Colab."""
    import subprocess
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not csv_files:
        print("\nNo CSV files to zip.")
        return

    zip_path = os.path.join(os.path.dirname(DATA_DIR), 'data.zip')
    # Remove old zip
    if os.path.exists(zip_path):
        os.remove(zip_path)

    subprocess.run(['zip', '-j', zip_path] + [os.path.join(DATA_DIR, f) for f in csv_files],
                   check=True)
    print(f"\n✓ Created {zip_path} with {len(csv_files)} datasets")


if __name__ == '__main__':
    print("=" * 60)
    print("Jailbreak Dataset Download & Normalize")
    print("=" * 60)

    downloaders = [
        ('DAN', download_dan),
        ('JailbreakBench', download_jailbreakbench),
        # ('markush', download_markush),  # Gated dataset — requires HuggingFace auth
        ('jackhhao', download_jackhhao),
        ('Hawkins', download_hawkins),
        ('AdvBench', download_advbench),
        ('HarmBench', download_harmbench),
    ]

    for name, fn in downloaders:
        try:
            fn()
        except Exception as e:
            print(f"\n  ✗ {name} FAILED: {e}")
            print(f"    You may need to download this dataset manually.")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith('.csv'):
            df = pd.read_csv(os.path.join(DATA_DIR, f))
            n_jb = int(df['label'].sum())
            print(f"  {f}: {len(df)} ({n_jb} jailbreak, {len(df)-n_jb} benign)")

    create_zip()
