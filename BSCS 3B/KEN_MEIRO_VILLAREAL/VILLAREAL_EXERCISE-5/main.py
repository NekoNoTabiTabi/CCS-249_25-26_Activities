"""
Skip-Gram Word Embedding Training and Evaluation Pipeline
= Main Orchestration Script =

This script orchestrates the complete workflow:
1. Download and preprocess Wikipedia article (task_1_dataset.py)
2. Train skip-gram model with negative sampling (task_2_training.py)
3. Evaluate embeddings on test sets (task_3_evaluation.py)
4. Generate comprehensive reports (task_4_reporting.py)

Requirements:
    pip install requests beautifulsoup4 nltk gensim scikit-learn scipy

Optional NLTK data download:
    python -m nltk.downloader punkt punkt_tab stopwords
"""

import subprocess
import sys
import os
from pathlib import Path


def run_task(script_name: str, task_num: int, task_desc: str):
    """Run a task script and handle errors."""
    print(f"\n{'='*70}")
    print(f"RUNNING TASK {task_num}: {task_desc}")
    print(f"Script: {script_name}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            cwd=Path(__file__).parent
        )
        print(f"\n✓ Task {task_num} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Task {task_num} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ Task {task_num} failed with error: {e}")
        return False


def main():
    print("\n" + "🚀 "*20)
    print("SKIP-GRAM WORD EMBEDDING TRAINING PIPELINE")
    print("🚀 "*20 + "\n")
    
    tasks = [
        ("task_1_dataset.py", 1, "Dataset Selection and Preparation (Wikipedia)"),
        ("task_2_training.py", 2, "Train Skip-gram with Negative Sampling"),
        ("task_3_evaluation.py", 3, "Evaluate Embeddings on Test Sets"),
        ("task_4_reporting.py", 4, "Generate Comprehensive Reports"),
    ]
    
    results = []
    
    for script, num, desc in tasks:
        success = run_task(script, num, desc)
        results.append((num, desc, success))
        if not success:
            print(f"\n⚠ Pipeline stopped at Task {num}")
            break
    
    # Print summary
    print(f"\n{'='*70}")
    print("PIPELINE EXECUTION SUMMARY")
    print(f"{'='*70}\n")
    
    all_success = True
    for num, desc, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  Task {num}: {status} - {desc}")
        if not success:
            all_success = False
    
    print()
    if all_success:
        print("✓ All tasks completed successfully!")
        print("\nGenerated Files:")
        print("  • corpus_sentences.pkl    - Preprocessed corpus")
        print("  • skipgram_model.model    - Trained word2vec model")
        print("  • evaluation_results.pkl  - Test set evaluation results")
        print("  • final_report.json       - Final comprehensive report")
    else:
        print("✗ Pipeline execution failed. Check errors above.")
        sys.exit(1)
    
    print()


if __name__ == "__main__":
    main()
