"""
Routing accuracy evaluation for CoordinationAgent.

Usage:
    python eval/routing_eval.py

Routes the coordinator can return:
    frequency | kwic | ngram | keyword | conversational | out_of_scope
"""
import sys
import os

# Ensure project root is on the path when run from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.coordination_agent import CoordinationAgent

# ---------------------------------------------------------------------------
# Test cases  (query, expected_route)
# Routes: frequency | kwic | ngram | keyword | conversational | out_of_scope
# ---------------------------------------------------------------------------
TEST_CASES = [
    # --- frequency (6) ---
    ("What are the top 10 most frequent words?",                    "frequency"),
    ("Show me the most common words in the corpus",                 "frequency"),
    ("How often does 'the' appear?",                               "frequency"),
    ("List the 20 highest frequency words",                         "frequency"),
    ("Give me a word frequency breakdown",                          "frequency"),
    ("Which words occur most in this text?",                        "frequency"),

    # --- kwic (5) ---
    ("Show me concordance lines for 'justice'",                     "kwic"),
    ("Show me 'economy' in context with 5 words on each side",      "kwic"),
    ("Find all occurrences of 'freedom' with surrounding context",  "kwic"),
    ("Give me KWIC output for the word 'rights'",                   "kwic"),
    ("I need concordance lines for 'language'",                     "kwic"),

    # --- ngram (5) ---
    ("Find bigrams in this corpus",                                 "ngram"),
    ("What are the top trigrams?",                                  "ngram"),
    ("List collocations with a high PMI score",                     "ngram"),
    ("Show me the most frequent two-word sequences",                "ngram"),
    ("What word combinations appear most often?",                   "ngram"),

    # --- keyword (4) ---
    ("Compare keywords to a reference text",                        "keyword"),
    ("What are the keywords compared to news articles?",            "keyword"),
    ("Which words are statistically distinctive in my corpus?",     "keyword"),
    ("Perform a keyword analysis against the reference corpus",     "keyword"),

    # --- conversational (3) ---
    ("What does KWIC stand for?",                                   "conversational"),
    ("What can you do?",                                            "conversational"),
    ("Hello, how does this system work?",                           "conversational"),

    # --- out_of_scope (2) ---
    ("Write me a poem about autumn",                                "out_of_scope"),
    ("What is the weather like today?",                             "out_of_scope"),
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("Initialising CoordinationAgent … (this may take a moment)\n")
    coordinator = CoordinationAgent()

    correct = 0
    failures = []

    for query, expected in TEST_CASES:
        actual = coordinator.route_query(query)
        match = actual == expected
        correct += match
        status = "PASS" if match else "FAIL"
        print(f"{status} | expected={expected:<15} | got={actual:<15} | {query[:55]}")
        if not match:
            failures.append((query, expected, actual))

    total = len(TEST_CASES)
    pct = correct / total * 100
    print(f"\n{'='*70}")
    print(f"Routing Accuracy: {correct}/{total} = {pct:.1f}%")

    if failures:
        print(f"\nFailed cases ({len(failures)}):")
        for query, expected, actual in failures:
            print(f"  expected={expected:<15} got={actual:<15} | {query}")

    # Exit with non-zero code if accuracy is below 70 %
    if pct < 70:
        print("\nWARNING: accuracy below 70% threshold.")
        sys.exit(1)


if __name__ == "__main__":
    main()
