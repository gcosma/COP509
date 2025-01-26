#Plagiarism Detection

# Short Answer Plagiarism Detection Dataset

This repository contains a corpus of short answer texts designed for evaluating plagiarism detection systems, originally created by Clough & Stevenson (2011).

## Dataset Overview

### Learning Tasks

A. What is inheritance in object oriented programming?
B. Explain the PageRank algorithm that is used by the Google search engine.
C. Explain the Vector Space Model that is used for Information Retrieval.
D. Explain Bayes Theorem from probability theory.
E. What is dynamic programming?

Each task was designed to:
- Represent different areas of Computer Science
- Be answerable in 200-300 words
- Not require specialised knowledge
- Have relevant Wikipedia articles as source material

## Dataset Overview

- 100 documents total (95 student answers + 5 Wikipedia source articles)
- 5 Computer Science topics with short answer questions
- 4 categories of answers:
  - Near copy (direct copying)
  - Light revision (minor paraphrasing)
  - Heavy revision (significant rewriting)
  - Non-plagiarized (independent answers)
- Average text length: 208 words
- Created by 19 participants (62% native English speakers)

## File Structure

```
File Structure
data/
├── wikipedia/     # Original Wikipedia source articles
├── answers/       # Student answers organized by task
└── metadata.csv   # Answer metadata and labels

## Dataset Statistics

- Total word count: 19,559
- Unique tokens: 22,230
- Files per task: 19 answers each
- Distribution:
  - Near copy: 19 files
  - Light revision: 19 files  
  - Heavy revision: 19 files
  - Non-plagiarized: 38 files

## Usage Examples

The dataset can be used to:
- Train/evaluate plagiarism detection models
- Study paraphrasing patterns
- Analyse differences between native/non-native writers
- Research text similarity measures

## Citation

If you use this dataset, please cite:
```
@article{clough2011developing,
  title={Developing a corpus of plagiarised short answers},
  author={Clough, Paul and Stevenson, Mark},
  journal={Language Resources and Evaluation},
  volume={45},
  number={1},
  pages={5--24},
  year={2011},
  publisher={Springer}
}
```

## License

The dataset is freely available for research purposes. See the original paper for terms of use.

## References

Clough, P., & Stevenson, M. (2011). Developing a corpus of plagiarised short answers. Language Resources and Evaluation, 45(1), 5-24.
