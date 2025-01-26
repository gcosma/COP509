# Short Answer Plagiarism Detection Dataset

This repository contains a corpus of short answer texts designed for evaluating plagiarism detection systems, originally created by Clough & Stevenson (2011). Please refer to this paper https://link.springer.com/article/10.1007/s10579-009-9112-1 

## Dataset Overview

### Learning Tasks

1. What is inheritance in object oriented programming?
2. Explain the PageRank algorithm that is used by the Google search engine.
3. Explain the Vector Space Model that is used for Information Retrieval.
4. Explain Bayes Theorem from probability theory.
5. What is dynamic programming?

Each task was designed to:
- Represent different areas of Computer Science
- Be answerable in 200-300 words
- Not require specialised knowledge
- Have relevant Wikipedia articles as source material

## Dataset Overview

- 100 documents total (95 student answers + 5 Wikipedia source articles)
- 5 Computer Science topics with short answer questions
- 4 categories of answers (see below for more details):
  - Near copy (direct copying) [labelled as cut]
  - Light revision (minor paraphrasing) [labelled as light]
  - Heavy revision (significant rewriting) [labelled as heavy]
  - Non-plagiarised (independent answers) [labelled as non]
- Average text length: 208 words
- Created by 19 participants (62% native English speakers)

## Answer Categories

### Near Copy (Cut)
- Answer provided by directly copying from Wikipedia article
- Students selected relevant text within 200-300 word limit
- No modifications to source text required

### Light Revision (light)
- Based on Wikipedia text with minor modifications allowed 
- Could substitute words/phrases with synonyms
- Basic grammatical changes and paraphrasing permitted
- Required to maintain original information order

### Heavy Revision (heavy)
- Based on Wikipedia but significantly rewritten
- Freedom to split or combine sentences
- No restrictions on restructuring while keeping meaning
- Could modify text organization extensively

### Non-Plagiarism (non)
- Used provided learning materials (lecture notes/textbooks)
- Written based on understanding, not copying
- Could reference any materials except Wikipedia
- Demonstrated acquired knowledge rather than text reuse
## Data Format

Each answer is labeled with:
- Task ID (A-E)
- Plagiarism category 
- Native/non-native speaker status
- Level of task difficulty (1-5)
- Student's knowledge level (1-5)

## File Structure

```
data/
├── wikipedia/     # Original Wikipedia source articles
├── answers/       # Student answers organized by task
└── metadata.csv   # Answer metadata and labels
```

## Dataset Statistics

- Total word count: 19,559
- Unique tokens: 22,230
- Files per task: 19 answers each
- Distribution:
  - Near copy: 19 files
  - Light revision: 19 files  
  - Heavy revision: 19 files
  - Non-plagiarised: 38 files

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
