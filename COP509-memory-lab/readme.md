# COP509 Memory-Efficient Model Loading

## Exercise Sheet

### About This Lab

This lab teaches you 4 techniques to reduce memory usage when loading large PyTorch models. You'll measure real CPU and GPU memory on a Google Colab T4 GPU and learn how to avoid the 2× memory overhead that occurs with PyTorch's default loading approach.

### Files

- `exercise_solution.ipynb` — Main notebook with tasks and explanations
- `_infrastructure.py` — Helper module for memory measurement
- `ExerciseSheet.pdf` — Full lab description and instructions

### References

- [Original notebook by Sebastian Raschka](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/08_memory_efficient_weight_loading/)
- [PyTorch torch.load() documentation](https://pytorch.org/docs/stable/generated/torch.load.html)
- [PyTorch Meta Tensors](https://pytorch.org/docs/stable/meta.html)
