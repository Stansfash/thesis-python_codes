# Chapter 3: Model Comparison and Optimisation

Comparative evaluation of ML models and Scipy differential evolution for concentrate allocation.

## Files
- `Chapter_3_modelling.ipynb` - Model comparison notebook
- `Chapter_3_Optimisation_code.py` - Differential evolution optimisation script

## Models Compared
- Random Forest (with grid search)
- Neural Networks (Adam, SGD)
- LSTM
- Gaussian Process

## Optimisation
- Algorithm: Scipy differential evolution
- Constraints: Linear constraints for daily budget
- Dynamic bounds based on previous day
- 91-day optimisation period

## Usage
```bash
jupyter notebook Chapter_3_modelling.ipynb
python Chapter_3_Optimisation_code.py
```
