# RSIM: Variance Reduction Simulation Tools in Python

RSIM is a collection of advanced statistical simulation tools designed to reduce variance in Monte Carlo simulations.  
Based on concepts from Sheldon M. Ross’s book *Simulation*, RSIM is ideal for researchers, students, and practitioners in statistics and data science.

---

## 🧠 Key Concepts (Classes and Usage)

### ▶️ `StratifiedSampling`
> Implements **stratified sampling** to reduce variance by dividing the input space into strata and sampling each separately.

<details>
  <summary>Learn more + Example</summary>

This class partitions the data into several strata and samples from each, producing a more accurate estimate of the expected value.

**Example:**
```python
from RSIM.stratified_sampling import StratifiedSampling

def function(x):
    return x**2

ss = StratifiedSampling(function=function, n_samples=1000, strata=10)
estimate = ss.estimate()
print("Stratified Estimate:", estimate)
```
</details>

---

### ▶️ `AntitheticSampling`
> Uses **antithetic variables** to reduce variance by pairing each sample with its antithetic counterpart.

<details>
  <summary>Learn more + Example</summary>

For every random sample \( U \), its antithetic variable \( 1 - U \) is used to smooth out fluctuations and improve estimate accuracy.

**Example:**
```python
from RSIM.antithetic_sampling import AntitheticSampling

def function(x):
    return x**2

asamp = AntitheticSampling(function=function, n_samples=1000)
estimate = asamp.estimate()
print("Antithetic Estimate:", estimate)
```
</details>

---

### ▶️ `ControlVariate`
> Applies **control variates** technique by using correlated auxiliary variables to reduce estimation variance.

<details>
  <summary>Learn more + Example</summary>

If an auxiliary variable with a known mean is correlated with the target variable, it can be used to improve the precision of estimates.

**Example:**
```python
from RSIM.control_variate import ControlVariate

def f(x):
    return x**2

def g(x):
    return x  # control variate

cv = ControlVariate(f, g, mu_g=0.5, n_samples=1000)
estimate = cv.estimate()
print("Control Variate Estimate:", estimate)
```
</details>

---

### ▶️ `ImportanceSampling`
> Uses **importance sampling** to reduce variance by sampling from a different distribution that focuses on important regions.

<details>
  <summary>Learn more + Example</summary>

By sampling from a carefully chosen distribution, estimates become more accurate and have lower variance.

**Example:**
```python
from RSIM.importance_sampling import ImportanceSampling
import numpy as np

def f(x):
    return np.exp(-x)

def g(x):
    return np.exp(-x)  # sampling pdf

isamp = ImportanceSampling(f=f, g=g, sampler=lambda: np.random.exponential(), n_samples=1000)
estimate = isamp.estimate()
print("Importance Sampling Estimate:", estimate)
```
</details>

---

## 🛠 Installation

```bash
pip install RSIM
```

---

## 📚 References

The main theoretical foundation for RSIM is Sheldon M. Ross’s book *Simulation* (Academic Press, 2022).

---

## 💬 Contributions

Contributions are welcome! Feel free to submit pull requests.

---

If you need help adding this README to your GitHub repository or pushing changes, just ask!
