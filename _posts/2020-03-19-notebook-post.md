---
layout: post
title: notebook post
image: /img/hello_world.jpeg
---

# Lambda School Data Science Module 123

## Introduction to Bayesian Inference




## Assignment - Code it up!

We used pure math to apply Bayes Theorem to drug tests. Now write Python code to reproduce the results! This is purposefully open ended - you'll have to think about how you should represent probabilities and events. You can and should look things up.

Specific goals/targets:

### 1) Write a function 

`def prob_drunk_given_positive(prob_drunk_prior, false_positive_rate):` 

You should only truly need these two values in order to apply Bayes Theorem. In this example, imagine that individuals are taking a breathalyzer test with an 8% false positive rate, a 100% true positive rate, and that our prior belief about drunk driving in the population is 1/1000. 
 - What is the probability that a person is drunk after one positive breathalyzer test?
 - What is the probability that a person is drunk after two positive breathalyzer tests?
 - How many positive breathalyzer tests are needed in order to have a probability that's greater than 95% that a person is drunk beyond the legal limit?

### 2) Explore `scipy.stats.bayes_mvs`  
Read its documentation, and experiment with it on data you've tested in other ways earlier this week.
 - Create a visualization comparing the results of a Bayesian approach to a traditional/frequentist approach. (with a large sample size they should look close to identical, however, take this opportunity to practice visualizing condfidence intervals in general. The following are some potential ways that you could visualize confidence intervals on your graph:
  - [Matplotlib Error Bars](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.errorbar.html)
  - [Seaborn barplot with error bars](https://seaborn.pydata.org/generated/seaborn.barplot.html)
  - [Vertical ines to show bounds of confidence interval](https://www.simplypsychology.org/confidence-interval.jpg)
  - [Confidence Intervals on Box Plots](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.boxplot.html)

### 3) In your own words, summarize the difference between Bayesian and Frequentist statistics

If you're unsure where to start, check out [this blog post of Bayes theorem with Python](https://dataconomy.com/2015/02/introduction-to-bayes-theorem-with-python/).




```python
# Write a function for Bayesian DUI tests

def prob_drunk_given_positive(prob_drunk_prior, false_positive_rate, true_positive_rate):
  numerator = (true_positive_rate * prob_drunk_prior)
  denominator = numerator + (false_positive_rate*(1-prob_drunk_prior))
  p = numerator / denominator
  return p
```


```python
# prob drunk after 1 positive test
p_value = prob_drunk_given_positive(.001, .08, 1)
p_value

```




    0.012357884330202669




```python
# prob drunk after 2 positive tests
prob_drunk_given_positive(p_value, .08, 1)
```




    0.13525210993291495




```python
# how many pos tests for > 95% prob drunk

count = 0
p_value1 = .001
while p_value1 < .95:
  p_value2 = prob_drunk_given_positive(p_value1, .08, 1)
  p_value1 = p_value1 + p_value2
  count = count + 1
print(count)


```

    4
    


```python
# Explore scipy.stats.bayes_mvs

# read/format data
import pandas as pd
from scipy import stats
import numpy as np

votes = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data', names=['party','handicapped-infants','water-project',
                          'budget','physician-fee-freeze', 'el-salvador-aid',
                          'religious-groups','anti-satellite-ban',
                          'aid-to-contras','mx-missile','immigration',
                          'synfuels', 'education', 'right-to-sue','crime','duty-free',
                          'south-africa'])

votes = votes.replace({'y':1, 'n':0, '?':np.NaN})
rep = votes[votes['party'] == 'republican']
rep = rep.drop(['party'], axis=1)
rep_infants = rep['handicapped-infants']
rep_infants = rep_infants.dropna()
rep_infants.head()


```




    0     0.0
    1     0.0
    7     0.0
    8     0.0
    10    0.0
    Name: handicapped-infants, dtype: float64




```python
# visualize bayesian approach
import matplotlib.pyplot as plt
mean_CI, _, _ = stats.bayes_mvs(rep_infants, alpha=.95)
rep_infants.plot.density()
plt.axvline(x=mean_CI.minmax[1], color='red')
plt.axvline(x=mean_CI.statistic, color='black')
plt.axvline(x=mean_CI.minmax[0], color='red')
plt.show()
```


![png](/img/output_7_0.png)



```python
# create confidence interval function

def confidence_interval (data, confidence = 0.95):
  data = np.array(data)
  mean = np.mean(data)
  n = len(data)
  s = data.std(ddof=1)
  stderr = s / np.sqrt(n)
  t = stats.t.ppf((1 + confidence) / 2.0, n - 1)
  margin_of_error = t*stderr
  return (mean, mean - margin_of_error, mean + margin_of_error)
  
```


```python
# Generate confidence interval for Republicans' votes about infants
mean, upper, lower = confidence_interval(rep_infants, confidence=.95)
print(mean)
print(upper)
print(lower)
```

    0.18787878787878787
    0.12765166444807918
    0.24810591130949655
    


```python
# visualize traditional approach 

ci = confidence_interval(rep_infants)
rep_infants.plot.density()
plt.axvline(x=ci[2], color='red')
plt.axvline(x=ci[0], color='black')
plt.axvline(x=ci[1], color='red')
plt.show()
```


![png](/img/output_10_0.png)


**In your own words, summarize the difference between Bayesian and Frequentist statistics**:

Beysian statistics use data from prior experiments, and frequentists do not.

## Resources

- [Worked example of Bayes rule calculation](https://en.wikipedia.org/wiki/Bayes'_theorem#Examples) (helpful as it fully breaks out the denominator)
- [Source code for mvsdist in scipy](https://github.com/scipy/scipy/blob/90534919e139d2a81c24bf08341734ff41a3db12/scipy/stats/morestats.py#L139)

## Stretch Goals:

- Go back and study the content from Modules 1 & 2 to make sure that you're really comfortable with them.
- Apply a Bayesian technique to a problem you previously worked (in an assignment or project work) on from a frequentist (standard) perspective
- Check out [PyMC3](https://docs.pymc.io/) (note this goes beyond hypothesis tests into modeling) - read the guides and work through some examples
- Take PyMC3 further - see if you can build something with it!


```python
# attempting to simulate the monty hall problem
vars = [car,goat1,goat2]
car = 1
goat1 = 2
goat2 = 3
wins = 0
losses = 0

count = 0
import random

while count < 1000:
  sample = random.sample(vars,3)
  if sample[0] == 1:
    if count % 2 != 0:
      wins = wins + 1
    if count % 2 == 0:
        if sample[1] == 1:
          wins = wins + 1
        else:
          losses = losses + 1
  count = count + 1
print('Contestant wins ' + str((wins / count)*100) + '% of the time with switching.')


  

```

    Contestant wins 18.2% of the time with switching.
    


```python
sample = random.sample(vars,3)
print(sample)
print(sample[0])
```

    [2, 1, 3]
    2
    


```python

```
