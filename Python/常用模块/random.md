

_random.randint()_





</br>

_random.random()_





</br>

_random.choice()_

```python
# 从 arr 中随机返回一个
random.choice([1,2,3])

# 依据概率返回一个 element
elements = ['p1', 'p2', 'p3']
probabilities = [0.2, 0.5, 0.3]
random.choices(elements, weights=probabilities, k=1)
```