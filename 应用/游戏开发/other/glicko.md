

Glicko 计算逻辑：https://zh.wikipedia.org/wiki/Glicko%E8%AF%84%E5%88%86%E7%B3%BB%E7%BB%9F

初始评分，DEFAULT_RATING = 1500
初始标准差，DEFAULT_RATING_DEVIATION = 350

（1）依据时间重新测算标准差

$$RD = min(\sqrt {RD_{old}^2 + c^2t}, 350)$$ 标准差

T 为自上次比赛至现在的时间长度（评分期）
R 为选手个人评分
S 比赛结果，为 1， 0， 1/2


（2）测算新评分

$$r = r_{old} + \frac{q}{\frac{1}{RD^2} + \frac{1}{d^2}}\sum_{i=1}^mg(RD_i)(s_i -E(s|r_0, r_i, RD_i))$$