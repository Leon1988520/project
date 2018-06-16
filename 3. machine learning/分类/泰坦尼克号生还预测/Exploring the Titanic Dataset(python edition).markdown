---
title: Exploring the Titanic Dataset(Python edition)
tags: titanic,python,data explore
grammar_cjkRuby: true
---

# 简介
> 这是本人在kaggle的第一次尝试，参考原文请点击[这里][1]，原文采用R语言进行数据分析。在原文的思想指导下，本文利用Python实现了数据分析过程。在此，对原文作者表示感谢，原文数据分析思路非常清晰且完备，是kaggle入门乃至数据分析入门非常好的案例。

本文主要由以下三部分组成：
+ 特征工程
+ 缺失值处理
+ 预测

## 1 数据加载与检视
```
# load data and check data
data_test = pd.read_csv("test.csv")
data_test_survived = pd.read_csv("gender_submission.csv")
data_train = pd.read_csv("train.csv")

df = pd.concat([data_test, data_train], axis = 0) # concat the test and train sample
print(df.head(2).T) # check the data， look at the detail of the first two samples
```
```
                            0                                 1
Age                      34.5                                47
Cabin                     NaN                               NaN
Embarked                    Q                                 S
Fare                   7.8292                                 7
Name         Kelly, Mr. James  Wilkes, Mrs. James (Ellen Needs)
Parch                       0                                 0
PassengerId               892                               893
Pclass                      3                                 3
Sex                      male                            female
SibSp                       0                                 1
Survived                  NaN                               NaN
Ticket                 330911                            363272
```

> 通过对数据的观察，我们可以了解到变量的个数及其数据类型

# 2 特征工程
## 2.1 名字背后的秘密
名字在分析中，容易被数据分析师忽视，在英文名字中，一般含有title等具有职位、家族标签的字段。对名字进行进一步的分解，可以得到更具有意义的变量，这些变量可以直接应用于模型或者构造新的变量。如title可表示一个人的职业和社会地位，surname可以表示一个人的家族属性。因此，对名字进行特征提取：

```
pattern = re.compile(r".+\,\s(.+?)\.") # ? is the indicator of lazy or not 

def get_title(string, pattern = pattern):
    return pattern.findall(string)[0]

df["Title"] = df["Name"].apply(get_title)
title_stat = df["Title"].value_counts()
title_stat_sex = df.groupby(["Title", "Sex"]).size()
```
```
Mr              757
Miss            260
Mrs             197
Master           61
Dr                8
Rev               8
Col               4
Mlle              2
Ms                2
Major             2
Jonkheer          1
the Countess      1
Dona              1
Don               1
Lady              1
Capt              1
Mme               1
Sir               1
```


## 2.2 家庭成员数与是否获救
上面，我们已经利用乘客姓名，构造出了一些新的变量。接下来，我们将构造一些和家庭相关的变量，例如根据parch以及sibsp构造出家庭的成员数。

```
df["Family_size"] = df["Parch"] + df["SibSp"] + 1 # create a new variable from parch and sibsp
df["Family_size"].hist()
family_stat = (df[df["Survived"] != np.nan].groupby(["Family_size", "Survived"]).size()).unstack()
family_stat.plot(kind = "bar")
```
![image](https://github.com/xeebudong/notebooks/blob/master/kaggle/titanic/1.%E5%AE%B6%E5%BA%AD%E4%BA%BA%E6%95%B0%E4%B8%8E%E8%8E%B7%E6%95%91.png?raw=true)
从图中可以看到，单身狗和家眷数量大于3个的乘客，生存概率小于50%， 而随行家眷在1到3个的乘客，有超过1半的乘客都生存下来了。

## 2.3 生成更多的变量
对乘客Title和家庭成员数量进行分析之后，还能再基于当前变量，衍生出什么变量呢？我们发现Passenger Cabin(座舱)包含了deck(甲板)信息。
```
## Deck
print(df[df["Cabin"].isnull() != True]["Cabin"].head())
```
```
12                B45
14                E31
24    B57 B59 B63 B66
26                B36
28                A21
Name: Cabin, dtype: object
```
该字段有大量空值，第一个字母，如A表示该座舱位于的甲板位置(可以看到，有一些座舱，如24，占了几个房间)，这里以字段的第一个字母表示座舱所在的甲板位置。(还可以刻画一个字段，即座舱占据的房间数量，以观察大房间和小房间的用户存活率是否有差异)

```
print(df[df["Cabin"].isnull() != True]["Cabin"].head())
df["Deck"] = df["Cabin"].apply(lambda x: str(x)[0] if(len(str(x)) > 0) else None)
```

# 3. 缺失值处理
现在，我们准备对缺失值进行探索处理。有很多种方法可以解决缺失值问题。对于小数据量的问题，一般不倾向于使用删除整个样本或者变量(包含空值的)，而一般根据数据的分布情况，采用均值、中值、或者众数来替换空值，最后，将数据应用于模型。通常，我们会通过数据可视化来帮助我们决定采用哪种方法处理缺失值。

## 3.1 合理的数值估算
```
print(df[df["Embarked"].isnull()].T)
```
```
                            479                                        1247
Age                           38                                         62
Cabin                        B28                                        B28
Embarked                     NaN                                        NaN
Fare                          80                                         80
Name         Icard, Miss. Amelie  Stone, Mrs. George Nelson (Martha Evelyn)
Parch                          0                                          0
PassengerId                   62                                        830
Pclass                         1                                          1
Sex                       female                                     female
SibSp                          0                                          0
Survived                       1                                          1
Ticket                    113572                                     113572
Title                       Miss                                        Mrs
M_Title                     Miss                                        Mrs
Family_size                    1                                          1
Deck                           B                                          B
```

有两个用户Embarked(是否登船)属性为空，我们将依据现有数据对其进行推断。假设用户登船了，或许会产生消费数据，而消费的多少又和乘客的经济能力有关(可通过Pclass体现)。我们尝试利用Fare(消费)和Pclass(乘客级别)来推断Embarked。

![image](https://github.com/xeebudong/notebooks/blob/master/kaggle/titanic/2.%E5%90%84%E7%B1%BB%E7%94%A8%E6%88%B7%E4%BB%98%E8%B4%B9%E9%A2%9D%E5%BA%A6.png?raw=true)
可以看出，当Pclass=1，Embarked="C"的用户付费均值为80， 可以比较肯定的推断，这两个用户的Embarked应该为"C"

```
print(df[df.Fare.isnull()].T)
df.loc[df["Embarked"].isnull(), "Embarked"] = "C"
```
```
Age                        60.5
Cabin                       NaN
Embarked                      S
Fare                        NaN
Name         Storey, Mr. Thomas
Parch                         0
PassengerId                1044
Pclass                        3
Sex                        male
SibSp                         0
Survived                    NaN
Ticket                     3701
Title                        Mr
M_Title                      Mr
Family_size                   1
Deck                          n
```
同理，能找出有以为用户的Fare值为空，利用上述箱状图，可以直接赋Embarked=S，Pclass=3的用户的付费均值或者众数。另外一种方法，绘制Embarked=S，Pclass=3的用户的付费的概率密度函数。
```
print("The mode of Passengers whose Embarked is %s and Pclass is %d", ("S", 3)),
print(df[df.Fare.isnull() != True]["Fare"].mode()) # the mode is 8.05
df.loc[df["Fare"].isnull(), "Fare"] = 8.05 
df[df.Fare.isnull() != True]["Fare"].plot(kind = "kde")
df[df.Fare.isnull(), "Fare"] = df[df.Fare.isnull() != True]["Fare"].mode()
```
## 3.2 通过预测
我们看到Age这个字段有相当一部分缺失值。我们将依据其他变量，建立模型对年龄的缺失值进行预测。

```
print("The count of samples which missing age character:"), 
print(len(df[df.Age.isnull()]))
```

R中，有[mice包](http://www.xueqing.tv/cms/article/98)(https://zhuanlan.zhihu.com/p/21549898?refer=wiser)专门针对缺失值进行分析处理。原文中分析过程如下：

```
mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf') 
```
这里以'Pclass','Sex','Embarked', 'Title', 'Family_size'为自变量，建立随机森林模型对Age进行预测。经笔者查证，Python中没有类似R中mise包。下面利用sklearn的[RF回归模型](http://scikit-learn.org/stable/auto_examples/missing_values.html#sphx-glr-auto-examples-missing-values-py)，对Age进行预测。


## 3.3 第二轮特征工程

### 成年人&未成年人
根据年龄，可以建立一个新的变量，以体现乘客是成年人还是未成年人， 未成年人特征：Age≤18

### 母亲&子女
从上面，可以看出，孩子获救的概率比成年人高，是否在救起孩子的时候，也一并将其母亲救起呢？我们再建立一个变量来表征乘客是不是已为母亲。母亲的特征：1)Sex = "Female"；2)Age＞18；3)Parch>0；4)Title != "Miss"。

我们所关注的所有变量，都不应该包含缺失数据。经过特征工程，我们已经获得了一些能够帮助我们建立可靠的预测模型的新变量，也已经对所关注变量的缺失值进行了处理。

# 预测
最后，我们建立随机森林分类模型，来对乘客是否获救进行预测。

## 分割测试和训练集
```
d_train = df.ix[:len(data_train)]
d_test = df.ix[len(data_train):]
```

## 建立模型


## 变量重要性

## 预测

# 结论
本文介绍了从变量构成、特征工程、缺失值的处理到建模和预测的整个过程，R中有mice包专门应用于缺失值的处理，在Python中，可以针对R的方法，书写相应的缺失值处理函数。

# 参考文献

  [1]: https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic

  [用Pandas作图](http://cloga.info/python/2014/02/23/plotting_with_pandas)
  
  [随机森林算法](http://www.jianshu.com/p/c4bcb2505360)
  
  [随机森林算法-Python](https://segmentfault.com/a/1190000007463203)
  
  [在R中填充缺失数据—mice包](https://zhuanlan.zhihu.com/p/21549898?refer=wiser)
  
  [数据处理方法](https://sanwen8.cn/p/211w9wT.html)
  
  [Titanic案例分析2](https://www.kaggle.com/arthurtok/titanic/introduction-to-ensembling-stacking-in-python)

  [Titanic案例分析3](http://blog.csdn.net/han_xiaoyang/article/details/49797143)