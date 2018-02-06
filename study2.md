---
title: "Head Start for Data Scientist"
output:
  html_document:
    number_sections: TRUE
    toc: TRUE
    fig_height: 4
    fig_width: 7
    code_folding: show
---


```r
#--显示R代码  knitr包的opts_chunk$set()函数可以配置 隐藏或者显示代码 
knitr::opts_chunk$set(echo=TRUE)
```


####早期，当我是一个机器学习的新手<br>
####我曾经不知所措，比如选择语言进行编码，选择正确的在线课程，或选择正确的算法。<br>
####所以，我打算让人们更容易进入机器学习。<br>
####我会假设我们中的很多人都是从机器学习的旅程开始的。 让我们来看看当前的专业人员如何达到目标，以及如何模仿他们。<br>


####第一阶段  保证独立完成<br>
####对于刚开始使用机器学习的人来说，把自己和学习，教授和练习机器学习的人们联系在一起非常重要。
####独自学习是不容易的。 所以，保证要自己去学习机器学习，可以找到数据科学论坛社区来帮助你减少困难。


####第二阶段  学习生态系统
####发现机器学习的生态系统
####数据科学是一个充分利用开源平台的领域。 虽然数据分析可以用多种语言进行，但使用正确的工具可以制定或中断项目。
####数据科学图书馆在Python和R生态系统中蓬勃发展。 在这个网址查看 用Python还是R进行数据分析。（https://www.datacamp.com/community/tutorials/r-or-python-for-data-analysis）
####无论您选择哪种语言，Jupyter Notebook和RStudio都让我们的生活变得更加轻松。 它们允许我们在操纵数据的同时可视化数据。 按照这个[链接]阅读更多关于Jupyter Notebook的功能。（http://blog.kaggle.com/2015/12/07/three-things-i-love-about-jupyter-notebooks/）
####Kaggle，Analytics Vidhya，MachineLearningMastery和KDNuggets是一些活跃的社区，世界各地的数据科学家在这里丰富彼此的学习。
####机器学习已经通过在线课程或者Coursera，EdX等的MOOC进行民主化，在这里我们向世界一流大学的杰出教授学习。 这里有一个[顶级MOOC列表]有关于现在可用的数据科学列表（https://medium.freecodecamp.org/i-ranked-all-the-best-data-science-intro-courses-based-on-thousands-of-data-points- db5dc7e3eb8e）。


#--第三阶段  巩固基础
#--学习操纵数据
#--根据访谈和专家估计，数据科学家将50％到80％的时间用于收集和准备不规则数字数据的世俗工作，然后才能探索有用的金块。 - 纽约时报的史蒂夫·洛尔
#--数据科学不仅仅是建立机器学习模型。 这也是解释模型并用它们来推动数据驱动的决策。 在从分析到数据驱动的结果的过程中，数据可视化以强有力和可信的方式呈现数据，扮演着非常重要的角色。
#--Python中的[Matplotlib]（https://matplotlib.org/）库或R中的[ggplot]（http://ggplot2.org/）提供了完整的2D图形支持，具有非常高的灵活性，可以创建高质量的数据可视化。
#--有一些图书馆您将花费大部分时间在上面当您在进行分析时。


#--第四阶段  日复一日的练习
#--学习机器学习算法并且每天练习
#--有了基础之后，你可以实现机器学习算法来预测和做一些很酷的东西
#--Python中的Scikit-learn库或R中的caret, e1071库通过一致的接口提供一系列有监督和无监督的学习算法。
#--这些让你实现一个算法，而不用担心内部工作或细节的细节。
#--将这些机器学习算法应用到您身边的用例中。这可能是在你的工作，或者你可以在Kaggle比赛中练习。 其中，全世界的数据科学家都在竞相建立模型来解决问题。
#--同时了解一种算法的内部运作情况。 从机器学习的“Hello World！”开始，线性回归然后转到Logistic回归，决策树到支持向量机。 这将需要你刷新你的统计和线性代数。
#--Coursera创始人Andrew Ng是AI的先驱，开发了一个[机器学习课程]（https://www.coursera.org/learn/machine-learning），为您理解机器学习算法的内部工作提供了一个很好的起点。


#--第五阶段  学习高级技能
#--了解复杂的机器学习算法和深度学习架构
#--虽然机器学习作为一个领域早已建立起来，但最近的炒作和媒体关注主要是由于机器学习在计算机视觉，语音识别，语言处理等AI领域的应用。 其中许多是由Google，Facebook，微软等科技巨头开创的。
#--这些最新的进展可以归功于廉价计算的进步，大规模数据的可用性以及深度学习架构的发展。
#--要深入学习，您需要学习如何处理非结构化数据 - 无论是文本，图像，
#--您将学习使用像TensorFlow或Torch这样的平台，这使我们可以应用深度学习，而不用担心低级别的硬件要求。 你将学习强化学习，这使得像AlphaGo Zero这样的现代AI奇迹成为可能


 
#--我在Kaggle看到许多新的学习者，想为他们创造一个kernal，让他们有一个良好的开端。
#--这个针对基础学习者的kernal，是试图快速理解数据科学，我选择了定期的对话方式。
#--在kernal中，我们将遇到两个字符“MARK”和“JAMES”，其中MARK是Data Science（Laymen）的新成员，JAMES使他理解概念
#为了容易的开始 我选择了泰坦尼克号的数据集


#--数据集简介
On 14 April 1912, the [RMS Titanic](https://en.wikipedia.org/wiki/RMS_Titanic) struck a large iceberg and took approximately 1,500 of its passengers and crew below the icy depths of the Atlantic Ocean. Considered one of the worst peacetime disasters at sea, this tragic event led to the creation of numerous [safety regulations and policies](http://www.gc.noaa.gov/gcil_titanic-history.html) to prevent such a catastrophe from happening again. Some critics, however, argue that circumstances other than luck resulted in a disproportionate number of deaths. The purpose of this analysis is to explore factors that influenced a person’s likelihood to survive.
#--上面的英文大概就是讲了一下泰坦尼克号的故事吧~


#--选择软件 R
The following analysis was conducted in the [R software environment for statistical computing](https://www.r-project.org/).


#---下面的讲解主要以对话的形式进行
问：今天学啥？
答：数据科学基础
 
问：啥是数据科学？
答：数据科学是数据推理，算法开发和技术的多学科融合，用来解决分析复杂的问题。
                      
问： 数据科学家如何进行数据挖掘
答： 用这些开始
	 1.收集--解决问题所需的原始数据。
	 2.加工--数据缠绕
	 3.探索--数据可视化
	 4.展示--深入分析（机器学习，统计模型，算法）
	 5.交流--分析的结果

问：能不能详写解释一下
答：不能


#--导入数据集                                     
问：怎么把数据集插入到Rstudio
答：自己百度

问：万一调用库失败咋办
答：自己查

问：我要开始了好激动
答：赶紧的吧 真墨迹

```
{r dependencies, message = FALSE, warning = FALSE}

# data wrangling  数据处理包
library(tidyverse)
library(forcats)
library(stringr)
library(caTools)

# data assessment/visualizations  数据可视化包
library(DT)
library(data.table)
library(pander)
library(ggplot2)
library(scales)
library(grid)
library(gridExtra)
library(corrplot)
library(VIM) 
library(knitr)
library(vcd)
library(caret)

# model  模型包
library(xgboost)
library(MLmetrics)
library('randomForest') 
library('rpart')
library('rpart.plot')
library('car')
library('e1071')
library(vcd)
library(ROCR)
library(pROC)
library(VIM)
library(glmnet) 
```

问：现在可以导入数据集了吧
答：嗯 

```
{r, message=FALSE, warning=FALSE, results='hide'}

train <- read_csv('../input/train.csv')
test  <- read_csv('../input/test.csv')
```

                                        
#--为了研究完整的数据集，可以加入测试和训练数据集。
#--在此之前，我们将添加一个新的列“set”，并为测试数据集命名为“test”
#--和“训练”列车数据集，以了解它是哪条记录。

```
{r , message=FALSE, warning=FALSE, results='hide'}

train$set <- "train"
test$set  <- "test"
test$Survived <- NA
full <- rbind(train, test)
```


问：我们已经处理了用于解决问题的原始数据
答：嗯 接着处理
                                     
问：为什么我们需要处理数据（数据缠绕）
答：你收集的数据目前仍是原始数据，这很可能包含错误，缺失和腐败的价值。
在您从数据中得出任何结论之前，您需要对其进行一些数据处理，
这是我们下一节的主题。我们选择我们想要操作的数据
                                                                        
问：在数据科学下进行了哪些操作？
答：这是所有的观点
	这个给了一些清晰的观点 自己看图
<center><img src="https://doubleclix.files.wordpress.com/2012/12/data-science-02.jpg"></center>
<center><img src="https://cdn-images-1.medium.com/max/1600/1*2T5rbjOBGVFdSvtlhCqlNg.png"></center>
                                                                          
问：这看起来不错，把工具和语言都显示出来了
答：在这里我们将使用R语言做处理                                                                                  

                                                                                                                         
#--在此之前，我们需要一个数据检擦 - 探索性分析（EDA）
1.检查数据集的疏散
2.数据集的维度
3.列名
4.每行有多少不同的值
5.缺失值

```
{r , message=FALSE, warning=FALSE, results='hide'}

# check data 检查数据
str(full)

# dataset dimensions  维度信息
dim(full)

# Unique values per column  每列有多少种值
lapply(full, function(x) length(unique(x))) 

#Check for Missing values 检查缺失值

library(tidyverse)
library(forcats)
library(stringr)
library(caTools)
missing_values <- full %>% summarize_all(funs(sum(is.na(.))/n()))    #缺失值比例  funs 函数列表  summarize_all将函数应用于每一列
missing_values <- gather(missing_values, key="feature", value="missing_pct") #gather 转化成key-value形式 
missing_values %>%
  ggplot(aes(x=reorder(feature,-missing_pct),y=missing_pct)) +   #reorder 重新排序  默认的将第一个参数作为分类变量处理，将第二个变量重新排序 就是先分类 再排序
  geom_bar(stat="identity",fill="red")+     #geom_bar 画条形图
  coord_flip()+theme_bw()    #coord_flip()  旋转横纵坐标  横坐标变为纵坐标 纵变横   theme_bw() 添加一个坐标的主题样式

<img src="/img/1.png"></img>

#对缺失值有用的数据质量函数
#检查 一个数据集df的某一列colname的类型  返回列的属性列表(列名，类型，总个数，缺失值个数，numInfinite，平均值，最小值，最大值)
checkColumn = function(df,colname){

  testData = df[[colname]]
  numMissing = max(sum(is.na(testData)|is.nan(testData)|testData==''),0)  #最大缺失值

  #class(x)得到x的类型
  if (class(testData) == 'numeric' | class(testData) == 'Date' | class(testData) == 'difftime' | class(testData) == 'integer'){
    list('col' = colname,'class' = class(testData), 'num' = length(testData) - numMissing, 'numMissing' = numMissing, 'numInfinite' = sum(is.infinite(testData)), 'avgVal' = mean(testData,na.rm=TRUE), 'minVal' = round(min(testData,na.rm = TRUE)), 'maxVal' = round(max(testData,na.rm = TRUE)))
  } else{
    list('col' = colname,'class' = class(testData), 'num' = length(testData) - numMissing, 'numMissing' = numMissing, 'numInfinite' = NA,  'avgVal' = NA, 'minVal' = NA, 'maxVal' = NA)
  }
}

#检查数据集的所有列属性  得到所有列属性列表的数据框
checkAllCols = function(df){
  resDF = data.frame()
  for (colName in names(df)){
    resDF = rbind(resDF,as.data.frame(checkColumn(df=df,colname=colName)))
  }
  resDF
}

#属性列表可视化
```
{r , message=FALSE, warning=FALSE, results='hide'}
library(DT)
library(data.table)
library(pander)
library(ggplot2)
library(scales)
library(grid)
library(gridExtra)
library(corrplot)
library(VIM) 
library(knitr)
library(vcd)
library(caret)
datatable(checkAllCols(full), style="bootstrap", class="table-condensed", options = list(dom = 'tp',scrollX = TRUE))

<img src="/img/1-1.png"></img>

#map_dbl返回一个与输入长度相同的向量。
#round(x, n) x的约数 精确到n位
miss_pct <- map_dbl(full, function(x) { round((sum(is.na(x)) / length(x)) * 100, 1) })  #缺失值比率
miss_pct <- miss_pct[miss_pct > 0]

data.frame(miss=miss_pct, var=names(miss_pct), row.names=NULL) %>%
    ggplot(aes(x=reorder(var, -miss), y=miss)) + 
    geom_bar(stat='identity', fill='red') +
    labs(x='', y='% missing', title='Percent missing data by feature') +
    theme(axis.text.x=element_text(angle=90, hjust=1))
<img src="/img/2.png"></img>
```


## Feature engineering.
问：什么是feature engineering
答：该过程试图从数据中现有的原始特征创建额外的相关特征，并提高学习算法的预测能力。详情查看上面这个网址 https://github.com/bobbbbbi/Machine-learning-Feature-engineering-techniques


#--数据操作                                                                                  
问：我们已经了解了我们的数据集吧
答：然后我们还需要做一些数据的处理
                                    
问：怎么进行数据处理
答：数据操作是一个改变数据的过程，为了使其更容易阅读并且更有组织性
                                                                                  
#--下面的部分着重于准备数据，以便用于学习训练，比如探索性数据分析和建模拟合。

### 处理Age字段
#-- 对于年龄的处理   将缺失值替换为平均值

```
{r age, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}

#-- mutate() 添加新的变量并保留现有的
full <- full %>%
    mutate(
      Age = ifelse(is.na(Age), mean(full$Age, na.rm=TRUE), Age),       #如果为空 就赋平均值
      `Age Group` = case_when(   Age < 13             ~ "Age.0012",    #case when 分组
                                 Age >= 13 & Age < 18 ~ "Age.1317",
                                 Age >= 18 & Age < 60 ~ "Age.1859",
                                 Age >= 60            ~ "Age.60Ov"))

```

### 处理Embarked字段
#--使用常见的符号（感觉是出现次数最多的 即S）来替换 Embarked 的空值
```
{r pp_embarked, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}
full$Embarked <- replace(full$Embarked, which(is.na(full$Embarked)), 'S')
```

###处理 Titles字段
#--从Name特征中提取个人标题
```
{r pp_titles, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}

names <- full$Name
title <-  gsub("^.*, (.*?)\\..*$", "\\1", names)
full$title <- title
table(title)
#--names---------------------------------------------------
#--[1297] Nourney, Mr. Alfred (Baron von Drachstedt")"     
#--[1298] Ware, Mr. William Jeffery             
