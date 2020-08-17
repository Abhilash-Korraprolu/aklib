This is a project I'm working on. It will consist of the classes and functions to streamline ML workflow. The goal is to eventually make this a python library.

It currently has the following features:

1. **Preprocessing** consists of:
(i) A power transformer called unskew() that reduces skewedness of the features using Box-Cox Transformation.

To Do:
(i) Incorporate Tukey's ladder in unskew().
(ii) Cross-validated target mean encoder.


2. **QuickPlots** allow for the creation beautiful visualisations in one line. It currently consists of:
(i) Count plot
(ii) Pie chart
(iii) Histogram
(iv) Heatmap
(v) Boxplot
(vi) Violin plot
(vii) Relplot
(viii) Bar plot

<b>Example 1: Bar Plot</b>
![Screenshot 2020-08-17 at 6 59 25 PM](https://user-images.githubusercontent.com/42868745/90401623-d7aa7480-e0bb-11ea-9526-32aa1545a8a1.png)

<b>Example 2: Histogram</b>
![Screenshot 2020-08-17 at 6 57 57 PM](https://user-images.githubusercontent.com/42868745/90401706-f6a90680-e0bb-11ea-9427-3d2252f939e2.png)
These examples are based on the dataset: https://archive.ics.uci.edu/ml/datasets/student+performance
