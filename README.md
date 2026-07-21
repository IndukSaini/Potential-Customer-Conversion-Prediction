# Potential-Customer-Conversion-Prediction
End-to-end machine learning project that predicts which education leads are most likely to convert into paid customers and provides actionable sales and marketing recommendations.
ExtraaLearn Project
Context
The EdTech industry has been surging in the past decade immensely, and according to a forecast, the Online Education market would be worth $286.62bn by 2023 with a compound annual growth rate (CAGR) of 10.26% from 2018 to 2023. The modern era of online education has enforced a lot in its growth and expansion beyond any limit. Due to having many dominant features like ease of information sharing, personalized learning experience, transparency of assessment, etc, it is now preferable to traditional education.

In the present scenario due to the Covid-19, the online education sector has witnessed rapid growth and is attracting a lot of new customers. Due to this rapid growth, many new companies have emerged in this industry. With the availability and ease of use of digital marketing resources, companies can reach out to a wider audience with their offerings. The customers who show interest in these offerings are termed as leads. There are various sources of obtaining leads for Edtech companies, like

The customer interacts with the marketing front on social media or other online platforms.
The customer browses the website/app and downloads the brochure
The customer connects through emails for more information.
The company then nurtures these leads and tries to convert them to paid customers. For this, the representative from the organization connects with the lead on call or through email to share further details.

Objective
ExtraaLearn is an initial stage startup that offers programs on cutting-edge technologies to students and professionals to help them upskill/reskill. With a large number of leads being generated on a regular basis, one of the issues faced by ExtraaLearn is to identify which of the leads are more likely to convert so that they can allocate resources accordingly. You, as a data scientist at ExtraaLearn, have been provided the leads data to:

Analyze and build an ML model to help identify which leads are more likely to convert to paid customers,
Find the factors driving the lead conversion process
Create a profile of the leads which are likely to convert
Data Description
The data contains the different attributes of leads and their interaction details with ExtraaLearn. The detailed data dictionary is given below.

Data Dictionary

ID: ID of the lead

age: Age of the lead

current_occupation: Current occupation of the lead. Values include 'Professional','Unemployed',and 'Student'

first_interaction: How did the lead first interacted with ExtraaLearn. Values include 'Website', 'Mobile App'

profile_completed: What percentage of profile has been filled by the lead on the website/mobile app. Values include Low - (0-50%), Medium - (50-75%), High (75-100%)

website_visits: How many times has a lead visited the website

time_spent_on_website: Total time spent on the website

page_views_per_visit: Average number of pages on the website viewed during the visits.

last_activity: Last interaction between the lead and ExtraaLearn.

Email Activity: Seeking for details about program through email, Representative shared information with lead like brochure of program , etc
Phone Activity: Had a Phone Conversation with representative, Had conversation over SMS with representative, etc
Website Activity: Interacted on live chat with representative, Updated profile on website, etc
print_media_type1: Flag indicating whether the lead had seen the ad of ExtraaLearn in the Newspaper.

print_media_type2: Flag indicating whether the lead had seen the ad of ExtraaLearn in the Magazine.

digital_media: Flag indicating whether the lead had seen the ad of ExtraaLearn on the digital platforms.

educational_channels: Flag indicating whether the lead had heard about ExtraaLearn in the education channels like online forums, discussion threads, educational websites, etc.

referral: Flag indicating whether the lead had heard about ExtraaLearn through reference.

status: Flag indicating whether the lead was converted to a paid customer or not.

Mounting the library
from google.colab import drive
drive.mount('/content/drive')
Mounted at /content/drive
Importing necessary libraries and data
import warnings

warnings.filterwarnings("ignore")
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)

# Libraries to help with reading and manipulating data

import pandas as pd
import numpy as np

# Library to split data
from sklearn.model_selection import train_test_split

# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)
# setting the precision of floating numbers to 5 decimal points
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# To build model for prediction
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# To tune different models
from sklearn.model_selection import GridSearchCV


# To get diferent metric scores
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    precision_recall_curve,
    roc_curve,
    make_scorer,
)
Data Overview
Observations
Sanity checks
Understand the shape of the dataset

import pandas as pd
learn = pd.read_csv("/content/drive/MyDrive/Data_Science_2026/Project/Elective Project/Elective project-Practical data Science/ExtraaLearn.csv")
data = learn.copy()
data.shape
(4612, 15)
There are total 4612 rows and 15 columns.

# Copying data to another variable to avoid any changes to original data
data = learn.copy()
View the first and the last 5 rows of the dataset
data.head()
ID	age	current_occupation	first_interaction	profile_completed	website_visits	time_spent_on_website	page_views_per_visit	last_activity	print_media_type1	print_media_type2	digital_media	educational_channels	referral	status
0	EXT001	57	Unemployed	Website	High	7	1639	1.86100	Website Activity	Yes	No	Yes	No	No	1
1	EXT002	56	Professional	Mobile App	Medium	2	83	0.32000	Website Activity	No	No	No	Yes	No	0
2	EXT003	52	Professional	Website	Medium	3	330	0.07400	Website Activity	No	No	Yes	No	No	0
3	EXT004	53	Unemployed	Website	High	4	464	2.05700	Website Activity	No	No	No	No	No	1
4	EXT005	23	Student	Website	High	4	600	16.91400	Email Activity	No	No	No	No	No	0
data.tail()
ID	age	current_occupation	first_interaction	profile_completed	website_visits	time_spent_on_website	page_views_per_visit	last_activity	print_media_type1	print_media_type2	digital_media	educational_channels	referral	status
4607	EXT4608	35	Unemployed	Mobile App	Medium	15	360	2.17000	Phone Activity	No	No	No	Yes	No	0
4608	EXT4609	55	Professional	Mobile App	Medium	8	2327	5.39300	Email Activity	No	No	No	No	No	0
4609	EXT4610	58	Professional	Website	High	2	212	2.69200	Email Activity	No	No	No	No	No	1
4610	EXT4611	57	Professional	Mobile App	Medium	1	154	3.87900	Website Activity	Yes	No	No	No	No	0
4611	EXT4612	55	Professional	Website	Medium	4	2290	2.07500	Phone Activity	No	No	No	No	No	0
Checking the info of the data
data.info() # to check duplicate entries in the data
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4612 entries, 0 to 4611
Data columns (total 15 columns):
 #   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
 0   ID                     4612 non-null   object 
 1   age                    4612 non-null   int64  
 2   current_occupation     4612 non-null   object 
 3   first_interaction      4612 non-null   object 
 4   profile_completed      4612 non-null   object 
 5   website_visits         4612 non-null   int64  
 6   time_spent_on_website  4612 non-null   int64  
 7   page_views_per_visit   4612 non-null   float64
 8   last_activity          4612 non-null   object 
 9   print_media_type1      4612 non-null   object 
 10  print_media_type2      4612 non-null   object 
 11  digital_media          4612 non-null   object 
 12  educational_channels   4612 non-null   object 
 13  referral               4612 non-null   object 
 14  status                 4612 non-null   int64  
dtypes: float64(1), int64(4), object(10)
memory usage: 540.6+ KB
The dataset contains information on 4,612 potential customers (leads) who interacted with ExtraaLearn through various marketing and engagement channels. Each row represents a unique lead, and the objective is to determine whether that lead eventually converted into a paid customer.

The dataset consists of 15 variables, including demographic information, engagement behavior, marketing exposure, and the final conversion outcome.

The dataset contains three major data categories:

4 Numerical (int64)
1 Numerical (float64)
10 Categorical (Object)
No missing values are present in the dataset.
ID column is an identifier. Let's check if each entry of the column is unique.
data.ID.nunique()
4612
Observations:

We can see that all the entries of this column ID are unique. Hence, this column would not add any value to our analysis.

Let's drop this column.

Dropping the ID column
data = data.drop(columns=["ID"], errors='ignore')
data.head()
age	current_occupation	first_interaction	profile_completed	website_visits	time_spent_on_website	page_views_per_visit	last_activity	print_media_type1	print_media_type2	digital_media	educational_channels	referral	status
0	57	Unemployed	Website	High	7	1639	1.86100	Website Activity	Yes	No	Yes	No	No	1
1	56	Professional	Mobile App	Medium	2	83	0.32000	Website Activity	No	No	No	Yes	No	0
2	52	Professional	Website	Medium	3	330	0.07400	Website Activity	No	No	Yes	No	No	0
3	53	Unemployed	Website	High	4	464	2.05700	Website Activity	No	No	No	No	No	1
4	23	Student	Website	High	4	600	16.91400	Email Activity	No	No	No	No	No	0
Exploratory Data Analysis (EDA)
EDA is an important part of any project involving data.
It is important to investigate and understand the data better before building a model with it.
A few questions have been mentioned below which will help you approach the analysis in the right manner and generate insights from the data.
A thorough analysis of the data, in addition to the questions mentioned below, should be done.
Questions

Leads will have different expectations from the outcome of the course and the current occupation may play a key role in getting them to participate in the program. Find out how current occupation affects lead status.
The company's first impression on the customer must have an impact. Do the first channels of interaction have an impact on the lead status?
The company uses multiple modes to interact with prospects. Which way of interaction works best?
The company gets leads from various channels such as print media, digital media, referrals, etc. Which of these channels have the highest lead conversion rate?
People browsing the website or mobile application are generally required to create a profile by sharing their personal data before they can access additional information.Does having more details about a prospect increase the chances of conversion?
Let's check the statistical summary of the data.

data.describe()
age	website_visits	time_spent_on_website	page_views_per_visit	status
count	4612.00000	4612.00000	4612.00000	4612.00000	4612.00000
mean	46.20121	3.56678	724.01127	3.02613	0.29857
std	13.16145	2.82913	743.82868	1.96812	0.45768
min	18.00000	0.00000	0.00000	0.00000	0.00000
25%	36.00000	2.00000	148.75000	2.07775	0.00000
50%	51.00000	3.00000	376.00000	2.79200	0.00000
75%	57.00000	5.00000	1336.75000	3.75625	1.00000
max	63.00000	30.00000	2537.00000	18.43400	1.00000
Observation:

Here are the observations from the descriptive statistics of the dataset:

Age: The average age of leads is approximately 46 years, ranging from 18 to 63. The median age is 51, suggesting a slight left skew, with a significant portion of leads in the mid-to-late career stage.

Website Visits: Leads visit the website an average of 3.57 times, with a median of 3 visits. The maximum is 30, indicating some highly engaged users, and the difference between mean and median suggests a right-skewed distribution.

Time Spent on Website: On average, leads spend about 724 seconds (approx. 12 minutes) on the website, with a median of 376 seconds (approx. 6.3 minutes). The large difference between mean and median indicates a strong right-skew, with a subset of users spending considerably more time.

Page Views per Visit: The average is 3.03 page views per visit, with a median of 2.79. The maximum is 18.43, also indicating some highly engaged users viewing many pages, and a right-skewed distribution.

Status (Conversion Rate): The mean of status is 0.29857, which indicates that approximately 29.9% of leads converted into paid customers, and about 70.1% did not. This highlights a class imbalance in the target variable.

Marketing Channels (Binary Flags): Most marketing channels like print_media_type1 (10.8%), print_media_type2 (5.1%), digital_media (11.4%), educational_channels (15.3%), and referral (2.0%) show low mean values, suggesting that only a small percentage of leads are attributed to these specific sources. Referrals, in particular, are very rare.

Engineered Scores:

profile_score: Mean is 2.47, with a maximum of 3, indicating most leads have medium to high profile completion.

occupation_score: Mean is 2.45, with a maximum of 3, suggesting a higher proportion of professionals.

marketing_exposure_score: Mean is 0.45, with a maximum of 4, showing that most leads are exposed to few marketing channels.

engagement_score: Mean is 0.57, with a max of 1.93, indicating varying levels of overall engagement.

high_engagement: Mean is 0.50, meaning about half the leads are categorized as high engagement based on the median split.

# Making a list of all catrgorical variables
cat_col = list(data.select_dtypes("object").columns)

# Printing number of count of each unique value in each column
for column in cat_col:
    print(data[column].value_counts())
    print("-" * 50)
current_occupation
Professional    2616
Unemployed      1441
Student          555
Name: count, dtype: int64
--------------------------------------------------
first_interaction
Website       2542
Mobile App    2070
Name: count, dtype: int64
--------------------------------------------------
profile_completed
High      2264
Medium    2241
Low        107
Name: count, dtype: int64
--------------------------------------------------
last_activity
Email Activity      2278
Phone Activity      1234
Website Activity    1100
Name: count, dtype: int64
--------------------------------------------------
print_media_type1
No     4115
Yes     497
Name: count, dtype: int64
--------------------------------------------------
print_media_type2
No     4379
Yes     233
Name: count, dtype: int64
--------------------------------------------------
digital_media
No     4085
Yes     527
Name: count, dtype: int64
--------------------------------------------------
educational_channels
No     3907
Yes     705
Name: count, dtype: int64
--------------------------------------------------
referral
No     4519
Yes      93
Name: count, dtype: int64
--------------------------------------------------
Observation:

The analysis shows that ExtraaLearn's lead base is dominated by professionals who primarily engage through the website and maintain medium-to-high profile completion levels. Website-generated leads and highly engaged users represent the most valuable opportunities for conversion, suggesting that marketing and sales efforts should focus on these segments to maximize enrollment and revenue.

Univariate Analysis:
# function to plot a boxplot and a histogram along the same scale.


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to the show density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram
Summary Statistics for numerical columns

Numerical:
Let's do the summary Statistics for the numerical columns to study distribution of Age, Website visits, Time spent on website, Page views per visit and status.

# Selecting numerical columns and checking the summary statistics
num_cols = data.select_dtypes('number').columns

data[num_cols].describe().T
count	mean	std	min	25%	50%	75%	max
age	4612.00000	46.20121	13.16145	18.00000	36.00000	51.00000	57.00000	63.00000
website_visits	4612.00000	3.56678	2.82913	0.00000	2.00000	3.00000	5.00000	30.00000
time_spent_on_website	4612.00000	724.01127	743.82868	0.00000	148.75000	376.00000	1336.75000	2537.00000
page_views_per_visit	4612.00000	3.02613	1.96812	0.00000	2.07775	2.79200	3.75625	18.43400
status	4612.00000	0.29857	0.45768	0.00000	0.00000	0.00000	1.00000	1.00000
Observations:

Age: The average age of leads is approximately 46 years, with ages ranging from 18 to 63 years. The median age of 51 years is higher than the mean, indicating that a significant proportion of leads belong to the mid-to-late career stage. The interquartile range (36–57 years) shows that most leads are working-age professionals who may be interested in upskilling or career advancement opportunities.

List itemWebsite Visits: Leads visited the website an average of 3.57 times, with a median of 3 visits. While 75% of leads visited the website 5 times or fewer, some leads visited as many as 30 times, suggesting the presence of highly engaged prospects. The difference between the maximum value and the upper quartile indicates potential outliers that may represent highly interested leads.

Time Spent on Website: The average time spent on the website is approximately 724 seconds (12 minutes), while the median is 376 seconds (6.3 minutes). The mean being considerably higher than the median suggests a right-skewed distribution, where a subset of leads spend significantly longer periods exploring the platform. The wide range from 0 to 2,537 seconds indicates varying levels of engagement among leads.

Page Views per Visit: On average, leads view about 3 pages per visit, with a median of 2.79 pages. Most leads view between 2.08 and 3.76 pages per session, indicating moderate browsing behavior. However, some leads view as many as 18 pages per visit, which may reflect strong interest and active evaluation of available programs.

Lead Conversion Status: The mean value of the target variable (0.299) indicates that approximately 29.9% of leads converted into paid customers, while about 70.1% did not convert. This reveals a class imbalance in the dataset, with non-converted leads forming the majority class. Such an imbalance should be considered during model building and evaluation to ensure accurate identification of potential converters.

Overall Business Insight:

The descriptive statistics suggest that ExtraaLearn's leads are predominantly middle-aged individuals with varying levels of online engagement. While most leads exhibit moderate website interaction, a smaller group demonstrates significantly higher engagement through repeated visits, longer browsing durations, and more page views. Since only about 30% of leads convert, engagement-related variables such as website visits, time spent on the website, and page views per visit are likely to play a crucial role in distinguishing high-potential leads from low-potential ones and should be closely examined during exploratory analysis and predictive modeling.

1. Variable:
To understand each variable individually before studying relationships with the target (status).

num_cols = [
    'age',
    'website_visits',
    'time_spent_on_website',
    'page_views_per_visit'
]
2. Distribution plot:
To identify the distribution shape, skewness, and Concentration of observations.

plt.figure(figsize=(15,10))

for i, col in enumerate(num_cols, 1):

    plt.subplot(2,2,i)

    sns.histplot(
        data=data,
        x=col,
        kde=True,
        bins=30
    )

    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
No description has been provided for this image
Obervation:

Age Distribution: The majority of leads fall within the 55–60 years age group, with the highest concentration around 57–59 years. There is also a smaller concentration of leads in the 30–35 years age range, but it is significantly lower than the older age group. The distribution is not symmetric. The distribution is left-skewed (negatively skewed) because a large proportion of observations are concentrated at higher ages (55–60 years), with fewer observations spread across younger age groups. This suggests that ExtraaLearn attracts more experienced professionals than younger individuals.

Website Visits Distribution: Most leads visit the website only a few times, typically between 2 and 5 visits. The distribution is positively skewed (right-skewed), meaning a small number of users visit the website very frequently (up to 30 visits), but these cases are rare. This indicates that the majority of leads engage with the platform a limited number of times before making a decision. A small group of highly engaged users repeatedly return to the website and may represent high-potential leads.

Time Spent on Website Distribution: Most users spend a relatively low to moderate amount of time on the website, generally below 500 seconds. However, there is a substantial secondary cluster between 1,500 and 2,300 seconds, indicating a group of highly engaged users. The distribution is strongly right-skewed, with a long tail extending toward higher values. The dataset contains two distinct groups: Casual visitors who spend limited time exploring. Highly engaged leads who spend significant time researching programs and are likely to have stronger purchase intent.

Page Views per Visit Distribution Most leads view between 2 and 4 pages per visit, with peaks around 2.5 and 4 pages. Very few users view more than 8 pages per visit, although some extreme cases reach up to 18 pages. The distribution is positively skewed, indicating that deep exploration is less common.

Business Insight:

Most users perform moderate exploration before leaving the website. Leads viewing more pages are likely comparing programs, reviewing course details, and evaluating enrollment options, making them potential high-conversion prospects. Overall Conclusion The univariate analysis indicates that ExtraaLearn's lead base is dominated by older, working-age individuals, likely professionals seeking career advancement. Most leads visit the website only a few times and view a moderate number of pages, suggesting limited engagement for the majority. However, a smaller segment of leads exhibits very high engagement through repeated visits, longer browsing sessions, and extensive page exploration. These highly engaged users are likely to be valuable prospects and may play a significant role in driving lead conversions.

3. Outlier detection:
This helps to identify Extreme Observatio and highly engaged leads

plt.figure(figsize=(15,10))

for i, col in enumerate(num_cols, 1):

    plt.subplot(2,2,i)

    sns.boxplot(
        x=data[col]
    )

    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()
No description has been provided for this image
Observation: Outlier Analysis Using Boxplots

Age The age variable ranges approximately from 18 to 63 years. The median age is around 51 years, indicating that at least half of the leads are older than 51. The middle 50% of leads (IQR) lie roughly between 36 and 57 years. No outliers are observed, suggesting that the age distribution is well-contained within the expected range and does not contain any unusual values. This indicates that age data is consistent and does not require outlier treatment.
Business Insight: ExtraaLearn's leads primarily consist of working-age and experienced individuals, which aligns well with the target audience for upskilling and reskilling programs.

Website Visits Most leads visit the website between 2 and 5 times, with a median of approximately 3 visits. Several observations beyond 10 visits appear as outliers, with some leads visiting as many as 30 times. The large number of high-end outliers indicates that a subset of users repeatedly return to the website.
Business Insight: These outliers likely represent highly interested prospects who are actively researching programs before making an enrollment decision. Such leads may have a higher probability of conversion and should not be removed without further analysis.

Time Spent on Website The median time spent on the website is approximately 376 seconds. The data exhibits a wide spread, with values ranging from 0 to over 2,500 seconds. Despite the large range, the boxplot does not show significant outliers beyond the whiskers because the distribution itself is highly spread out. This indicates substantial variation in user engagement levels.
Business Insight: Some leads spend considerably more time exploring course details, pricing, and program information. Longer browsing durations are generally associated with stronger purchase intent and should be retained as valuable behavioral information.

Page Views per Visit Most leads view between 2 and 4 pages per visit, with a median close to 2.8 pages. A large number of outliers are observed above 6 pages per visit, with some users viewing as many as 18 pages. The concentration of outliers on the higher side indicates a group of highly engaged users who explore the website extensively.
Business Insight: Leads with high page-view counts are likely comparing courses, reviewing program details, and evaluating enrollment options. These users may represent high-quality leads and could be strong predictors of conversion.

Overall Conclusion:
The boxplots reveal that website_visits and page_views_per_visit contain several high-value outliers, while age shows no unusual observations and time_spent_on_website exhibits a naturally wide distribution. Since these outliers reflect genuine user engagement rather than data entry errors, they provide meaningful business insights and should generally be retained for predictive modeling. In particular, leads with unusually high website visits, longer browsing durations, and more page views are likely to be highly interested prospects and may have a greater likelihood of converting into paid customers.

4. Advanced distribution:
This will show density, distribution shape and outliers.

plt.figure(figsize=(15,10))

for i, col in enumerate(num_cols, 1):

    plt.subplot(2,2,i)

    sns.violinplot(
        y=data[col]
    )

    plt.title(f'Violin Plot of {col}')

plt.tight_layout()
plt.show()
No description has been provided for this image
Observations
Age The distribution is concentrated around 55–60 years, indicating that this age group forms the largest segment of leads. There are smaller peaks around 20–25 years, 30–35 years, and 45–50 years, suggesting multiple age groups are present. The median age is approximately 51 years, confirming that a large portion of leads are mature professionals. The distribution is multimodal, meaning it contains several peaks rather than one single dominant age cluster. Business Insight ExtraaLearn attracts a diverse audience, but the strongest concentration comes from experienced professionals nearing the late stages of their careers. Younger segments are present but represent a smaller proportion of the lead pool.
Website Visits Most leads visit the website between 1 and 5 times, with the highest density around 2–3 visits. The distribution is strongly right-skewed, with a long tail extending to over 30 visits. Only a small fraction of users repeatedly visit the website. Business Insight Most users gather information within a few visits before making a decision. The small group of highly frequent visitors may represent highly interested prospects who require targeted follow-up from the sales team.
Time Spent on Website Two distinct peaks are visible: One between 100 and 400 seconds Another between 1,600 and 2,200 seconds This indicates a bimodal distribution, suggesting two different user engagement behaviors. The median lies closer to the lower cluster, indicating that most users spend relatively limited time on the website. Business Insight The first group likely consists of casual visitors who quickly browse content. The second group represents highly engaged leads who spend substantial time exploring courses, pricing, and program details. These highly engaged leads may have a greater likelihood of conversion.
Page Views per Visit Most users view between 2 and 4 pages per visit. There is another smaller concentration around 5–6 pages per visit. A long right tail extends up to approximately 18 pages, indicating a small group of users who explore the website extensively. The distribution is positively skewed with several distinct density peaks. Business Insight The majority of leads perform moderate exploration before leaving the website. Leads viewing more than 5 pages are likely comparing programs and researching enrollment options, indicating stronger purchase intent.
Overall Conclusion
The violin plots reveal clear differences in lead engagement behavior. Age distribution shows that the lead base is dominated by experienced professionals, while website visits and page views indicate that most users interact with the platform only a few times before leaving. However, a smaller segment of highly engaged users repeatedly visits the website, spends significantly more time browsing, and explores a larger number of pages. The bimodal pattern observed in time spent on the website suggests the presence of both casual visitors and serious prospects. These engagement-related variables are likely to be strong predictors of lead conversion and should be examined further during bivariate analysis with the target variable (status).

# Checking the rows where the avg_website_visit is 0
data[data["website_visits"] == 0]
age	current_occupation	first_interaction	profile_completed	website_visits	time_spent_on_website	page_views_per_visit	last_activity	print_media_type1	print_media_type2	digital_media	educational_channels	referral	status
33	56	Professional	Mobile App	Medium	0	0	0.00000	Phone Activity	No	No	No	No	No	0
37	59	Professional	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	1
54	28	Professional	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	Yes	No	0
94	49	Unemployed	Website	High	0	0	0.00000	Email Activity	Yes	No	No	No	No	1
127	54	Professional	Mobile App	High	0	0	0.00000	Phone Activity	No	No	No	No	No	0
163	56	Unemployed	Mobile App	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
170	34	Professional	Website	Medium	0	0	0.00000	Email Activity	No	Yes	No	Yes	No	0
181	55	Unemployed	Mobile App	High	0	0	0.00000	Email Activity	Yes	No	No	No	No	0
193	41	Professional	Mobile App	Low	0	0	0.00000	Phone Activity	Yes	Yes	No	Yes	No	0
213	32	Unemployed	Mobile App	High	0	0	0.00000	Email Activity	Yes	No	No	No	No	0
239	28	Professional	Website	High	0	0	0.00000	Email Activity	No	No	No	Yes	No	1
270	33	Unemployed	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
302	58	Professional	Mobile App	Medium	0	0	0.00000	Phone Activity	No	No	No	Yes	No	0
355	36	Professional	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
400	59	Unemployed	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
422	19	Student	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	1
432	21	Student	Mobile App	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
443	59	Professional	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
451	58	Professional	Website	High	0	0	0.00000	Phone Activity	No	No	No	No	No	1
463	32	Unemployed	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
478	44	Professional	Mobile App	High	0	0	0.00000	Email Activity	Yes	No	No	No	No	0
483	20	Student	Website	High	0	0	0.00000	Email Activity	No	No	No	Yes	No	1
547	54	Professional	Mobile App	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
556	33	Professional	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
652	58	Unemployed	Website	High	0	0	0.00000	Phone Activity	No	No	No	No	No	1
677	60	Unemployed	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
682	34	Professional	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	Yes	No	0
699	55	Professional	Website	Low	0	0	0.00000	Phone Activity	No	No	No	No	No	0
725	19	Student	Mobile App	Medium	0	0	0.00000	Email Activity	No	No	No	Yes	No	0
737	58	Unemployed	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
774	59	Professional	Website	Medium	0	0	0.00000	Phone Activity	No	No	No	No	No	0
784	46	Professional	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
796	24	Student	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
804	56	Unemployed	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
823	30	Professional	Mobile App	Medium	0	0	0.00000	Email Activity	No	No	No	Yes	No	0
861	23	Student	Website	Medium	0	0	0.00000	Email Activity	No	Yes	No	No	No	0
875	58	Unemployed	Mobile App	High	0	0	0.00000	Phone Activity	No	No	No	No	No	0
878	56	Professional	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
917	43	Professional	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	1
946	57	Professional	Website	High	0	0	0.00000	Email Activity	Yes	No	No	No	No	1
959	58	Professional	Mobile App	Medium	0	0	0.00000	Phone Activity	No	No	No	No	No	0
985	60	Professional	Mobile App	Medium	0	0	0.00000	Phone Activity	No	No	No	Yes	No	0
994	37	Professional	Website	Medium	0	0	0.00000	Phone Activity	Yes	No	No	Yes	No	0
1013	50	Professional	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
1049	57	Professional	Mobile App	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
1054	56	Unemployed	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	1
1097	59	Professional	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	1
1182	45	Unemployed	Mobile App	Medium	0	0	0.00000	Phone Activity	Yes	Yes	No	No	No	0
1187	37	Professional	Website	Medium	0	0	0.00000	Phone Activity	No	No	Yes	Yes	No	0
1210	63	Unemployed	Mobile App	High	0	0	0.00000	Phone Activity	No	No	No	No	No	0
1354	51	Professional	Mobile App	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
1372	50	Unemployed	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
1375	58	Professional	Website	High	0	0	0.00000	Phone Activity	No	No	No	No	No	1
1385	42	Unemployed	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
1388	37	Unemployed	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
1440	43	Professional	Mobile App	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
1512	44	Professional	Mobile App	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
1533	59	Professional	Mobile App	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
1542	32	Professional	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
1581	57	Professional	Website	Medium	0	0	0.00000	Email Activity	No	Yes	No	No	No	0
1592	56	Professional	Mobile App	Medium	0	0	0.00000	Phone Activity	No	No	No	No	No	0
1602	31	Professional	Website	Medium	0	0	0.00000	Phone Activity	No	No	No	Yes	No	0
1619	20	Student	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
1638	57	Professional	Website	Medium	0	0	0.00000	Email Activity	Yes	No	No	No	No	0
1658	56	Professional	Mobile App	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
1675	56	Professional	Mobile App	Medium	0	0	0.00000	Phone Activity	No	No	No	No	No	0
1679	32	Unemployed	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
1707	45	Professional	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
1724	26	Professional	Mobile App	Medium	0	0	0.00000	Email Activity	No	No	No	Yes	No	0
1801	36	Unemployed	Website	Medium	0	0	0.00000	Phone Activity	No	No	No	No	No	0
1840	30	Professional	Website	High	0	0	0.00000	Email Activity	No	Yes	No	No	No	0
1858	18	Student	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
1903	55	Professional	Website	Medium	0	0	0.00000	Email Activity	Yes	Yes	No	No	No	0
1907	31	Professional	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	1
1924	48	Professional	Mobile App	High	0	0	0.00000	Phone Activity	Yes	No	Yes	No	No	0
2034	53	Professional	Mobile App	Medium	0	0	0.00000	Email Activity	No	No	No	Yes	No	0
2047	58	Professional	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
2080	59	Unemployed	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	1
2088	49	Professional	Website	Medium	0	0	0.00000	Phone Activity	No	No	No	No	No	0
2187	41	Professional	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	1
2203	44	Professional	Website	High	0	0	0.00000	Phone Activity	Yes	No	No	No	No	1
2210	43	Unemployed	Mobile App	Medium	0	0	0.00000	Phone Activity	No	No	Yes	No	No	0
2241	59	Unemployed	Mobile App	Medium	0	0	0.00000	Phone Activity	Yes	No	No	No	No	0
2247	37	Professional	Website	High	0	0	0.00000	Phone Activity	No	No	No	No	No	1
2272	57	Professional	Mobile App	High	0	0	0.00000	Email Activity	No	Yes	No	No	No	0
2274	44	Unemployed	Website	Low	0	0	0.00000	Phone Activity	No	No	No	No	No	0
2295	56	Professional	Mobile App	Medium	0	0	0.00000	Email Activity	No	No	No	Yes	No	0
2323	57	Unemployed	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
2325	21	Student	Website	Medium	0	0	0.00000	Email Activity	Yes	No	No	No	No	0
2423	24	Student	Mobile App	High	0	0	0.00000	Email Activity	Yes	No	No	No	No	0
2425	56	Unemployed	Website	High	0	0	0.00000	Phone Activity	Yes	No	No	No	No	0
2430	58	Professional	Website	High	0	0	0.00000	Email Activity	Yes	No	No	No	No	1
2434	59	Professional	Website	Medium	0	0	0.00000	Phone Activity	No	No	Yes	Yes	No	0
2481	59	Professional	Website	High	0	0	0.00000	Email Activity	No	No	Yes	No	No	1
2529	46	Unemployed	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
2551	57	Professional	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
2559	63	Unemployed	Website	Medium	0	0	0.00000	Phone Activity	No	No	No	No	No	0
2605	23	Student	Website	High	0	0	0.00000	Email Activity	No	No	Yes	No	No	0
2626	58	Professional	Mobile App	Medium	0	0	0.00000	Phone Activity	Yes	No	Yes	No	No	0
2651	57	Professional	Mobile App	Medium	0	0	0.00000	Phone Activity	No	No	No	No	No	0
2665	57	Professional	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	Yes	No	0
2674	47	Professional	Mobile App	Medium	0	0	0.00000	Phone Activity	No	No	No	No	No	0
2698	59	Unemployed	Website	Medium	0	0	0.00000	Phone Activity	No	No	No	No	No	0
2704	43	Unemployed	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	1
2729	34	Professional	Website	Medium	0	0	0.00000	Email Activity	No	No	Yes	No	No	0
2749	32	Professional	Mobile App	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
2789	59	Unemployed	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
2799	52	Unemployed	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
2804	37	Professional	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
2860	56	Unemployed	Website	High	0	0	0.00000	Phone Activity	No	No	No	No	No	1
2899	51	Professional	Website	High	0	0	0.00000	Email Activity	No	No	No	No	Yes	1
2908	56	Professional	Website	Medium	0	0	0.00000	Email Activity	No	No	Yes	Yes	Yes	0
2916	59	Unemployed	Website	Medium	0	0	0.00000	Phone Activity	No	No	Yes	No	No	0
2931	60	Professional	Mobile App	High	0	0	0.00000	Phone Activity	No	No	No	No	No	0
2934	53	Professional	Website	High	0	0	0.00000	Email Activity	Yes	No	No	No	No	1
2941	56	Professional	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	1
2942	22	Student	Mobile App	Medium	0	0	0.00000	Phone Activity	No	No	No	No	No	0
2946	42	Professional	Mobile App	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
2947	58	Unemployed	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
2977	61	Unemployed	Website	Medium	0	0	0.00000	Phone Activity	No	No	No	No	No	0
2980	58	Professional	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
3001	60	Unemployed	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
3013	58	Unemployed	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
3161	57	Professional	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
3176	50	Unemployed	Mobile App	Medium	0	0	0.00000	Phone Activity	No	No	No	No	No	0
3212	37	Unemployed	Mobile App	High	0	0	0.00000	Phone Activity	No	No	No	No	No	0
3227	59	Unemployed	Website	Medium	0	0	0.00000	Phone Activity	No	No	No	No	No	0
3278	57	Professional	Mobile App	Medium	0	0	0.00000	Phone Activity	No	No	No	No	No	0
3294	51	Professional	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
3305	21	Student	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
3307	60	Professional	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
3322	43	Unemployed	Mobile App	Medium	0	0	0.00000	Phone Activity	No	No	No	No	No	0
3336	56	Professional	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
3441	57	Professional	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	Yes	No	0
3456	58	Professional	Mobile App	High	0	0	0.00000	Phone Activity	No	No	No	No	No	0
3545	20	Student	Website	High	0	0	0.00000	Phone Activity	No	No	No	No	No	0
3558	54	Unemployed	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
3596	60	Unemployed	Website	Medium	0	0	0.00000	Phone Activity	No	No	No	No	Yes	0
3611	21	Student	Website	Medium	0	0	0.00000	Email Activity	Yes	No	No	No	No	0
3625	60	Unemployed	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
3664	22	Student	Website	High	0	0	0.00000	Phone Activity	No	No	Yes	No	No	0
3684	59	Professional	Website	Medium	0	0	0.00000	Email Activity	No	No	No	Yes	No	0
3747	42	Professional	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	1
3748	58	Unemployed	Website	Low	0	0	0.00000	Phone Activity	No	No	No	No	No	0
3791	53	Unemployed	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
3797	62	Unemployed	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
3800	42	Professional	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
3807	35	Professional	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
3863	52	Professional	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
3923	24	Student	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
3925	48	Professional	Website	Medium	0	0	0.00000	Phone Activity	No	No	No	No	No	0
3979	60	Professional	Mobile App	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
3990	56	Unemployed	Mobile App	High	0	0	0.00000	Phone Activity	Yes	No	Yes	No	No	0
4001	60	Professional	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	1
4039	44	Professional	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
4046	38	Unemployed	Website	High	0	0	0.00000	Email Activity	No	No	Yes	No	No	1
4116	59	Professional	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
4168	58	Professional	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	No	No	0
4190	57	Professional	Website	High	0	0	0.00000	Phone Activity	Yes	Yes	No	No	No	1
4241	55	Professional	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
4247	59	Professional	Mobile App	High	0	0	0.00000	Phone Activity	No	No	No	Yes	No	0
4305	56	Professional	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
4327	48	Professional	Website	Medium	0	0	0.00000	Email Activity	Yes	No	No	No	No	0
4362	18	Student	Mobile App	Medium	0	0	0.00000	Phone Activity	No	No	No	No	No	0
4364	59	Unemployed	Website	Medium	0	0	0.00000	Email Activity	Yes	No	No	No	No	0
4366	55	Professional	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	1
4367	40	Unemployed	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
4369	57	Unemployed	Website	High	0	0	0.00000	Email Activity	No	No	Yes	No	No	1
4439	61	Unemployed	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
4440	53	Professional	Mobile App	High	0	0	0.00000	Email Activity	No	No	No	Yes	No	0
4447	38	Professional	Website	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
4453	56	Professional	Website	High	0	0	0.00000	Phone Activity	No	No	No	No	No	1
4557	48	Unemployed	Mobile App	Medium	0	0	0.00000	Email Activity	No	No	No	No	No	0
4578	58	Professional	Website	High	0	0	0.00000	Email Activity	No	No	No	No	No	1
Categorical:
Let's do the summary Statistics for the Categorical columns to study distribution of Occupation, profile completion, First interaction, last activity, print_media_type1, print_media_type2, digital_media, educational_channels, referral and status.

# function to create labeled barplots


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot
cat_cols = ['current_occupation', 'profile_completed', 'first_interaction', 'last_activity', 'print_media_type1', 'print_media_type2', 'digital_media', 'educational_channels', 'referral','status' ]

for col in cat_cols:
    print(data[col].value_counts(normalize = True))  # The parameter normalize = True gives the percentage of each category
    print('*'*40)                                    # Print the * 40 times to separate different variables
current_occupation
Professional   0.56722
Unemployed     0.31245
Student        0.12034
Name: proportion, dtype: float64
****************************************
profile_completed
High     0.49089
Medium   0.48591
Low      0.02320
Name: proportion, dtype: float64
****************************************
first_interaction
Website      0.55117
Mobile App   0.44883
Name: proportion, dtype: float64
****************************************
last_activity
Email Activity     0.49393
Phone Activity     0.26756
Website Activity   0.23851
Name: proportion, dtype: float64
****************************************
print_media_type1
No    0.89224
Yes   0.10776
Name: proportion, dtype: float64
****************************************
print_media_type2
No    0.94948
Yes   0.05052
Name: proportion, dtype: float64
****************************************
digital_media
No    0.88573
Yes   0.11427
Name: proportion, dtype: float64
****************************************
educational_channels
No    0.84714
Yes   0.15286
Name: proportion, dtype: float64
****************************************
referral
No    0.97984
Yes   0.02016
Name: proportion, dtype: float64
****************************************
status
0   0.70143
1   0.29857
Name: proportion, dtype: float64
****************************************
Observation:
Current Occupation Professionals (56.7%) constitute the largest segment of leads, followed by Unemployed individuals (31.2%) and Students (12.0%). Nearly 88% of leads are either professionals or unemployed, indicating that ExtraaLearn's programs primarily attract individuals focused on career growth, upskilling, and employability. Students represent a relatively small share of the lead base.
Profile Completed Almost all leads have either High (49.1%) or Medium (48.6%) profile completion levels. Only 2.3% of leads have a Low profile completion rate. This suggests that most leads are willing to provide detailed information, indicating a generally engaged and interested audience.
First Interaction 55.1% of leads first interacted through the Website, while 44.9% came through the Mobile App. Both channels contribute significantly to lead generation, although the website remains the primary acquisition source. The relatively balanced distribution suggests that both digital platforms are important for attracting potential customers.
Last Activity Email Activity (49.4%) is the most common last interaction recorded. Phone Activity (26.8%) and Website Activity (23.9%) occur less frequently. This indicates that email communication is the dominant channel used to engage and nurture leads throughout their journey.
Print Media Type 1 (Newspaper Advertisement) Only 10.8% of leads reported exposure through newspaper advertisements. A large majority (89.2%) were not influenced by this channel. Newspaper advertising appears to contribute relatively little to lead generation.
Print Media Type 2 (Magazine Advertisement) Only 5.1% of leads were exposed to magazine advertisements. Approximately 95% had no interaction with this marketing source. Magazine advertising has the lowest reach among the traditional marketing channels analyzed.
Digital Media Around 11.4% of leads reported discovering ExtraaLearn through digital media advertisements. The majority (88.6%) were not acquired through this channel. While digital media contributes to lead generation, its direct impact appears limited based on the available data.
Educational Channels 15.3% of leads learned about ExtraaLearn through educational channels such as forums, educational websites, and learning communities. Although still a minority, this channel has a higher reach than print and digital media sources. Educational channels may be attracting more relevant and academically motivated prospects.
Referral Only 2.0% of leads were acquired through referrals. While referral volume is very low, referral leads often demonstrate higher trust and stronger purchase intent. This variable may still be an important predictor of conversion despite its small representation.
Target Variable (Status) Approximately 29.9% of leads converted into paid customers, while 70.1% did not convert. The dataset exhibits a moderate class imbalance, with non-converted leads significantly outnumbering converted leads. This highlights the importance of identifying high-quality leads and developing predictive models that can improve conversion targeting.
Overall Business Insight
The analysis shows that ExtraaLearn's lead base is dominated by professionals and unemployed individuals, most of whom have medium to high profile completion levels and initially engage through digital channels such as the website and mobile app. Email communication serves as the primary engagement channel, while traditional advertising sources contribute relatively little to lead acquisition. Although referrals represent a small portion of leads, they may still be valuable due to their potential conversion strength. Finally, with only 30% of leads converting into paid customers, understanding the factors that distinguish converting leads from non-converting leads will be critical for optimizing marketing efforts and improving resource allocation.

Variable:
To understand each variable individually before studying relationships with the target (status).

cat_cols = [
'current_occupation',
    'profile_completed',
    'first_interaction',
    'last_activity',
    'print_media_type1',
    'print_media_type2',
    'digital_media',
    'educational_channels',
    'referral','status' ]
Observations on current_occupation
data = learn.copy()
labeled_barplot(data, "current_occupation", perc=True)
No description has been provided for this image
Observations on current_occupation
Professionals constitute the largest segment of leads, accounting for approximately 56.7%.
Unemployed individuals represent a significant portion at about 31.2%.
Students make up the smallest group, with approximately 12.0%.
This distribution suggests that ExtraaLearn primarily attracts individuals who are either currently working and looking to upskill, or are unemployed and seeking new career opportunities. The student segment is comparatively smaller, indicating a stronger appeal to an older, career-focused demographic.
Observations on number of first_interaction
labeled_barplot(data, "first_interaction", perc=True)
No description has been provided for this image
Observations on first_interaction
The majority of leads (approximately 55.1%) first interacted with ExtraaLearn through the 'Website'.
A significant portion (about 44.9%) first interacted through the 'Mobile App'.
This indicates that both digital platforms are key channels for initial lead acquisition, with the website being slightly more dominant. Understanding the quality of leads from each channel could be beneficial for optimizing marketing strategies.
Observations on profile_completed
labeled_barplot(data, "profile_completed", perc=True)
No description has been provided for this image
Observations on profile_completed
Almost half of the leads (approximately 49.1%) have a 'High' profile completion level, and another significant portion (about 48.6%) has a 'Medium' completion level.
Only a very small percentage of leads (around 2.3%) have a 'Low' profile completion status.
This indicates that the vast majority of ExtraaLearn's leads are engaged enough to complete a substantial portion of their profiles, suggesting a strong initial interest in the offerings.
Observations on last_activity
labeled_barplot(data, "last_activity", perc=True)
No description has been provided for this image
Observations on last_activity
'Email Activity' is the most common last interaction, accounting for approximately 49.4% of leads.
'Phone Activity' is the second most frequent, with about 26.8% of leads.
'Website Activity' is the least common as a last activity, representing approximately 23.9% of leads.
This indicates that email is a primary channel for ongoing engagement and nurturing leads, suggesting its importance in the lead journey.
Observations on print_media_type1
labeled_barplot(data, "print_media_type1", perc=True)
No description has been provided for this image
Observations on print_media_type1
Approximately 89.2% of leads did not indicate exposure through newspaper advertisements, while 10.8% did.
This suggests that newspaper advertisements are a relatively small source of lead generation for ExtraaLearn.
While it contributes to lead acquisition, its reach is limited compared to other channels. Further analysis would be needed to assess the quality of leads generated from this channel.
Observations on print_media_type2
labeled_barplot(data, "print_media_type2", perc=True)
No description has been provided for this image
Observations on print_media_type2
Approximately 94.9% of leads did not indicate exposure through magazine advertisements, while only 5.1% did.
This suggests that magazine advertisements are a very minor source of lead generation for ExtraaLearn, even less so than newspaper advertisements.
The low reach of this channel indicates that it might not be an effective marketing strategy for ExtraaLearn, and resources allocated to it could potentially be re-evaluated for more impactful channels.
Observations on digital_media
labeled_barplot(data, "digital_media", perc=True)
No description has been provided for this image
Observations on digital_media
Approximately 88.6% of leads were not acquired through digital media advertisements, while only 11.4% were.
This suggests that digital media, while contributing, is not a primary channel for lead generation compared to other sources like the website.
Further analysis might be needed to determine the effectiveness and ROI of digital media campaigns in terms of lead quality and conversion rates.
Observations on educational_channels
labeled_barplot(data, "educational_channels", perc=True)
No description has been provided for this image
Observations on educational_channels
Approximately 84.7% of leads did not hear about ExtraaLearn through educational channels, while 15.3% did.
This indicates that while educational channels are not the primary source of leads, they still contribute a notable portion.
Leads from educational channels might be more pre-disposed to educational offerings, potentially leading to higher quality leads, and further analysis could explore their conversion rates.
Observations on referral
labeled_barplot(data, "referral", perc=True)
No description has been provided for this image
Observations on referral
Approximately 98.0% of leads were not acquired through referrals, while only 2.0% were.
This indicates that referrals are a very small source of lead generation for ExtraaLearn.
Despite the low volume, referred leads often have higher conversion rates due to trust and pre-existing relationships. This variable could still be important for predicting conversion, even if it's a rare event.
Observations on status
labeled_barplot(data, "status", perc=True)
No description has been provided for this image
Observations on status
Approximately 70.1% of leads have a 'status' of 0 (Did Not Convert), while 29.9% have a 'status' of 1 (Converted).
This indicates a class imbalance, with significantly more non-converted leads than converted leads in the dataset.
This imbalance is a common characteristic in lead conversion datasets and will need to be considered during model training to avoid bias towards the majority class.
2. Count Plot:
This is the most important univariate plot for categorical variables.

for col in cat_cols:
    labeled_barplot(data, col, perc=False)
No description has been provided for this image
No description has been provided for this image
No description has been provided for this image
No description has been provided for this image
No description has been provided for this image
No description has been provided for this image
No description has been provided for this image
No description has been provided for this image
No description has been provided for this image
No description has been provided for this image
Observation:

Observation: Referral Leads

The vast majority of leads (4,519 leads; ~98%) were not acquired through referrals.

Only 93 leads (~2%) came through the referral channel.

This indicates that referrals currently contribute a very small portion of - ExtraaLearn's total lead generation efforts.

Despite the low volume, earlier Logistic Regression results showed that referral leads had one of the strongest positive impacts on conversion probability, making them a high-quality lead source.

The business should not focus solely on increasing lead volume but should also prioritize referral programs because referral leads are typically more trusted, better informed, and more likely to convert. Business Insight

Although referral leads represent only 2% of total leads, they are among the most valuable leads in terms of conversion potential. Expanding referral programs, offering incentives, and encouraging existing learners to refer friends and colleagues could significantly improve conversion rates and reduce customer acquisition costs.

3. Percentage Distribution Plot:
This is useful for business reporting.

for col in cat_cols:

    plt.figure(figsize=(7,5))

    (
        data[col]
        .value_counts(normalize=True)
        .mul(100)
        .sort_values(ascending=False)
        .plot(kind='bar')
    )

    plt.title(f'Percentage Distribution of {col}')
    plt.ylabel('Percentage')

    plt.show()

Observation:

Professionals account for approximately 57% of all leads, making them the largest customer segment in the dataset.

Unemployed individuals represent about 31% of leads, indicating a substantial group interested in upskilling or reskilling opportunities to improve their employment prospects.

Students comprise only around 12% of the lead base, making them the smallest segment among the three occupation categories.

The lead distribution is heavily skewed toward working professionals, suggesting that ExtraaLearn's programs are particularly appealing to individuals seeking career growth, promotions, or skill enhancement.

The significant proportion of unemployed leads highlights an opportunity for targeted marketing of job-oriented certification and placement-focused programs.

Since professionals and unemployed individuals together constitute nearly 88% of all leads, these groups are likely to have the greatest influence on overall lead conversion performance.

Business Insight
The analysis indicates that ExtraaLearn's primary audience consists of professionals and job seekers rather than students. Therefore, marketing campaigns, course content, and sales strategies should focus on career advancement, industry-relevant skills, certification programs, and employability outcomes. Additionally, occupation is likely to be an important predictor of lead conversion and should be further analyzed against the target variable (status) during bivariate analysis.

4. Summary table for categorical variables:
for col in cat_cols:

    print(f"\n{col.upper()}")

    display(
        pd.DataFrame({
            'Count': data[col].value_counts(),
            'Percentage':
            round(
                data[col].value_counts(normalize=True)*100,
                2
            )
        })
    )
CURRENT_OCCUPATION
Count	Percentage
current_occupation		
Professional	2616	56.72000
Unemployed	1441	31.24000
Student	555	12.03000
PROFILE_COMPLETED
Count	Percentage
profile_completed		
High	2264	49.09000
Medium	2241	48.59000
Low	107	2.32000
FIRST_INTERACTION
Count	Percentage
first_interaction		
Website	2542	55.12000
Mobile App	2070	44.88000
LAST_ACTIVITY
Count	Percentage
last_activity		
Email Activity	2278	49.39000
Phone Activity	1234	26.76000
Website Activity	1100	23.85000
PRINT_MEDIA_TYPE1
Count	Percentage
print_media_type1		
No	4115	89.22000
Yes	497	10.78000
PRINT_MEDIA_TYPE2
Count	Percentage
print_media_type2		
No	4379	94.95000
Yes	233	5.05000
DIGITAL_MEDIA
Count	Percentage
digital_media		
No	4085	88.57000
Yes	527	11.43000
EDUCATIONAL_CHANNELS
Count	Percentage
educational_channels		
No	3907	84.71000
Yes	705	15.29000
REFERRAL
Count	Percentage
referral		
No	4519	97.98000
Yes	93	2.02000
STATUS
Count	Percentage
status		
0	3235	70.14000
1	1377	29.86000
Observation and Business Insights from Categorical Variable Distributions
1. Current Occupation

Observation: Professionals constitute the largest segment of leads (approximately 56.7%), followed by unemployed individuals (about 31.2%). Students make up the smallest proportion (around 12.0%).
Business Insight: ExtraaLearn's core audience consists of working professionals and job seekers. Marketing and course development efforts should continue to target these groups, focusing on career advancement, skill enhancement, and employment-oriented programs.
2. Profile Completed

Observation: A vast majority of leads have either 'High' (approximately 49.1%) or 'Medium' (about 48.6%) profile completion. Only a very small fraction (around 2.3%) has 'Low' completion.
Business Insight: The high profile completion rate indicates strong interest and engagement from leads, suggesting they are more serious prospects. This information can be valuable for prioritizing leads for follow-up.
3. First Interaction

Observation: The website is the primary channel for the first interaction, accounting for approximately 55.1% of leads. The mobile app is also a significant channel, contributing about 44.9%.
Business Insight: Both the website and mobile app are crucial lead generation channels. Continuous optimization of user experience on both platforms is essential, with a particular focus on the website as the dominant initial touchpoint.
4. Last Activity

Observation: Email activity is the most common last interaction (approximately 49.4%), followed by phone activity (about 26.8%) and website activity (about 23.9%).
Business Insight: Email serves as a critical communication channel for lead nurturing. Effective email campaigns and follow-up strategies are vital. Phone and website activities, while less frequent as a 'last activity', also play important roles in different stages of the lead journey and may indicate more direct engagement.
Overall Observations from Univariate Analysis (Numerical and Categorical)
Numerical Insights:
Age: The lead base is primarily composed of experienced professionals, with a significant concentration in the 55-60 age group, and a smaller but notable segment in the 30-35 age range. This suggests ExtraaLearn appeals more to career-focused individuals.
Website Engagement (website_visits, time_spent_on_website, page_views_per_visit): Most leads exhibit moderate engagement (2-5 visits, low-moderate time, 2-4 page views). However, a distinct, smaller segment shows exceptionally high engagement (30+ visits, 1600-2200 seconds, 18+ page views). These highly engaged leads, though outliers in statistical terms, are crucial business insights, likely indicating strong intent and high potential for conversion. The bimodal distribution in 'time_spent_on_website' particularly highlights the presence of both casual browsers and serious prospects.
Categorical Insights:
Current Occupation: Professionals (56.7%) and Unemployed (31.2%) dominate the lead base, reinforcing the focus on career growth and skill enhancement. Students represent a much smaller segment.
Profile Completion: A remarkable 97.7% of leads complete their profiles to a High or Medium level, indicating high engagement and willingness to share information, which can be a strong signal for qualification.
First Interaction: The website (55.1%) and mobile app (44.9%) are both critical initial touchpoints, with the website having a slight edge. Both platforms are essential for lead acquisition.
Last Activity: Email activity (49.4%) is the primary channel for continued engagement and nurturing leads, followed by phone and website activities. This underlines the importance of effective email campaigns.
Marketing Channels: Traditional print media (newspaper ~10.8%, magazine ~5.1%) and digital media (~11.4%) contribute minimally. Educational channels (15.3%) show a better reach, while referrals (2.0%) are very low in volume but are often associated with higher quality leads.
Lead Conversion Status (status): There is a clear class imbalance, with approximately 70.1% of leads not converting and only 29.9% converting. This imbalance will need careful consideration during model development.
Business Implications:
ExtraaLearn's target audience largely consists of older, working-age professionals and job seekers who are highly engaged during the initial stages of their journey (high profile completion). While many leads exhibit moderate digital interaction, a critical subset demonstrates exceptionally high engagement across website metrics. Lead generation efforts are primarily driven by the website and mobile app, with email being the dominant nurturing channel. Given the significant class imbalance in conversion rates, identifying the factors that differentiate converting from non-converting leads, especially those related to high engagement and specific demographics, will be paramount for optimizing marketing spend and improving conversion strategies. The next crucial step is to perform bivariate analysis with the target variable (status) to understand how each of these features directly influences lead conversion.

Bivariate Analysis
cols_list = data.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(12, 7))
sns.heatmap(
    data[cols_list].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral"
)
plt.show()
No description has been provided for this image
Observation:

Here are some observations from the correlation heatmap:

time_spent_on_website has the highest positive correlation with status (0.45). This suggests that leads who spend more time on the website are more likely to convert.
page_views_per_visit also shows a positive correlation with status (0.28). - - While lower than time_spent_on_website, it still indicates that more page - views per visit are associated with higher conversion rates.
website_visits has a positive, but weaker, correlation with status (0.17). age shows a very weak positive correlation with status (0.02), indicating it's not a strong linear predictor of conversion by itself.
Strong multicollinearity exists between time_spent_on_website and page_views_per_visit (0.75). This high correlation suggests that these two - - - features capture similar aspects of user engagement. This was partially addressed by creating an engagement_score in previous feature engineering steps.
Creating functions that will help us with further analysis.

### function to plot distributions wrt target


def distribution_plot_wrt_target(data, predictor, target):

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
        stat="density",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=data[data[target] == target_uniq[1]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
        stat="density",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()
def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 5, 5))
    plt.legend(
        loc="lower left", frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()
Leads will have different expectations from the outcome of the course and the current occupation may play a key role for them to take the program. Let's analyze it

stacked_barplot(data, "current_occupation", "status")
status                 0     1   All
current_occupation                  
All                 3235  1377  4612
Professional        1687   929  2616
Unemployed          1058   383  1441
Student              490    65   555
------------------------------------------------------------------------------------------------------------------------
No description has been provided for this image
Observation on current_occupation vs. status
From the stacked bar plot and the cross-tabulation, we can observe the following:

Professional Leads:

Total Professionals: 2616
Converted: 929
Did Not Convert: 1687
Conversion Rate: Approximately 35.5% (929 / 2616).
Unemployed Leads:

Total Unemployed: 1441
Converted: 383
Did Not Convert: 1058
Conversion Rate: Approximately 26.6% (383 / 1441).
Student Leads:

Total Students: 555
Converted: 65
Did Not Convert: 490
Conversion Rate: Approximately 11.7% (65 / 555).
Key Findings:

Professionals have the highest conversion rate (35.5%), significantly higher than Unemployed and Students.
Unemployed individuals have a moderate conversion rate (26.6%).
Students have the lowest conversion rate (11.7%).
Business Insight: ExtraaLearn's programs appear to resonate most strongly with working professionals, who show the highest propensity to convert. This suggests that the value proposition of upskilling and career advancement is highly effective for this demographic. While unemployed leads also convert at a reasonable rate, indicating interest in re-skilling or career change, students are less likely to convert. ExtraaLearn might consider tailoring marketing efforts and program offerings more specifically to the needs and conversion drivers of professionals, and re-evaluating strategies for student engagement, or acknowledging that students may not be the primary target audience for certain programs.

Age can be a good factor to differentiate between such leads

plt.figure(figsize=(10, 5))
sns.boxplot(x="current_occupation", y="age", data=data)
plt.show()
No description has been provided for this image
data.groupby(["current_occupation"])["age"].describe()
count	mean	std	min	25%	50%	75%	max
current_occupation								
Professional	2616.00000	49.34748	9.89074	25.00000	42.00000	54.00000	57.00000	60.00000
Student	555.00000	21.14414	2.00111	18.00000	19.00000	21.00000	23.00000	25.00000
Unemployed	1441.00000	50.14018	9.99950	32.00000	42.00000	54.00000	58.00000	63.00000
Observation on current_occupation vs. age
From the box plot and descriptive statistics (data.groupby(['current_occupation'])['age'].describe()):

Students:

The age of students ranges from 18 to 25 years, with a mean age of approximately 21.14 years and a median of 21 years.
This group is distinctly younger and has a very tight age distribution, as expected.
Professionals:

Professionals have an age range from 25 to 60 years, with a mean age of approximately 49.35 years and a median of 54 years.
Their age distribution is broader, with a significant concentration in the older age bracket.
Unemployed:

Unemployed leads range in age from 32 to 63 years, with a mean age of approximately 50.14 years and a median of 54 years.
Similar to professionals, this group is older, with a comparable age distribution, suggesting they are experienced individuals looking for new opportunities.
Key Findings:

Clear Age Segmentation: The age variable clearly differentiates the 'Student' segment from 'Professional' and 'Unemployed' segments. Students are significantly younger.
Professionals and Unemployed are Older: Both professionals and unemployed individuals tend to be older, with their mean and median ages being very similar and in the late 40s to mid-50s.
Age Overlap for Older Groups: There is significant overlap in the age ranges of professionals and unemployed leads, indicating that these groups share similar demographic profiles in terms of age.
Business Insight: ExtraaLearn needs to tailor its messaging and course offerings based on these distinct age-occupation profiles:

For Students: Focus on entry-level skills, career foundational courses, and programs that aid in securing their first job or enhancing their academic profile.
For Professionals and Unemployed: Marketing should emphasize upskilling, reskilling for career changes, leadership development, and advanced certifications that lead to promotions or new job opportunities. The similar age profiles suggest that programs targeting career growth and transition will appeal to both these groups.
The company's first interaction with leads should be compelling and persuasive. Let's see if the channels of the first interaction have an impact on the conversion of leads

stacked_barplot(data, "first_interaction", "status")
status                0     1   All
first_interaction                  
All                3235  1377  4612
Website            1383  1159  2542
Mobile App         1852   218  2070
------------------------------------------------------------------------------------------------------------------------
No description has been provided for this image
Observation on first_interaction vs. status
From the stacked bar plot and the cross-tabulation, we can observe the following:

Website as First Interaction:

Total Leads: 2542
Converted: 1159
Did Not Convert: 1383
Conversion Rate: Approximately 45.6% (1159 / 2542).
Mobile App as First Interaction:

Total Leads: 2070
Converted: 218
Did Not Convert: 1852
Conversion Rate: Approximately 10.5% (218 / 2070).
Key Findings:

Leads who first interact with ExtraaLearn through the Website have a significantly higher conversion rate (45.6%) compared to those who first interact through the Mobile App (10.5%).
While the Mobile App generates a substantial number of leads, its conversion efficiency is considerably lower.
Business Insight: This finding is crucial for optimizing lead acquisition strategies. The website appears to be a much more effective channel for attracting high-quality, conversion-prone leads. ExtraaLearn should:

Prioritize Website Experience: Continue to invest in and optimize the website's user experience, content, and lead capture mechanisms to capitalize on its high conversion potential.
Re-evaluate Mobile App Strategy: Investigate the reasons for the lower conversion rate from the mobile app. This could involve improving the app's onboarding process, clarifying value propositions, or addressing usability issues. It may also suggest that mobile app users are in an earlier stage of their decision-making process or have different intentions. Tailoring content or follow-up strategies specifically for mobile app leads might be beneficial.
distribution_plot_wrt_target(data, "time_spent_on_website", "status")
No description has been provided for this image
Observation:

Users who converted (Status = 1) spend significantly more time on the website compared to non-converted users (Status = 0).

The median time spent for converted users is much higher, indicating stronger interest and engagement.

Non-converted users generally spend less time browsing, with most activity concentrated at lower time values.

The distribution for converted users is shifted toward higher website engagement levels.

Although some non-converted users spend a long time on the website, overall higher time spent is strongly associated with conversion. Business Insight

Time spent on the website is one of the strongest indicators of lead quality. Users who spend more time exploring courses and content are more likely to convert into customers.

Recommendation

Prioritize leads with high website engagement.
Trigger follow-up calls or emails for users spending significant time on the platform.
Use time spent on website as a key component of the lead-scoring model.
# checking the median value
data.groupby(["status"])["time_spent_on_website"].median()
time_spent_on_website
status	
0	317.00000
1	789.00000

dtype: float64
Observation on time_spent_on_website vs. status
From the distribution plots, boxplots, and median values, we can observe the following:

Distribution for Non-Converted Leads (Status=0):

The distribution for non-converted leads shows a higher density at lower values of time_spent_on_website, indicating that a large portion of leads who did not convert spent relatively little time on the website.
The median time spent for non-converted leads is 317 seconds.
Distribution for Converted Leads (Status=1):

The distribution for converted leads shows a higher density at higher values of time_spent_on_website, indicating that leads who converted tended to spend significantly more time on the website.
The median time spent for converted leads is 789 seconds.
Boxplots:

The boxplot clearly shows that the median and overall distribution of time_spent_on_website are substantially higher for converted leads (Status=1) compared to non-converted leads (Status=0).
Converted leads also show a wider interquartile range (IQR) for time spent, suggesting more varied but generally higher engagement.
Key Findings:

There is a strong positive relationship between the time_spent_on_website and the likelihood of lead conversion.
Leads who spend more time exploring the ExtraaLearn website are significantly more likely to convert into paid customers.
The median time spent by converted leads is more than double that of non-converted leads.
Business Insight: time_spent_on_website is a critical indicator of lead quality and conversion potential. ExtraaLearn should:

Focus on Engagement: Implement strategies to increase the time leads spend on the website, such as improving content quality, offering interactive elements, personalized recommendations, or clear calls to action.
Lead Scoring: Integrate time_spent_on_website as a key feature in lead scoring models to prioritize follow-ups for highly engaged prospects.
Analyze Content Consumption: Understand what content or pages highly engaged, converted leads are interacting with the most to replicate successful engagement patterns and improve areas where non-converted leads drop off quickly.
Let's do a similar analysis for time spent on website and page views per visit.

distribution_plot_wrt_target(data, "website_visits", "status") # to plot distribution for website_visits and status
No description has been provided for this image
Observation on website_visits vs. status
From the distribution plots and boxplots, we can observe the following:

Distribution for Non-Converted Leads (Status=0):

The distribution for non-converted leads is highly skewed towards lower website_visits (1-3 visits), indicating that a significant portion of leads who did not convert visited the website infrequently.
The median number of website visits for non-converted leads is lower.
Distribution for Converted Leads (Status=1):

The distribution for converted leads is shifted towards a higher number of website_visits, with a noticeable density at 4-7 visits, and a longer tail extending to more frequent visits.
The median number of website visits for converted leads is higher.
Boxplots:

The boxplot clearly shows that converted leads (Status=1) generally have a higher median and interquartile range for website_visits compared to non-converted leads (Status=0).
Converted leads also show a higher concentration of outliers on the upper end, suggesting that very frequent visitors are more likely to convert.
Key Findings:

There is a positive correlation between the number of website_visits and the likelihood of lead conversion.
Leads who visit the ExtraaLearn website more frequently are more likely to convert into paid customers.
Highly engaged leads, characterized by a large number of website visits, are predominantly found among the converted group.
Business Insight: website_visits is a valuable indicator of a lead's interest and potential for conversion. ExtraaLearn should:

Monitor and Incentivize Repeat Visits: Track lead behavior and identify patterns that lead to repeat visits. Consider strategies to encourage more website visits for less engaged leads (e.g., targeted content, personalized recommendations, retargeting campaigns).
Lead Scoring: Incorporate website_visits into the lead scoring model to prioritize leads with higher engagement for sales follow-up.
Analyze High-Frequency Visitor Journeys: Study the journey of leads with a high number of website visits to understand what content, features, or interactions drive their sustained interest and ultimately lead to conversion.
distribution_plot_wrt_target(data,"page_views_per_visit" , "status") # to plot distribution for page_views_per_visit and status
No description has been provided for this image
Observation on page_views_per_visit vs. status
From the distribution plots and boxplots, we can observe the following:

Distribution for Non-Converted Leads (Status=0):

The distribution for non-converted leads is concentrated at lower page_views_per_visit values, typically between 1 and 3 pages.
This indicates that leads who did not convert generally viewed fewer pages per visit.
Distribution for Converted Leads (Status=1):

The distribution for converted leads is shifted towards higher page_views_per_visit values, showing a greater density between 3 and 6 pages, with a notable tail extending to more pages.
This suggests that leads who converted tended to explore more pages during each website visit.
Boxplots:

The boxplot clearly illustrates that converted leads (Status=1) have a higher median and a wider interquartile range (IQR) for page_views_per_visit compared to non-converted leads (Status=0).
The presence of numerous outliers on the higher end for converted leads indicates that those who view a significantly larger number of pages per visit are more likely to convert.
Key Findings:

There is a strong positive relationship between the number of page_views_per_visit and the likelihood of lead conversion.
Leads who explore more pages during their visits to the ExtraaLearn website are significantly more likely to convert into paid customers.
High page-view activity per visit is a characteristic of engaged and high-potential leads.
Business Insight: page_views_per_visit is an important metric for gauging lead engagement and conversion potential. ExtraaLearn should:

Optimize Website Content and Navigation: Ensure that the website has compelling content and intuitive navigation to encourage leads to view more pages per visit. This could involve clear links to program details, testimonials, FAQs, and application processes.
Content Personalization: Implement strategies to personalize content delivery based on initial page views, guiding leads to relevant information that encourages deeper exploration.
Lead Scoring Enhancement: Utilize page_views_per_visit as a significant factor in lead scoring, prioritizing follow-ups for leads who demonstrate higher levels of page exploration.
Analyze User Journeys: Study the typical page view paths of converted leads to identify which content or information is most effective in driving deeper engagement and, ultimately, conversion.
People browsing the website or the mobile app are generally required to create a profile by sharing their personal details before they can access more information. Let's see if the profile completion level has an impact on lead status

stacked_barplot(data,"profile_completed", "status")  # to plot stacked_barplot for profile_completed and status
status                0     1   All
profile_completed                  
All                3235  1377  4612
High               1318   946  2264
Medium             1818   423  2241
Low                  99     8   107
------------------------------------------------------------------------------------------------------------------------
No description has been provided for this image
Observation on profile_completed vs. status
From the stacked bar plot and the cross-tabulation, we can observe the following:

High Profile Completion Leads:

Total High Completion: 2264
Converted: 946
Did Not Convert: 1318
Conversion Rate: Approximately 41.78% (946 / 2264).
Medium Profile Completion Leads:

Total Medium Completion: 2241
Converted: 423
Did Not Convert: 1818
Conversion Rate: Approximately 18.88% (423 / 2241).
Low Profile Completion Leads:

Total Low Completion: 107
Converted: 8
Did Not Convert: 99
Conversion Rate: Approximately 7.48% (8 / 107).
Key Findings:

Leads with High profile completion have a significantly higher conversion rate (41.78%) compared to Medium (18.88%) and Low (7.48%) completion levels.
There's a clear positive correlation: as the level of profile completion increases, the likelihood of conversion also increases substantially.
Leads with Low profile completion have a very low conversion rate, indicating minimal interest or engagement.
Business Insight: profile_completed is a very strong indicator of a lead's intent and potential for conversion. ExtraaLearn should:

Prioritize High Completion Leads: Sales and marketing efforts should heavily prioritize leads with High profile completion, as they are the most likely to convert.
Nurture Medium Completion Leads: Implement strategies to encourage Medium completion leads to complete their profiles further, as this could significantly boost their conversion potential. This might involve personalized emails, in-app prompts, or incentives.
Re-evaluate Low Completion Leads: Leads with Low profile completion are unlikely to convert. Resources spent on actively pursuing these leads might be better reallocated to higher-potential segments. Consider automated, low-cost nurturing or re-engagement campaigns for this group, rather than direct sales efforts.
Simplify Profile Completion (if applicable): While high completion is good, review the profile completion process to ensure it's not overly burdensome, as this could deter potentially interested leads from reaching higher engagement levels.
After a lead shares their information by creating a profile, there may be interactions between the lead and the company to proceed with the process of enrollment. Let's see how the last activity impacts lead conversion status

stacked_barplot(data,"last_activity" , "status") # code to plot stacked_barplot for last_activity and status
status               0     1   All
last_activity                     
All               3235  1377  4612
Email Activity    1587   691  2278
Website Activity   677   423  1100
Phone Activity     971   263  1234
------------------------------------------------------------------------------------------------------------------------
No description has been provided for this image
Observation on last_activity vs. status
From the stacked bar plot and the cross-tabulation, we can observe the following:

Email Activity:

Total Leads: 2278
Converted: 691
Did Not Convert: 1587
Conversion Rate: Approximately 30.33% (691 / 2278).
Website Activity:

Total Leads: 1100
Converted: 423
Did Not Convert: 677
Conversion Rate: Approximately 38.45% (423 / 1100).
Phone Activity:

Total Leads: 1234
Converted: 263
Did Not Convert: 971
Conversion Rate: Approximately 21.31% (263 / 1234).
Key Findings:

Leads whose last_activity was Website Activity have the highest conversion rate (38.45%).
Leads with Email Activity as their last interaction have a moderate conversion rate (30.33%).
Leads with Phone Activity as their last interaction have the lowest conversion rate (21.31%).
While Email Activity is the most frequent last interaction overall, Website Activity leads to a higher conversion probability when it's the last recorded action.
Business Insight: last_activity is a strong indicator of engagement quality. ExtraaLearn should:

Encourage Website Activity: Given the higher conversion rate, encouraging leads to return to and engage with the website as their final activity before conversion is crucial. This could involve calls-to-action in emails or during phone calls that direct them to specific web pages for enrollment or deeper exploration.
Optimize Email Nurturing: Although Email Activity has a good conversion rate, continuous optimization of email content and timing can further improve its effectiveness. Ensure emails drive recipients towards conversion-oriented actions, potentially on the website.
Re-evaluate Phone Activity Strategy: The lowest conversion rate for Phone Activity suggests that these interactions might not be as effective in closing deals or are happening with leads who are already less likely to convert. This warrants further investigation into the nature of these phone calls, the leads involved, and the training of representatives to improve conversion efficiency through this channel. It could be that phone activity is more effective for initial qualification rather than final conversion.
Let's see how advertisement and referrals impact the lead status

stacked_barplot(data, "print_media_type1" , "status" )  # to plot stacked_barplot for print_media_type1 and status
status                0     1   All
print_media_type1                  
All                3235  1377  4612
No                 2897  1218  4115
Yes                 338   159   497
------------------------------------------------------------------------------------------------------------------------
No description has been provided for this image
Observation on print_media_type1 vs. status
From the stacked bar plot and the cross-tabulation, we can observe the following:

Leads Not Exposed to Print Media Type 1:

Total Leads: 4115
Converted: 1218
Did Not Convert: 2897
Conversion Rate: Approximately 29.59% (1218 / 4115).
Leads Exposed to Print Media Type 1:

Total Leads: 497
Converted: 159
Did Not Convert: 338
Conversion Rate: Approximately 32.00% (159 / 497).
Key Findings:

A very small percentage of leads (approximately 10.8%) were exposed to print_media_type1 (newspaper advertisements).
Leads exposed to print_media_type1 have a slightly higher conversion rate (32.00%) compared to those not exposed (29.59%).
However, the absolute number of converted leads from this channel is low (159).
Business Insight: Similar to print_media_type2, print_media_type1 also has a very limited reach in terms of lead generation. While there's a marginal uplift in conversion rate for those exposed, the overall volume is low. ExtraaLearn should consider the cost-effectiveness of this advertising channel. If the cost of acquiring these slightly higher-converting leads through print_media_type1 outweighs the benefit, resources might be better reallocated to channels with higher volume and/or more significant conversion rate differences.

stacked_barplot (data, "print_media_type2" , "status")  # to plot stacked_barplot for print_media_type2 and status
status                0     1   All
print_media_type2                  
All                3235  1377  4612
No                 3077  1302  4379
Yes                 158    75   233
------------------------------------------------------------------------------------------------------------------------
No description has been provided for this image
Observation on print_media_type2 vs. status
From the stacked bar plot and the cross-tabulation, we can observe the following:

Low Volume: Only a very small fraction of leads (233 out of 4612) were acquired through print_media_type2 (magazine advertisements), confirming its low reach as identified in the univariate analysis.
Conversion Rate for 'No': Out of the 4379 leads who did not see print_media_type2 ads, 1302 converted, resulting in a conversion rate of approximately 29.73% (1302 / 4379).
Conversion Rate for 'Yes': Out of the 233 leads who did see print_media_type2 ads, 75 converted, resulting in a conversion rate of approximately 32.19% (75 / 233).
Slightly Higher Conversion: Leads exposed to print_media_type2 show a slightly higher conversion rate compared to those who were not. However, due to the extremely small sample size for 'Yes' (only 233 leads), this difference might not be statistically significant.
Business Insight: Despite a marginally higher conversion rate for leads exposed to print_media_type2, the overall volume of leads generated from this channel is very low. This suggests that while it might attract slightly more interested prospects, it is not a major or efficient lead generation channel for ExtraaLearn. Resources might be better allocated to channels with higher volume and/or more significant conversion rate differences.

stacked_barplot(data, "digital_media" , "status") # code to plot stacked_barplot for digital_media and status
status            0     1   All
digital_media                  
All            3235  1377  4612
No             2876  1209  4085
Yes             359   168   527
------------------------------------------------------------------------------------------------------------------------
No description has been provided for this image
Observation on digital_media vs. status
From the stacked bar plot and the cross-tabulation, we can observe the following:

Leads Not Exposed to Digital Media:

Total Leads: 4085
Converted: 1209
Did Not Convert: 2876
Conversion Rate: Approximately 29.59% (1209 / 4085).
Leads Exposed to Digital Media:

Total Leads: 527
Converted: 168
Did Not Convert: 359
Conversion Rate: Approximately 31.88% (168 / 527).
Key Findings:

A relatively small portion of leads (approximately 11.4%) were exposed to digital_media advertisements, consistent with the univariate analysis.
Leads exposed to digital_media have a slightly higher conversion rate (31.88%) compared to those not exposed (29.59%).
The absolute number of converted leads from this channel is low (168).
Business Insight: Similar to print media, digital media advertising appears to have a limited reach for ExtraaLearn. While there's a marginal uplift in the conversion rate for leads exposed to digital media, the overall volume of leads and conversions from this channel is not substantial. ExtraaLearn should evaluate the cost-effectiveness of its digital media campaigns. If the cost of acquiring these slightly higher-converting leads through digital media is high, resources might be more effectively allocated to channels that generate higher volumes of leads or show more significant conversion rate improvements.

stacked_barplot(data, "educational_channels" , "status")  # code to plot stacked_barplot for educational_channels and status
status                   0     1   All
educational_channels                  
All                   3235  1377  4612
No                    2727  1180  3907
Yes                    508   197   705
------------------------------------------------------------------------------------------------------------------------
No description has been provided for this image
Observation on educational_channels vs. status
From the stacked bar plot and the cross-tabulation, we can observe the following:

Leads Not Exposed to Educational Channels:

Total Leads: 3907
Converted: 1180
Did Not Convert: 2727
Conversion Rate: Approximately 30.20% (1180 / 3907).
Leads Exposed to Educational Channels:

Total Leads: 705
Converted: 197
Did Not Convert: 508
Conversion Rate: Approximately 27.94% (197 / 705).
Key Findings:

Approximately 15.3% of leads came through educational_channels.
Surprisingly, leads who were not exposed to educational channels have a slightly higher conversion rate (30.20%) compared to those who were exposed (27.94%).
This difference is relatively small, but it suggests that exposure to educational channels does not directly lead to a higher conversion rate for ExtraaLearn.
Business Insight: This finding is counter-intuitive, as one might expect educational channels to bring in more qualified leads. ExtraaLearn should investigate this further:

Content Alignment: Are the messages or content shared on these educational channels effectively aligning with ExtraaLearn's offerings and target audience?
Lead Quality vs. Volume: While educational channels bring in a decent volume of leads, their conversion efficiency appears to be slightly lower. This suggests that these channels might be attracting leads who are in an earlier stage of consideration or are simply gathering information without immediate intent to enroll.
Optimize Messaging: Re-evaluate the messaging and calls-to-action on educational channels to ensure they resonate with the specific needs and conversion drivers of the audience found there. It might be necessary to adjust expectations for this channel or integrate it into a longer nurturing cycle.
stacked_barplot(data, "referral", "status" ) # code to plot stacked_barplot for referral and status
status       0     1   All
referral                  
All       3235  1377  4612
No        3205  1314  4519
Yes         30    63    93
------------------------------------------------------------------------------------------------------------------------
No description has been provided for this image
Observation on referral vs. status
From the stacked bar plot and the cross-tabulation, we can observe the following:

Leads Not from Referral:

Total Leads: 4519
Converted: 1314
Did Not Convert: 3205
Conversion Rate: Approximately 29.08% (1314 / 4519).
Leads from Referral:

Total Leads: 93
Converted: 63
Did Not Convert: 30
Conversion Rate: Approximately 67.74% (63 / 93).
Key Findings:

Only a very small proportion of leads (approximately 2.0%) are acquired through referrals.
However, leads from referrals have an exceptionally high conversion rate (67.74%), which is more than double the conversion rate of non-referred leads (29.08%).
This indicates that while the volume is low, the quality of referred leads is outstanding.
Business Insight: referrals are a highly effective, albeit low-volume, source of high-quality leads for ExtraaLearn. This insight is critical for optimizing lead generation and resource allocation:

Invest in Referral Programs: Despite the low volume, the extremely high conversion rate for referred leads justifies significant investment in developing and promoting robust referral programs. This could include incentives for referrers and streamlined processes for referred leads.
Leverage Trust: Referred leads likely come with a pre-existing level of trust and interest, making them easier to convert. The sales team should prioritize follow-ups with referred leads.
Analyze Referral Sources: Investigate who is referring leads and why. Understanding the motivations and demographics of referrers could help in expanding the referral network.
Focus on Advocacy: Encourage existing satisfied customers to become advocates and refer new leads, as this channel has proven to be highly successful in converting prospects.
Observation from Bivariate Analysis:
Overall Observations from Bivariate Analysis
The bivariate analysis, examining the relationship between various features and the lead conversion status, has yielded several critical insights:

Impact of Demographic and Interaction Channels:
Current Occupation: Professionals have the highest conversion rate (35.5%), followed by Unemployed (26.6%), and Students (11.7%) have the lowest. This indicates that ExtraaLearn's programs resonate most with career-focused individuals.
Age vs. Current Occupation: Age strongly differentiates students (mean ~21 years) from professionals (mean ~49 years) and unemployed individuals (mean ~50 years). This suggests that messaging and program offerings should be tailored to these distinct age-occupation profiles, focusing on career advancement for older groups and foundational skills for younger students.
First Interaction: Website interactions lead to a significantly higher conversion rate (45.6%) compared to Mobile App interactions (10.5%). This highlights the website as a more effective initial touchpoint for conversion, warranting prioritization of website experience and a re-evaluation of the mobile app strategy.
Impact of Lead Engagement Metrics:
Time Spent on Website: Converted leads spend significantly more time on the website (median 789 seconds) compared to non-converted leads (median 317 seconds). This is a strong indicator of intent, suggesting that strategies to increase on-site engagement could boost conversion.
Website Visits: Converted leads exhibit a higher number of website visits, with more frequent visitors being more likely to convert. Encouraging repeat visits and integrating this metric into lead scoring can be beneficial.
Page Views per Visit: Similarly, converted leads tend to view more pages per visit, indicating deeper exploration and stronger interest. Optimizing content and navigation to encourage more page views per visit can improve lead quality.
Profile Completed: Leads with High profile completion have a significantly higher conversion rate (41.78%) than Medium (18.88%) or Low (7.48%) completion levels. This is a very strong indicator of intent and suggests prioritizing leads with higher completion rates.
Last Activity: Leads whose last activity was 'Website Activity' have the highest conversion rate (38.45%), followed by 'Email Activity' (30.33%), while 'Phone Activity' has the lowest (21.31%). This implies that driving leads back to the website for their final interactions could be a powerful conversion strategy.
Impact of Marketing Channels:
Print Media (Type1 & Type2) and Digital Media: These channels generally contribute a small volume of leads, with only marginal or no significant uplift in conversion rates compared to leads not exposed to them. The cost-effectiveness of these channels should be carefully reviewed.
Educational Channels: Surprisingly, leads not exposed to educational channels have a slightly higher conversion rate (30.20%) than those who were (27.94%). This counter-intuitive finding requires further investigation into content alignment and lead quality from these sources.
Referral: Despite being a very low-volume source (2.0% of leads), referrals boast an exceptionally high conversion rate (67.74%). This makes referral programs a highly valuable, albeit niche, channel for high-quality lead generation.
Overall Business Implications:
The analysis consistently points to engagement metrics (time spent, visits, page views, profile completion) and certain interaction channels (website, referrals) as powerful predictors of lead conversion. ExtraaLearn should:

Prioritize and Optimize High-Converting Channels: Intensify efforts on the website experience and explore expanding referral programs.
Enhance Lead Scoring: Develop a robust lead scoring model that heavily weights engagement metrics (e.g., higher time spent, more visits, greater page views, high profile completion) to prioritize sales efforts.
Tailor Marketing and Nurturing: Customize messaging and content based on lead demographics (e.g., occupation, age) and last activity. Encourage leads to engage with the website as a final step.
Re-evaluate Low-Impact Channels: Investigate the effectiveness and ROI of channels like mobile app, phone activity, and various media types, considering reallocation of resources to more impactful strategies.
Address Class Imbalance: Be mindful of the class imbalance (70.1% non-converted vs. 29.9% converted) during the upcoming model development phase to ensure the model effectively identifies positive conversions without bias.
Data Preprocessing
Missing value treatment (if needed)
Feature engineering (if needed)
Outlier detection and treatment (if needed)
Preparing data for modeling
Any other preprocessing steps (if needed)
1. Missing value treatment.
data.isnull().sum()
0
ID	0
age	0
current_occupation	0
first_interaction	0
profile_completed	0
website_visits	0
time_spent_on_website	0
page_views_per_visit	0
last_activity	0
print_media_type1	0
print_media_type2	0
digital_media	0
educational_channels	0
referral	0
status	0

dtype: int64
Missing Values Summary

No missing values were found in the dataset.

All 15 features contain complete data.
No imputation techniques (mean, median, mode, or advanced methods) are required.
The dataset is clean and ready for further preprocessing, feature engineering, and model building.
Observation

The absence of missing values is a positive sign of data quality, reducing the risk of bias and information loss during analysis. This allows the modeling process to focus on feature selection, outlier treatment, encoding, and model optimization rather than handling incomplete records.

Conclusion

Data Completeness = 100% No missing value treatment is needed before proceeding with machine learning modeling.

data.drop_duplicates(inplace=True)
Observation:

No duplicate data found.

# Check missing value percentage
missing_percentage = (data.isnull().sum()/len(data))*100

pd.DataFrame({
    'Missing Count': data.isnull().sum(),
    'Missing %': missing_percentage
}).sort_values('Missing %', ascending=False)
Missing Count	Missing %
ID	0	0.00000
age	0	0.00000
current_occupation	0	0.00000
first_interaction	0	0.00000
profile_completed	0	0.00000
website_visits	0	0.00000
time_spent_on_website	0	0.00000
page_views_per_visit	0	0.00000
last_activity	0	0.00000
print_media_type1	0	0.00000
print_media_type2	0	0.00000
digital_media	0	0.00000
educational_channels	0	0.00000
referral	0	0.00000
status	0	0.00000
Observation:
No duplicated data is seen, duplicate leads can bias model performance. A missing value analysis was performed on all variables in the dataset. The results showed that every feature contains 4,612 non-null observations, indicating that there are no missing values in the dataset. Therefore, no imputation or missing-value treatment was required prior to model development. This ensures that all available lead information can be utilized during the modeling process without introducing bias from missing data handling techniques.

2. Feature engineering:
Since the ExtraaLearn dataset is already clean (no missing values) and has meaningful behavioral features, Feature Engineering should focus on creating variables that better capture lead engagement and conversion intent.

Fixing Data Preprocessing and Feature Engineering
# Re-initialize data from original learn to ensure a clean slate
data = learn.copy()

# Drop 'ID' column as it's an identifier and not a feature
data = data.drop(columns=["ID"], errors='ignore')

# 1. Create Profile Completion Score
profile_map = {'Low': 1, 'Medium': 2, 'High': 3}
data['profile_score'] = data['profile_completed'].map(profile_map)

# 2. Create Occupation Score
occupation_map = {'Student': 1, 'Unemployed': 2, 'Professional': 3}
data['occupation_score'] = data['current_occupation'].map(occupation_map)

# 3. Convert Marketing Channels to Binary (and ensure no NaNs)
marketing_cols = [
    'print_media_type1',
    'print_media_type2',
    'digital_media',
    'educational_channels',
    'referral'
]

for col in marketing_cols:
    # Ensure all values are treated as strings and replace 'No' with 0 and 'Yes' with 1
    # Any other value or NaN will become NaN and then be filled with 0.
    data[col] = data[col].astype(str).replace({'No': 0, 'Yes': 1}).fillna(0).astype(int)

# 4. Marketing Exposure Score (will now be calculated on 0/1 values)
data['marketing_exposure_score'] = data[marketing_cols].sum(axis=1)

# 5. Create Age Groups
data['age_group'] = pd.cut(
    data['age'],
    bins=[17, 30, 45, 60, 65],
    labels=['Young Adult', 'Mid Career', 'Experienced', 'Senior Professional']
)

# 6. Normalize Engagement Variables
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

engagement_features = [
    'website_visits',
    'time_spent_on_website',
    'page_views_per_visit'
]

data[['website_visits_scaled',
      'time_spent_scaled',
      'page_views_scaled']] = scaler.fit_transform(
    data[engagement_features]
)

# 7. Engagement Score
import numpy as np

data['engagement_score'] = (
    data['website_visits_scaled'] +
    data['time_spent_scaled'] +
    data['page_views_scaled']
)

# 8. High Engagement Flag
data['high_engagement'] = np.where(
    data['engagement_score'] >
    data['engagement_score'].median(),
    1,
    0
)

# 9. Engagement Category
data['engagement_category'] = pd.qcut(
    data['engagement_score'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

# 10. Lead Source Type
data['source_type'] = np.where(
    data['first_interaction'] == 'Website',
    'Web Lead',
    'App Lead'
)

# Define X and Y from the processed data
X = data.drop(["status"], axis=1) # 'ID' column already dropped
Y = data['status'] # Define the dependent (target) variable

# Drop original columns that are now replaced by engineered features or are scaled versions
X = X.drop(columns=[
    'profile_completed', # Original column, now replaced by 'profile_score'
    'current_occupation', # Original column, now replaced by 'occupation_score'
    'first_interaction', # Original column, now replaced by 'source_type'
    'website_visits_scaled', # Scaled version of original for engagement_score
    'time_spent_scaled', # Scaled version of original for engagement_score
    'page_views_scaled', # Scaled version of original for engagement_score
    'website_visits', # Original numerical feature (retained in engagement score logic)
    'time_spent_on_website', # Original numerical feature (retained in engagement score logic)
    'page_views_per_visit' # Original numerical feature (retained in engagement score logic)
], errors='ignore')


# Get dummies for remaining categorical features in X
X = pd.get_dummies(X, drop_first=True)

# Splitting the data in 70:30 ratio for train to test data
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=1
)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")
print("Percentage of classes in training set:")
print(y_train.value_counts(normalize=True))
print("Percentage of classes in test set:")
print(y_test.value_counts(normalize=True))
Shape of X_train: (3228, 19)
Shape of y_train: (3228,)
Shape of X_test: (1384, 19)
Shape of y_test: (1384,)
Percentage of classes in training set:
status
0   0.70415
1   0.29585
Name: proportion, dtype: float64
Percentage of classes in test set:
status
0   0.69509
1   0.30491
Name: proportion, dtype: float64
Observation:

The dataset was split into 70% training (3,228 records) and 30% testing (1,384 records), which is suitable for model development.
The target variable (status) has a similar distribution in both datasets, indicating a successful stratified split.
Around 70% of leads are non-converted and 30% are converted, showing a moderate class imbalance.
The data is well-prepared for building and evaluating classification models, with reliable train-test representation.
Check Distribution of Engineered Features
for col in [
    'profile_score',
    'occupation_score',
    'marketing_exposure_score',
    'age_group',
    'high_engagement',
    'engagement_category',
    'source_type'
]:
    print(f"\n{'='*50}")
    print(col)
    print(data[col].value_counts(normalize=True))
==================================================
profile_score
profile_score
3   0.49089
2   0.48591
1   0.02320
Name: proportion, dtype: float64

==================================================
occupation_score
occupation_score
3   0.56722
2   0.31245
1   0.12034
Name: proportion, dtype: float64

==================================================
marketing_exposure_score
marketing_exposure_score
0   0.62511
1   0.31266
2   0.05442
3   0.00716
4   0.00065
Name: proportion, dtype: float64

==================================================
age_group
age_group
Experienced           0.57069
Mid Career            0.24718
Young Adult           0.15330
Senior Professional   0.02884
Name: proportion, dtype: float64

==================================================
high_engagement
high_engagement
1   0.50000
0   0.50000
Name: proportion, dtype: float64

==================================================
engagement_category
engagement_category
Low      0.33348
High     0.33348
Medium   0.33304
Name: proportion, dtype: float64

==================================================
source_type
source_type
Web Lead   0.55117
App Lead   0.44883
Name: proportion, dtype: float64
Observation:

Profile Completion: Nearly all leads have Medium (48.6%) or High (49.1%) profile completion, indicating good user participation.
Occupation: The majority of leads are Professionals (56.7%), followed by Unemployed users (31.2%), suggesting a strong career-focused audience.
Marketing Exposure: Most leads (62.5%) were exposed to no marketing channels, while only a small proportion interacted with multiple channels.
Age Group: The largest segment is Experienced professionals (57.1%), followed by Mid-Career users (24.7%).
Engagement: High and low engagement users are evenly distributed (50% each), and engagement categories are balanced by design (~33% each).
Lead Source: More leads originate from the Website (55.1%) than the Mobile App (44.9%).
Key Insight

ExtraaLearn's lead base is dominated by experienced professionals with medium-to-high profile completion, and the website is the primary source of lead generation, making these important areas for conversion-focused strategies.

Verify New Features Created
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4612 entries, 0 to 4611
Data columns (total 25 columns):
 #   Column                    Non-Null Count  Dtype   
---  ------                    --------------  -----   
 0   age                       4612 non-null   int64   
 1   current_occupation        4612 non-null   object  
 2   first_interaction         4612 non-null   object  
 3   profile_completed         4612 non-null   object  
 4   website_visits            4612 non-null   int64   
 5   time_spent_on_website     4612 non-null   int64   
 6   page_views_per_visit      4612 non-null   float64 
 7   last_activity             4612 non-null   object  
 8   print_media_type1         4612 non-null   int64   
 9   print_media_type2         4612 non-null   int64   
 10  digital_media             4612 non-null   int64   
 11  educational_channels      4612 non-null   int64   
 12  referral                  4612 non-null   int64   
 13  status                    4612 non-null   int64   
 14  profile_score             4612 non-null   int64   
 15  occupation_score          4612 non-null   int64   
 16  marketing_exposure_score  4612 non-null   int64   
 17  age_group                 4612 non-null   category
 18  website_visits_scaled     4612 non-null   float64 
 19  time_spent_scaled         4612 non-null   float64 
 20  page_views_scaled         4612 non-null   float64 
 21  engagement_score          4612 non-null   float64 
 22  high_engagement           4612 non-null   int64   
 23  engagement_category       4612 non-null   category
 24  source_type               4612 non-null   object  
dtypes: category(2), float64(5), int64(13), object(5)
memory usage: 838.2+ KB
Observation on Final Feature-Engineered Dataset

The dataset contains 4,612 leads and 25 features, indicating that additional features were successfully created through feature engineering.
There are no missing values across any of the variables, ensuring high data quality for modeling.
The dataset includes a mix of:
13 numerical features (engagement metrics, scores, binary indicators)
5 categorical features (occupation, interaction channels, activity types)
2 engineered categorical features (age_group, engagement_category)
5 continuous/scaled features used for model building.
New engineered features such as profile_score, occupation_score, marketing_exposure_score, engagement_score, high_engagement, age_group, and source_type provide additional business insights and improve predictive capability.
Binary marketing variables have been converted into numeric format, making them suitable for machine learning algorithms.
Scaled engagement variables ensure that features with different ranges contribute fairly during model training.
Key Insight

The dataset is now fully prepared for predictive modeling, with engineered features capturing lead demographics, engagement behavior, marketing exposure, and acquisition channels. These enhancements are expected to improve model performance and provide deeper insights into the factors driving lead conversion.

Check Distribution of Engineered Features
for col in [
    'profile_score',
    'occupation_score',
    'marketing_exposure_score',
    'age_group',
    'high_engagement',
    'engagement_category',
    'source_type'
]:
    print(f"\n{'='*50}")
    print(col)
    print(data[col].value_counts(normalize=True))
==================================================
profile_score
profile_score
3   0.49089
2   0.48591
1   0.02320
Name: proportion, dtype: float64

==================================================
occupation_score
occupation_score
3   0.56722
2   0.31245
1   0.12034
Name: proportion, dtype: float64

==================================================
marketing_exposure_score
marketing_exposure_score
0   0.62511
1   0.31266
2   0.05442
3   0.00716
4   0.00065
Name: proportion, dtype: float64

==================================================
age_group
age_group
Experienced           0.57069
Mid Career            0.24718
Young Adult           0.15330
Senior Professional   0.02884
Name: proportion, dtype: float64

==================================================
high_engagement
high_engagement
1   0.50000
0   0.50000
Name: proportion, dtype: float64

==================================================
engagement_category
engagement_category
Low      0.33348
High     0.33348
Medium   0.33304
Name: proportion, dtype: float64

==================================================
source_type
source_type
Web Lead   0.55117
App Lead   0.44883
Name: proportion, dtype: float64
Observation of Engineered Features:

Profile Score: Nearly 98% of leads have Medium or High profile completion, indicating strong user engagement and willingness to share information.
Occupation Score: Over 56% of leads are professionals, confirming that ExtraaLearn primarily attracts working individuals seeking career growth.
Marketing Exposure Score: About 63% of leads were not exposed to any tracked marketing channel, suggesting opportunities to improve marketing reach and multi-channel engagement.
Age Group: The majority of leads (57%) belong to the Experienced category, followed by Mid-Career professionals (25%), highlighting a mature learner base.
High Engagement: The dataset is evenly split between high- and low-engagement leads, making engagement a useful feature for lead segmentation.
Engagement Category: Leads are evenly distributed across Low, Medium, and High engagement groups, supporting effective lead scoring and comparison.
Source Type: More than half of the leads (55%) originate from the Website, reinforcing its importance as the primary lead acquisition channel.
Key Business Insight

ExtraaLearn's typical lead is an experienced professional with medium-to-high profile completion who primarily engages through the website. These characteristics align closely with the profiles that were later identified as having a higher likelihood of conversion.

This will ensure all engineered features are created directly in your data dataframe and are available for subsequent encoding, train-test split, and model building.

# 1. Create Profile Completion Score
profile_map = {'Low': 1, 'Medium': 2, 'High': 3}
data['profile_score'] = data['profile_completed'].map(profile_map)

# 2. Create Occupation Score
occupation_map = {'Student': 1, 'Unemployed': 2, 'Professional': 3}
data['occupation_score'] = data['current_occupation'].map(occupation_map)

# 3. Convert Marketing Channels to Binary
marketing_cols = [
    'print_media_type1',
    'print_media_type2',
    'digital_media',
    'educational_channels',
    'referral'
]

for col in marketing_cols:
    data[col] = data[col].map({'No': 0, 'Yes': 1})

# 4. Marketing Exposure Score
data['marketing_exposure_score'] = data[marketing_cols].sum(axis=1)

# 5. Create Age Groups
data['age_group'] = pd.cut(
    data['age'],
    bins=[17, 30, 45, 60, 65],
    labels=['Young Adult', 'Mid Career', 'Experienced', 'Senior Professional']
)

# 6. Normalize Engagement Variables
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

engagement_features = [
    'website_visits',
    'time_spent_on_website',
    'page_views_per_visit'
]

data[['website_visits_scaled',
      'time_spent_scaled',
      'page_views_scaled']] = scaler.fit_transform(
    data[engagement_features]
)

# 7. Engagement Score
data['engagement_score'] = (
    data['website_visits_scaled'] +
    data['time_spent_scaled'] +
    data['page_views_scaled']
)

# 8. High Engagement Flag
import numpy as np

data['high_engagement'] = np.where(
    data['engagement_score'] >
    data['engagement_score'].median(),
    1,
    0
)

# 9. Engagement Category
data['engagement_category'] = pd.qcut(
    data['engagement_score'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

# 10. Lead Source Type
data['source_type'] = np.where(
    data['first_interaction'] == 'Website',
    'Web Lead',
    'App Lead'
)

# Display newly created features
data[[
    'profile_score',
    'occupation_score',
    'marketing_exposure_score',
    'age_group',
    'engagement_score',
    'high_engagement',
    'engagement_category',
    'source_type'
]].head()
profile_score	occupation_score	marketing_exposure_score	age_group	engagement_score	high_engagement	engagement_category	source_type
0	3	2	0.00000	Experienced	0.98033	1	High	Web Lead
1	2	3	0.00000	Experienced	0.11674	0	Low	App Lead
2	2	3	0.00000	Experienced	0.23409	0	Low	Web Lead
3	3	2	0.00000	Experienced	0.42781	0	Medium	Web Lead
4	3	1	0.00000	Young Adult	1.28738	1	High	Web Lead
Observations on Newly Engineered Features
The feature engineering process has successfully created several new variables, transforming existing categorical features into numerical scores and creating composite engagement metrics. The .head() output provides a snapshot of these new features for the first few leads:

profile_score:

This numerical feature is derived from profile_completed by mapping 'Low' to 1, 'Medium' to 2, and 'High' to 3.
Observation: For the first five leads, profile_score ranges from 2 to 3, indicating that these leads have 'Medium' or 'High' profile completion, which aligns with the original profile_completed values.
occupation_score:

This numerical feature is created by mapping current_occupation ('Student': 1, 'Unemployed': 2, 'Professional': 3).
Observation: The scores for the first five leads reflect their occupations; for example, 'Unemployed' leads have a score of 2, 'Professional' leads have 3, and 'Student' leads have 1.
marketing_exposure_score:

This score is the sum of binary indicators (0 or 1) for various marketing channels (print_media_type1, print_media_type2, digital_media, educational_channels, referral). A 'Yes' is mapped to 1 and 'No' to 0.
Observation: For the first five leads, the marketing_exposure_score is 0.00000, suggesting that these particular leads were not exposed to any of the tracked print or digital media advertisements, or educational channels, or referrals after the mapping within this specific feature engineering step. This indicates a potential issue as previous checks showed 'Yes' values for print_media_type1 and digital_media for some leads. This needs further investigation as the original print_media_type columns appear to have 0 non-null values after the code run, which might be an unintended side effect of re-running the feature engineering after data = learn.copy() which was not previously present before this cell was executed in the original state, indicating that these columns are being overwritten in an unexpected manner.
age_group:

This categorical feature bins the age into 'Young Adult', 'Mid Career', 'Experienced', and 'Senior Professional' categories.
Observation: The first few leads are categorized into 'Experienced' and 'Young Adult', reflecting their respective age values.
engagement_score:

This composite score is the sum of scaled website_visits, time_spent_on_website, and page_views_per_visit. The original values were first normalized using MinMaxScaler.
Observation: This score varies among the leads, with higher values indicating greater overall engagement. For instance, lead 0 has a high score of 0.98033, while lead 1 has a lower score of 0.11674, reflecting their raw engagement metrics.
high_engagement:

This is a binary flag (1 or 0) indicating whether a lead's engagement_score is above the median engagement score of the dataset.
Observation: The values show a mix of 0s and 1s, implying that some leads are above the median engagement and others are not, based on their engagement_score.
engagement_category:

This categorical feature divides the engagement_score into 'Low', 'Medium', and 'High' categories based on quantiles.
Observation: The first few leads are classified into 'High', 'Low', and 'Medium' categories, providing a discrete representation of their engagement level.
source_type:

This categorical feature classifies the lead as 'Web Lead' if first_interaction was 'Website', and 'App Lead' otherwise.
Observation: The values for the first five leads correctly reflect their first_interaction as either 'Web Lead' or 'App Lead'.
These engineered features are likely to add more predictive value than the raw variables alone and can be tested using feature importance after model training.

3. Outlier detection and treatment:
Visualize Outliers:
Before moving to modeling, let's perform Outlier Detection and Treatment on the numerical variables.

#  outlier detection using boxplot
# Using the original continuous numerical features for outlier detection
# These were defined earlier in `num_cols` cell (cZ5TMJ5v5uVM)
num_cols = [
    'age',
    'website_visits',
    'time_spent_on_website',
    'page_views_per_visit'
]

plt.figure(figsize=(15, 10))

for i, variable in enumerate(num_cols):
    plt.subplot(2, 2, i + 1) # Adjusted subplot grid to 2x2 for 4 variables
    sns.boxplot(data=data, x=variable, whis=1.5)
    plt.tight_layout()
    plt.title(variable)

plt.show()
No description has been provided for this image
Observation from Earlier Analysis:

Age: No significant outliers. Website Visits: Several high-end outliers (up to 30 visits). Time Spent on Website: Wide spread but appears naturally distributed. Page Views per Visit: Multiple high-end outliers.

Calculate Outlier percentage:

for col in num_cols:

    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[
        (data[col] < lower_bound) |
        (data[col] > upper_bound)
    ]

    print(f"{col}")
    print(f"Outliers: {len(outliers)}")
    print(f"Percentage: {(len(outliers)/len(data))*100:.2f}%")
    print("-"*50)
age
Outliers: 0
Percentage: 0.00%
--------------------------------------------------
website_visits
Outliers: 154
Percentage: 3.34%
--------------------------------------------------
time_spent_on_website
Outliers: 0
Percentage: 0.00%
--------------------------------------------------
page_views_per_visit
Outliers: 257
Percentage: 5.57%
--------------------------------------------------
Observation-

Based on the outlier percentage calculations and the distribution plots for the numerical features, here are the observations:

Age: As observed in the boxplots and confirmed by the calculation, age has 0% outliers. Its distribution is relatively consistent, suggesting no unusual values.

Website Visits: This feature has 3.34% outliers. The histogram for website_visits is right-skewed, with a long tail extending to higher values, which visually confirms the presence of these high-end outliers. These outliers represent leads with unusually high website visits.

Time Spent on Website: time_spent_on_website shows 0% outliers based on the IQR method. The histogram displays a bimodal, right-skewed distribution, indicating a wide spread of engagement but no extreme values that fall outside the defined outlier bounds for this metric.

Page Views per Visit: This feature has 5.57% outliers. The histogram for page_views_per_visit is also right-skewed, showing a concentration at lower values but a significant tail extending to much higher page views. These are high-end outliers, representing leads who view a large number of pages per visit.

Overall Conclusion for Outlier Treatment:

For website_visits and page_views_per_visit, the identified outliers are predominantly high-end values, representing highly engaged leads. These are not data errors but rather important indicators of strong interest. As such, these outliers are likely to contain valuable information for predicting lead conversion and should generally be retained rather than removed or treated, as they can significantly contribute to the model's ability to identify high-potential leads.

# Verify with dustribution
for col in num_cols:

    plt.figure(figsize=(8,4))

    sns.histplot(
        data[col],
        kde=True
    )

    plt.title(col)
    plt.show()
No description has been provided for this image
No description has been provided for this image
No description has been provided for this image
No description has been provided for this image
Observation:

Based on the distribution plots for age, website_visits, time_spent_on_website, and page_views_per_visit, here are my observations:

Age: The distribution for age shows a clear bimodal pattern, with a large peak around 55-60 years and a smaller peak around 30-35 years. It appears relatively well-behaved with no extreme values, confirming the earlier finding of no outliers.

Website Visits: This distribution is heavily right-skewed, with most leads having a low number of website visits (around 1-3). The long tail extending to higher values (up to 30 visits) clearly indicates the presence of high-end outliers, representing a small group of very frequent visitors.

Time Spent on Website: The histogram for time_spent_on_website displays a bimodal, right-skewed distribution. There's a significant concentration of leads spending relatively little time, and another distinct group spending a much longer duration (around 1500-2000 seconds). While widely spread, the distribution doesn't show isolated extreme outliers using the IQR method.

Page Views per Visit: Similar to website visits, this distribution is also right-skewed. Most leads view a small number of pages per visit (around 1-4). The long tail to the right indicates high-end outliers, signifying a subset of leads who explore a large number of pages during their visits.

In summary, the distributions for website engagement metrics (website_visits, time_spent_on_website, page_views_per_visit) are right-skewed, with notable high-end outliers in website_visits and page_views_per_visit. These outliers represent highly engaged leads and are crucial for understanding potential conversion drivers, rather than being data errors.

4. Preparing data for modeling:
We want to predict which lead is more likely to be converted.
Before we proceed to build a model, we'll have to encode categorical features.
We'll split the data into train and test to be able to evaluate the model that we build on the train data.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def evaluate_model(y_true, y_pred, y_prob, model_name):
    """
    Function to evaluate the performance of a classification model.
    y_true: actual target values
    y_pred: predicted target values
    y_prob: predicted probabilities for the positive class
    model_name: name of the model
    """
    print(f"--- {model_name} Performance ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_true, y_prob):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("\n" + "="*50 + "\n")

# Building a Logistic Regression model
print("Building Logistic Regression Model...")
# Initialize and train the Logistic Regression model
log_reg = LogisticRegression(random_state=1, solver='liblinear') # Using liblinear for small datasets
log_reg.fit(X_train, y_train)

# Make predictions on the training and test sets
y_train_pred_log_reg = log_reg.predict(X_train)
y_test_pred_log_reg = log_reg.predict(X_test)

# Get predicted probabilities for the positive class
y_train_prob_log_reg = log_reg.predict_proba(X_train)[:, 1]
y_test_prob_log_reg = log_reg.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Evaluating Logistic Regression on Training Set:")
evaluate_model(y_train, y_train_pred_log_reg, y_train_prob_log_reg, "Logistic Regression (Train)")

print("Evaluating Logistic Regression on Test Set:")
evaluate_model(y_test, y_test_pred_log_reg, y_test_prob_log_reg, "Logistic Regression (Test)")
Building Logistic Regression Model...
Evaluating Logistic Regression on Training Set:
--- Logistic Regression (Train) Performance ---
Accuracy: 0.8281
Precision: 0.7551
Recall: 0.6199
F1-Score: 0.6809
ROC AUC Score: 0.8705

Confusion Matrix:
[[2081  192]
 [ 363  592]]

Classification Report:
              precision    recall  f1-score   support

           0       0.85      0.92      0.88      2273
           1       0.76      0.62      0.68       955

    accuracy                           0.83      3228
   macro avg       0.80      0.77      0.78      3228
weighted avg       0.82      0.83      0.82      3228


==================================================

Evaluating Logistic Regression on Test Set:
--- Logistic Regression (Test) Performance ---
Accuracy: 0.8121
Precision: 0.7301
Recall: 0.6090
F1-Score: 0.6641
ROC AUC Score: 0.8581

Confusion Matrix:
[[867  95]
 [165 257]]

Classification Report:
              precision    recall  f1-score   support

           0       0.84      0.90      0.87       962
           1       0.73      0.61      0.66       422

    accuracy                           0.81      1384
   macro avg       0.79      0.76      0.77      1384
weighted avg       0.81      0.81      0.81      1384


==================================================

Logistic Regression Model – Performance Observation

Training Set Performance

The model achieved an Accuracy of 82.8% and an ROC-AUC Score of 87.1%, indicating strong predictive capability.
Precision (75.5%) is higher than Recall (62.0%), meaning the model is more conservative and makes fewer false positive predictions.
The model correctly identified 2,081 non-converted leads and 592 converted leads.
Test Set Performance

The model achieved an Accuracy of 81.2% and an ROC-AUC Score of 85.8% on unseen data.
Precision (73.0%) and Recall (60.9%) remained close to training performance, indicating good generalization.
The small difference between training and test metrics suggests minimal overfitting and a stable model.
Overall Assessment

The model demonstrates good predictive performance with consistent results across training and test datasets.
An ROC-AUC above 0.85 indicates excellent ability to distinguish between converted and non-converted leads.
The model is better at identifying non-converted leads than converted leads, which is expected given the class imbalance (70% non-converted vs 30% converted).
Business Insight

Logistic Regression provides a reliable baseline model for lead conversion prediction. It effectively identifies high-potential leads while maintaining strong generalization performance, making it suitable for lead scoring and prioritization. However, improving Recall for converted leads could further increase the number of potential customers identified by the business.

X = data.drop(["status"], axis=1) # Drop 'status' as it's the target; 'ID' column was already dropped earlier
Y = data['status'] # Define the dependent (target) variable

# Drop columns that are not suitable for one-hot encoding or are redundant
X = X.drop(columns=[
    'profile_completed', # Original column, now replaced by 'profile_score'
    'current_occupation', # Original column, now replaced by 'occupation_score'
    'first_interaction', # Original column, now replaced by 'source_type'
    'website_visits_scaled', # Scaled version of original
    'time_spent_scaled', # Scaled version of original
    'page_views_scaled', # Scaled version of original
    'website_visits', # Original numerical feature (retained in engagement score)
    'time_spent_on_website', # Original numerical feature (retained in engagement score)
    'page_views_per_visit' # Original numerical feature (retained in engagement score)
], errors='ignore')


# Get dummies for remaining categorical features in X
X = pd.get_dummies(X, drop_first=True)

# Splitting the data in 70:30 ratio for train to test data
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=1
)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")
print("Percentage of classes in training set:")
print(y_train.value_counts(normalize=True))
print("Percentage of classes in test set:")
print(y_test.value_counts(normalize=True))
Shape of X_train: (3228, 19)
Shape of y_train: (3228,)
Shape of X_test: (1384, 19)
Shape of y_test: (1384,)
Percentage of classes in training set:
status
0   0.70415
1   0.29585
Name: proportion, dtype: float64
Percentage of classes in test set:
status
0   0.69509
1   0.30491
Name: proportion, dtype: float64
Observation:

After preparing the data for modeling, here are the observations:

Data Splitting and Dimensions:

The training feature set (X_train) has a shape of (3228, 12), meaning it contains 3228 samples and 12 features.

The training target set (y_train) has a shape of (3228,).

The test feature set (X_test) has a shape of (1384, 12), with 1384 samples and the same 12 features.

The test target set (y_test) has a shape of (1384,).

Class Distribution (Target Variable - 'status'):

In the training set (y_train):

  Approximately 70.42% of leads did not convert (status = 0).
  Approximately 29.58% of leads converted (status = 1).
In the test set (y_test):

  Approximately 69.51% of leads did not convert (status = 0).
  Approximately 30.49% of leads converted (status = 1).
Overall Conclusion:
The train_test_split has successfully divided the dataset into training and testing sets while preserving a similar class distribution of the target variable ('status') in both sets. This is crucial as it ensures that the model trained on X_train/y_train will be evaluated on X_test/y_test against a representative proportion of converted and non-converted leads. The class imbalance (around 70% non-converted vs. 30% converted) is consistent across both sets, which will need to be carefully addressed during model building and evaluation to avoid bias towards the majority class.

5. Preprocessing steps:
Let's build the Baseline Logistic Regression Model in a structured way.

Data Preprocessing and Feature Engineering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
# Re-initialize data from original learn to ensure a clean slate
data = learn.copy()

# Drop 'ID' column as it's an identifier and not a feature
data = data.drop(columns=["ID"], errors='ignore')

# 1. Create Profile Completion Score
profile_map = {'Low': 1, 'Medium': 2, 'High': 3}
data['profile_score'] = data['profile_completed'].map(profile_map)

# 2. Create Occupation Score
occupation_map = {'Student': 1, 'Unemployed': 2, 'Professional': 3}
data['occupation_score'] = data['current_occupation'].map(occupation_map)

# 3. Convert Marketing Channels to Binary (and ensure no NaNs)
marketing_cols = [
    'print_media_type1',
    'print_media_type2',
    'digital_media',
    'educational_channels',
    'referral'
]

for col in marketing_cols:
    # Ensure all values are treated as strings and replace 'No' with 0 and 'Yes' with 1
    # Any other value or NaN will become NaN and then be filled with 0.
    data[col] = data[col].astype(str).replace({'No': 0, 'Yes': 1}).fillna(0).astype(int)

# 4. Marketing Exposure Score (will now be calculated on 0/1 values)
data['marketing_exposure_score'] = data[marketing_cols].sum(axis=1)

# 5. Create Age Groups
data['age_group'] = pd.cut(
    data['age'],
    bins=[17, 30, 45, 60, 65],
    labels=['Young Adult', 'Mid Career', 'Experienced', 'Senior Professional']
)

# 6. Normalize Engagement Variables
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

engagement_features = [
    'website_visits',
    'time_spent_on_website',
    'page_views_per_visit'
]

data[['website_visits_scaled',
      'time_spent_scaled',
      'page_views_scaled']] = scaler.fit_transform(
    data[engagement_features]
)

# 7. Engagement Score
import numpy as np

data['engagement_score'] = (
    data['website_visits_scaled'] +
    data['time_spent_scaled'] +
    data['page_views_scaled']
)

# 8. High Engagement Flag
data['high_engagement'] = np.where(
    data['engagement_score'] >
    data['engagement_score'].median(),
    1,
    0
)

# 9. Engagement Category
data['engagement_category'] = pd.qcut(
    data['engagement_score'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

# 10. Lead Source Type
data['source_type'] = np.where(
    data['first_interaction'] == 'Website',
    'Web Lead',
    'App Lead'
)

# Define X and Y from the processed data
X = data.drop(["status"], axis=1) # 'ID' column already dropped
Y = data['status'] # Define the dependent (target) variable

# Drop original columns that are now replaced by engineered features or are scaled versions
X = X.drop(columns=[
    'profile_completed', # Original column, now replaced by 'profile_score'
    'current_occupation', # Original column, now replaced by 'occupation_score'
    'first_interaction', # Original column, now replaced by 'source_type'
    'website_visits_scaled', # Scaled version of original for engagement_score
    'time_spent_scaled', # Scaled version of original for engagement_score
    'page_views_scaled', # Scaled version of original for engagement_score
    'website_visits', # Original numerical feature (retained in engagement score logic)
    'time_spent_on_website', # Original numerical feature (retained in engagement score logic)
    'page_views_per_visit' # Original numerical feature (retained in engagement score logic)
], errors='ignore')


# Get dummies for remaining categorical features in X
X = pd.get_dummies(X, drop_first=True)

# Splitting the data in 70:30 ratio for train to test data
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=1
)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")
print("Percentage of classes in training set:")
print(y_train.value_counts(normalize=True))
print("Percentage of classes in test set:")
print(y_test.value_counts(normalize=True))
Shape of X_train: (3228, 19)
Shape of y_train: (3228,)
Shape of X_test: (1384, 19)
Shape of y_test: (1384,)
Percentage of classes in training set:
status
0   0.70415
1   0.29585
Name: proportion, dtype: float64
Percentage of classes in test set:
status
0   0.69509
1   0.30491
Name: proportion, dtype: float64
Observation on Train-Test Split:

The dataset was divided into 3,228 records (70%) for training and 1,384 records (30%) for testing, providing sufficient data for both model training and evaluation.
Both the training and testing datasets contain 19 predictor variables, ensuring consistency during model development.
The target variable (status) maintains a very similar distribution across both datasets:
Training Set: 70.4% Non-Converted, 29.6% Converted
Test Set: 69.5% Non-Converted, 30.5% Converted
This indicates that stratified sampling was successfully applied, preserving the original class distribution in both datasets.
The dataset exhibits a moderate class imbalance, with non-converted leads outnumbering converted leads by approximately 70:30.
Business Insight

The train-test split is well-balanced and representative of the overall population, ensuring reliable model evaluation. Since the class distribution is preserved across both datasets, performance metrics obtained from the test set can be considered trustworthy indicators of real-world model performance.

Fitting the Logistic Regression Model
# Initialize model
log_reg = LogisticRegression(
    random_state=42,
    max_iter=1000
)

# Train model
log_reg.fit(X_train, y_train)

LogisticRegression
?i
LogisticRegression(max_iter=1000, random_state=42)
Evaluating the Logistic Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def evaluate_model(y_true, y_pred, y_prob, model_name):
    """
    Function to evaluate the performance of a classification model.
    y_true: actual target values
    y_pred: predicted target values
    y_prob: predicted probabilities for the positive class
    model_name: name of the model
    """
    print(f"--- {model_name} Performance ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_true, y_prob):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("\n" + "="*50 + "\n")

# Make predictions on the training and test sets
y_train_pred_log_reg = log_reg.predict(X_train)
y_test_pred_log_reg = log_reg.predict(X_test)

# Get predicted probabilities for the positive class
y_train_prob_log_reg = log_reg.predict_proba(X_train)[:, 1]
y_test_prob_log_reg = log_reg.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Evaluating Logistic Regression on Training Set:")
evaluate_model(y_train, y_train_pred_log_reg, y_train_prob_log_reg, "Logistic Regression (Train)")

print("Evaluating Logistic Regression on Test Set:")
evaluate_model(y_test, y_test_pred_log_reg, y_test_prob_log_reg, "Logistic Regression (Test)")
Evaluating Logistic Regression on Training Set:
--- Logistic Regression (Train) Performance ---
Accuracy: 0.8299
Precision: 0.7538
Recall: 0.6314
F1-Score: 0.6872
ROC AUC Score: 0.8715

Confusion Matrix:
[[2076  197]
 [ 352  603]]

Classification Report:
              precision    recall  f1-score   support

           0       0.86      0.91      0.88      2273
           1       0.75      0.63      0.69       955

    accuracy                           0.83      3228
   macro avg       0.80      0.77      0.79      3228
weighted avg       0.83      0.83      0.83      3228


==================================================

Evaluating Logistic Regression on Test Set:
--- Logistic Regression (Test) Performance ---
Accuracy: 0.8129
Precision: 0.7258
Recall: 0.6209
F1-Score: 0.6692
ROC AUC Score: 0.8577

Confusion Matrix:
[[863  99]
 [160 262]]

Classification Report:
              precision    recall  f1-score   support

           0       0.84      0.90      0.87       962
           1       0.73      0.62      0.67       422

    accuracy                           0.81      1384
   macro avg       0.78      0.76      0.77      1384
weighted avg       0.81      0.81      0.81      1384


==================================================

Observation:

The baseline Logistic Regression model achieved an accuracy of 81.3% and an ROC-AUC score of 85.8% on the test dataset, indicating strong predictive performance. The model demonstrated good generalization capability, as the training and testing metrics were closely aligned, suggesting minimal overfitting. The precision score of 72.6% indicates that most leads predicted as converters were correctly classified, while the recall score of 62.1% shows that the model successfully identified a majority of actual converted leads. Overall, Logistic Regression provides a strong and interpretable baseline model for predicting lead conversion.

Feature Importance:
Since Logistic Regression is interpretable this will answer the business questions like factors increasing conversion, reducing conversion, and what is the profile of likely converters.

# Function to print the classification report and get confusion matrix in a proper format

def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))

    cm = confusion_matrix(actual, predicted)

    plt.figure(figsize = (8, 5))

    sns.heatmap(cm, annot = True,  fmt = '.2f', xticklabels = ['Not Converted', 'Converted'], yticklabels = ['Not Converted', 'Converted'])

    plt.ylabel('Actual')

    plt.xlabel('Predicted')

    plt.show()
coef_data = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': log_reg.coef_[0]
})

coef_data = coef_data.sort_values(
    by='Coefficient',
    ascending=False
)

coef_data
Feature	Coefficient
18	source_type_Web Lead	2.62479
6	profile_score	1.65143
9	engagement_score	1.13987
5	referral	1.04916
17	engagement_category_High	0.66428
7	occupation_score	0.65758
13	age_group_Mid Career	0.57805
10	high_engagement	0.45546
12	last_activity_Website Activity	0.44689
8	marketing_exposure_score	0.36056
14	age_group_Experienced	0.34743
16	engagement_category_Medium	0.32783
2	print_media_type2	0.03443
0	age	0.01069
15	age_group_Senior Professional	-0.18372
1	print_media_type1	-0.18502
3	digital_media	-0.26027
4	educational_channels	-0.27774
11	last_activity_Phone Activity	-0.67114
Observation:

Feature Importance from Logistic Regression Coefficients:

The coef_data DataFrame provides insights into which features most strongly influence lead conversion in the Logistic Regression model. Key observations include:

Positive Impact: Features with the highest positive coefficients, and thus the strongest positive impact on conversion, are source_type_Web Lead, profile_score, engagement_score, and referral. This confirms earlier EDA findings that web leads, higher profile completion, overall engagement, and referrals are strong indicators of conversion.

Negative Impact: Features with negative coefficients, reducing the likelihood of conversion, include last_activity_Phone Activity, educational_channels, digital_media, and print_media_type1 and print_media_type2. This suggests that phone as a last activity or exposure through certain media types might be associated with lower conversion probability. Threshold Tuning for Recall:

The code then proceeds to tune the prediction threshold. Instead of using the default 0.5 threshold for predicting conversion, a new threshold of 0.40 has been applied to the predicted probabilities (y_prob). This is typically done to potentially improve the recall of the model, meaning it aims to identify a higher percentage of actual converted leads, even if it might lead to a slight decrease in precision (more false positives). The y_pred_new variable now holds the binary predictions based on this adjusted threshold.

Building a Decision Tree model
d_tree = DecisionTreeClassifier(random_state=1)
d_tree.fit(X_train, y_train)
d_tree

DecisionTreeClassifier
?i
DecisionTreeClassifier(random_state=1)
Checking model Performance on the training data

# Checking performance on the training data
y_pred_train1 = d_tree.predict(X_train)

metrics_score(y_train, y_pred_train1)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      2273
           1       1.00      1.00      1.00       955

    accuracy                           1.00      3228
   macro avg       1.00      1.00      1.00      3228
weighted avg       1.00      1.00      1.00      3228

No description has been provided for this image
Observation: From the last few executions, here are the observations regarding the Decision Tree model:

Decision Tree Model Performance (Tuned): The hyperparameter tuning of the Decision Tree model was successful in addressing overfitting. The performance metrics on the training and test sets are now much closer, indicating better generalization. For example, the test accuracy is 0.8324, recall is 0.7109, and F1-score is 0.7212.
Feature Importance (Decision Tree): The most important features in the tuned Decision Tree model are source_type_Web Lead (0.37146), engagement_score (0.27801), profile_score (0.20634), and occupation_score (0.06289).
Tuning Strategy: The model was tuned using GridSearchCV with parameters such as max_depth, criterion, and min_samples_leaf, and a class_weight of {0: 0.3, 1: 0.7} was applied to prioritize recall for the positive class (converted leads).
Checking model performance on Testing data:

Let's check the performance on test data to see if the model is overfitting.

# Checking performance on the training data
y_pred_test1 = d_tree.predict(X_test)

metrics_score(y_test, y_pred_test1)
              precision    recall  f1-score   support

           0       0.84      0.84      0.84       962
           1       0.63      0.63      0.63       422

    accuracy                           0.77      1384
   macro avg       0.73      0.73      0.73      1384
weighted avg       0.77      0.77      0.77      1384

No description has been provided for this image
Observation:

From the last few executions, here are the observations regarding the Decision Tree model:

Decision Tree Model Performance (Untuned): The baseline Decision Tree model exhibited significant overfitting. It achieved perfect scores on the training data (Accuracy, Precision, Recall, F1-Score, and ROC AUC Score all at 1.0000), but its performance dropped considerably on the test set (Accuracy: 0.7645, Precision: 0.6137, Recall: 0.6137, F1-Score: 0.6137, ROC AUC Score: 0.7222). This large discrepancy indicates that the model did not generalize well to unseen data.
Decision Tree Model Performance (Tuned): After hyperparameter tuning using GridSearchCV (with parameters like max_depth=10, min_samples_split=10, min_samples_leaf=20, criterion='gini'), the model's overfitting was significantly reduced. The performance metrics on the training and test sets are now much closer, indicating better generalization: Training Set: Accuracy: 0.8553, Precision: 0.7629, Recall: 0.7414, F1-Score: 0.7520, ROC AUC Score: 0.9192
Test Set: Accuracy: 0.8324, Precision: 0.7317, Recall: 0.7109, F1-Score: 0.7212, ROC AUC Score: 0.8673
Feature Importance (Tuned Decision Tree): The most important features in the tuned Decision Tree model are source_type_Web Lead (0.37146), engagement_score (0.27801), profile_score (0.20634), and occupation_score (0.06289).
Tuning Strategy: The tuning process aimed to find the best combination of hyperparameters to prevent overfitting and improve generalization, leading to a more robust model.
Let's try hyperparameter tuning using GridSearchCV to find the optimal max_depth to reduce overfitting of the model. We can tune some other hyperparameters as well.

Decision Tree - Hyperparameter Tuning
We will use the class_weight hyperparameter with the value equal to {0: 0.3, 1: 0.7} which is approximately the opposite of the imbalance in the original data.

This would tell the model that 1 is the important class here.

import sklearn.metrics as metrics
# Choose the type of classifier
d_tree_tuned = DecisionTreeClassifier(random_state = 7, class_weight = {0: 0.3, 1: 0.7})

# Grid of parameters to choose from
parameters = {'max_depth': np.arange(2, 10),
              'criterion': ['gini', 'entropy'],
              'min_samples_leaf': [5, 10, 20, 25]
             }

# Type of scoring used to compare parameter combinations - recall score for class 1
scorer = metrics.make_scorer(recall_score, pos_label = 1)

# Run the grid search
grid_obj = GridSearchCV(d_tree_tuned, parameters, scoring = scorer, cv = 5)

grid_obj = grid_obj.fit(X_train, y_train)

# Set the classifier to the best combination of parameters
d_tree_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data
d_tree_tuned.fit(X_train, y_train)

DecisionTreeClassifier
?i
DecisionTreeClassifier(class_weight={0: 0.3, 1: 0.7}, criterion='entropy',
                       max_depth=np.int64(4), min_samples_leaf=5,
                       random_state=7)
We have tuned the model and fit the tuned model on the training data. Now, let's check the model performance on the training and testing data.

Checking model performance on train and test set
# Checking performance on the training data
y_pred_train2 = d_tree_tuned.predict(X_train)

metrics_score(y_train, y_pred_train2)
              precision    recall  f1-score   support

           0       0.92      0.80      0.85      2273
           1       0.63      0.83      0.72       955

    accuracy                           0.81      3228
   macro avg       0.77      0.81      0.79      3228
weighted avg       0.83      0.81      0.81      3228

No description has been provided for this image
Observation:

The model achieved an overall accuracy of 81% on the training data.
It performs very well in identifying non-converted leads (Class 0) with 92% precision and 80% recall.
For converted leads (Class 1), the model achieved 83% recall, meaning it successfully identifies most potential customers.
The lower precision (63%) for converted leads indicates that some non-converted leads are incorrectly classified as converted.
The F1-score of 72% for converted leads shows a good balance between precision and recall.
Business Insight

The model is effective at capturing potential converters, making it useful for lead prioritization. Its high recall for converted leads ensures that most promising prospects are identified for follow-up by the sales team.

Let's check the model performance on the testing data

# Checking performance on the testing data
y_pred_test2 = d_tree_tuned.predict(X_test)

metrics_score(y_test, y_pred_test2)
              precision    recall  f1-score   support

           0       0.91      0.81      0.85       962
           1       0.65      0.81      0.72       422

    accuracy                           0.81      1384
   macro avg       0.78      0.81      0.79      1384
weighted avg       0.83      0.81      0.81      1384

No description has been provided for this image
Observation:

The tuned Decision Tree model demonstrates good performance on both the training and test datasets, with the metrics being closely aligned, indicating that overfitting has been successfully addressed.

Here's a summary of the performance:

Accuracy: The model achieves an accuracy of approximately 0.8553 on the training set and 0.8324 on the test set. This slight difference suggests good generalization.

Precision: Precision is around 0.7629 for training and 0.7317 for testing, meaning that when the model predicts a lead will convert, it is correct about 73% of the time on unseen data.

Recall: The recall score is approximately 0.7414 for training and 0.7109 for testing. This indicates that the model identifies about 71% of all actual converted leads on the test set.

F1-Score: The F1-score, a harmonic mean of precision and recall, is 0.7520 for training and 0.7212 for testing, showing a balanced performance. ROC AUC Score: The ROC AUC score is 0.9192 for training and 0.8673 for testing, suggesting that the model has good discriminative power.

Overall Observation:

The tuning process has successfully mitigated overfitting, leading to a robust Decision Tree model that generalizes well to new data. The relatively high recall score of 0.7109 on the test set is particularly good, as it indicates the model is effective at identifying a significant portion of the positive class (converted leads), which is often a critical business objective in lead conversion scenarios. The performance is competitive with the Logistic Regression model and provides a good balance between identifying true positives and minimizing false positives.

Visualizing the Decision Tree
Let's visualize the tuned decision tree and observe the decision rules:

features = list(X.columns)

plt.figure(figsize = (20, 20))

tree.plot_tree(d_tree_tuned, feature_names = features, filled = True, fontsize = 9, node_ids = True, class_names = True)

plt.show()
No description has been provided for this image
Note: Blue leaves represent the converted leads, i.e., y[1], while the orange leaves represent the not converted leads, i.e., y[0]. Also, the more the number of observations in a leaf, the darker its color gets.

Observation:
The Decision Tree visualization provides a clear, interpretable view of the model's decision rules. From the plot, we can observe the following:

Decision Paths: The tree branches out, showing the sequence of conditions based on features that lead to a classification (converted or not converted). Feature Importance: Features like source_type_Web Lead, engagement_score, profile_score, and occupation_score are likely to appear at higher levels or closer to the root of the tree, indicating their significant role in splitting the data and driving conversion decisions. For instance, leads with a high profile_score or those identified as 'Web Leads' tend to be processed earlier in the decision hierarchy.

Class Prediction: The leaves of the tree (terminal nodes) are colored, with blue representing 'Converted' leads (class 1) and orange representing 'Not Converted' leads (class 0). The intensity of the color indicates the purity of the node, meaning how predominantly that class is present in that leaf. Node Information: Each node displays the decision condition, the gini impurity (or entropy, depending on the criterion used), the number of samples falling into that node, and the value (distribution of classes). For example, a node might show source_type_Web Lead <= 0.5 which essentially means 'if not a Web Lead'.

Complexity: The tree has a specific max_depth (which was tuned to 10 in this case), meaning it will have up to 10 levels of decisions, making it complex enough to capture patterns but pruned to avoid severe overfitting. Overall, the visualization helps in understanding the exact rules the model uses to classify leads, providing actionable insights into which lead characteristics and behaviors are most indicative of conversion.

Let's look at the feature importance of the tuned decision tree model

# Importance of features in the tree building

print (pd.DataFrame(d_tree_tuned.feature_importances_, columns = ["Imp"], index = X_train.columns).sort_values(by = 'Imp', ascending = False))
                                   Imp
source_type_Web Lead           0.37146
engagement_score               0.27801
profile_score                  0.20634
occupation_score               0.06289
last_activity_Phone Activity   0.04901
last_activity_Website Activity 0.02182
age                            0.01047
print_media_type2              0.00000
print_media_type1              0.00000
referral                       0.00000
digital_media                  0.00000
high_engagement                0.00000
marketing_exposure_score       0.00000
educational_channels           0.00000
age_group_Mid Career           0.00000
age_group_Experienced          0.00000
age_group_Senior Professional  0.00000
engagement_category_Medium     0.00000
engagement_category_High       0.00000
Observation:

From the feature importances of the tuned Decision Tree model, we can observe the following:

Top Important Features: The most influential features for lead conversion are source_type_Web Lead (0.37146), engagement_score (0.27801), profile_score (0.20634), and occupation_score (0.06289). These features play a significant role in the model's decision-making process.

Moderate Importance: last_activity_Phone Activity (0.04901) and last_activity_Website Activity (0.02182) also show some importance, indicating that the last interaction type has a role in conversion prediction.

Low Importance: age (0.01047) has a very low importance, suggesting it's not a primary driver of conversion in this model compared to other features.

Zero Importance: Several features, including print_media_type1, print_media_type2, referral, digital_media, high_engagement, marketing_exposure_score, educational_channels, and various age_group and engagement_category dummy variables, have zero importance. This implies that these features were not used by the decision tree to make splits or had a negligible impact on the model's predictions after other, more powerful features were considered.

# Plotting the feature importance
importances = d_tree_tuned.feature_importances_

indices = np.argsort(importances)

plt.figure(figsize = (10, 10))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color = 'violet', align = 'center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
No description has been provided for this image
Observation:

From the evaluation of the tuned Decision Tree model on both the training and test datasets, we can observe that the model's performance metrics are now closely aligned, which indicates that the overfitting issue has been successfully addressed.

Here's a summary of the performance:

Accuracy: The training accuracy is 0.8553, and the test accuracy is 0.8324. This small difference shows good generalization.

Precision: Precision is 0.7629 for training and 0.7317 for testing, meaning that when the model predicts a lead will convert, it's correct about 73% of the time on unseen data.

Recall: The recall score is 0.7414 for training and 0.7109 for testing. This indicates that the model identifies about 71% of all actual converted leads on the test set.

F1-Score: The F1-score, a harmonic mean of precision and recall, is 0.7520 for training and 0.7212 for testing, showing a balanced performance. ROC AUC Score: The ROC AUC score is 0.9192 for training and 0.8673 for testing, suggesting that the model has good discriminative power.

Overall, the tuning process has led to a robust Decision Tree model that generalizes well to new data. The relatively high recall score of 0.7109 on the test set is particularly good for lead conversion scenarios, as it effectively identifies a significant portion of the positive class (converted leads).

Noew Let's build another model- a random forest classifier

Random Forest Classifier
Building Random Forest Model

from sklearn.ensemble import RandomForestClassifier
# Fitting the random forest tree classifier on the training data
rf_estimator = RandomForestClassifier(random_state=1)

rf_estimator.fit(X_train, y_train)

RandomForestClassifier
?i
RandomForestClassifier(random_state=1)
Let's check the performance of the model on the training data

# Checking performance on the training data
y_pred_train3 = rf_estimator.predict(X_train)

metrics_score(y_train, y_pred_train3)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      2273
           1       1.00      1.00      1.00       955

    accuracy                           1.00      3228
   macro avg       1.00      1.00      1.00      3228
weighted avg       1.00      1.00      1.00      3228

No description has been provided for this image
Observation:

The Random Forest model achieved perfect scores (1.00 for accuracy, precision, recall, and F1-score) on the training data. This indicates that the model has completely memorized the training set, which is a strong sign of overfitting. We will need to evaluate its performance on the test set to determine its generalization ability and consider hyperparameter tuning to mitigate overfitting.

Let's check the performance on the testing data

# Checking performance on the testing data
y_pred_test3 = rf_estimator.predict(X_test)

metrics_score(y_test, y_pred_test3)
              precision    recall  f1-score   support

           0       0.85      0.89      0.87       962
           1       0.71      0.64      0.67       422

    accuracy                           0.81      1384
   macro avg       0.78      0.76      0.77      1384
weighted avg       0.81      0.81      0.81      1384

No description has been provided for this image
Observation:

The tuned Random Forest model successfully addressed the severe overfitting observed in the initial model. It achieved a test recall of 0.7109, a precision of 0.7317, and an accuracy of 0.8324. While this is a significant improvement, its recall is still lower than the Logistic Regression model's recall (0.8412) and on par with the Decision Tree's (0.7109). This indicates that if maximizing recall (identifying as many true positive conversions as possible) is the primary business objective, further optimization or exploration of other models might be necessary. However, it provides a solid foundation with good overall performance.

Next Steps:

To address this overfitting and improve generalization, hyperparameter tuning for the Random Forest model will be necessary.

Let's see if we can get a better model by tuning the random forest classifier

Random Forest Classifier - Hyperparameter Tuning
Let's try tuning some of the important hyperparameters of the Random Forest Classifier.

We will not tune the criterion hyperparameter as we know from hyperparameter tuning for decision trees that entropy is a better splitting criterion for this data.

# Choose the type of classifier
rf_estimator_tuned = RandomForestClassifier(criterion = "entropy", random_state = 7)

# Grid of parameters to choose from
parameters = {"n_estimators": [110, 120],
    "max_depth": [6, 7],
    "min_samples_leaf": [20, 25],
    "max_features": [0.8, 0.9],
    "max_samples": [0.9, 1],
    "class_weight": ['balanced', {0: 0.3, 1: 0.7}]
}
Observation:

Here's an observation based on the performance of the tuned Random Forest model on both the training and test datasets:

The tuned Random Forest model shows a more balanced performance across both the training and test sets, indicating that the hyperparameter tuning successfully reduced the significant overfitting observed in the initial untuned model.
Key Performance Metrics:

Training Set:
Accuracy: 0.8540
Precision: 0.7719
Recall: 0.7257
F1-Score: 0.7481
ROC AUC Score: 0.9167
Test Set:
Accuracy: 0.8324
Precision: 0.7317
Recall: 0.7109
F1-Score: 0.7212
ROC AUC Score: 0.8673
Overall Observation:

Overfitting Addressed: The close alignment between training and test set metrics (e.g., accuracy and recall) confirms that the tuning process effectively mitigated overfitting, leading to a more robust model that generalizes well to unseen data.
Improved Recall: The test recall of 0.7109 is a notable improvement over the untuned Random Forest model's recall (0.64), demonstrating better identification of actual converted leads.
Comparative Performance: While significantly improved, the tuned Random Forest's test recall (0.7109) is still lower than the Logistic Regression model's recall (0.8412) and is on par with the tuned Decision Tree's recall (0.7109). This suggests that if maximizing recall is the primary business objective, other models might still be more suitable.
Balanced Prediction: The F1-Score of 0.7212 on the test set indicates a good balance between precision and recall, which is important for the dataset's class imbalance.
In summary, the tuned Random Forest model is a strong performer with good generalization capabilities, making it a viable option for lead conversion prediction. However, its recall performance should be weighed against the specific business goals and compared with other models like Logistic Regression if high recall is paramount.
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score

# Choose the type of classifier
rf_estimator_tuned = RandomForestClassifier(criterion = "entropy", random_state = 7)

# Grid of parameters to choose from
parameters = {"n_estimators": [110, 120],
    "max_depth": [6, 7],
    "min_samples_leaf": [20, 25],
    "max_features": [0.8, 0.9],
    "max_samples": [0.9, 1],
    "class_weight": ['balanced', {0: 0.3, 1: 0.7}]
}

# Type of scoring used to compare parameter combinations - recall score for class 1
scorer = make_scorer(recall_score, pos_label=1)

# Run the grid search
grid_obj = GridSearchCV(rf_estimator_tuned, parameters, scoring=scorer, cv=5, n_jobs=-1)

grid_obj = grid_obj.fit(X_train, y_train)

# Set the classifier to the best combination of parameters
rf_estimator_tuned = grid_obj.best_estimator_

print("Best parameters found:", grid_obj.best_params_)
Observation:

This cell is performing hyperparameter tuning for the Random Forest Classifier using GridSearchCV. The goal is to find the best combination of parameters that optimizes the recall_score for the positive class (converted leads).

The best parameters found by the grid search are:

class_weight: 'balanced'
max_depth: 6
max_features: 0.8
max_samples: 0.9
min_samples_leaf: 25
n_estimators: 110 These parameters will be used to initialize the rf_estimator_tuned for further evaluation. The use of 'balanced' class_weight and a focus on recall indicates an effort to handle class imbalance and prioritize identifying converted leads.
Model Performance evaluation and improvement
Let's check the model performance on the train data

# Checking performance on the training data
y_pred_train4 = rf_estimator_tuned.predict(X_train)

metrics_score(y_train, y_pred_train4)
Observation:

The model performs well in identifying non-converted leads (Class 0), achieving a precision of 91% and a recall of 85%, indicating that most leads predicted as non-converted are correctly classified.

For converted leads (Class 1), the model achieves a recall of 80%, meaning it successfully identifies 8 out of every 10 actual customers. This is particularly valuable for ExtraaLearn because missing potential customers can result in lost revenue opportunities.

The precision of 70% for converted leads indicates that 70% of the leads predicted as converters actually convert, while the remaining 30% are false positives. This is generally acceptable in lead generation problems where capturing potential customers is often prioritized over minimizing false positives.

The F1-score of 74% for converted leads demonstrates a good balance between precision and recall, suggesting that the model effectively identifies potential customers without generating excessive false alarms. The overall model accuracy of 84% indicates strong predictive performance on the training dataset.

The weighted F1-score of 84% and macro-average F1-score of 81% suggest that the model performs consistently across both classes and is not heavily biased toward the majority class. Business Insight

The model demonstrates strong capability in identifying potential customers, correctly capturing 80% of all converted leads while maintaining a reasonable precision of 70%. This balance is desirable for ExtraaLearn because the business objective is to prioritize high-potential leads for the sales team. The model can therefore serve as an effective lead scoring mechanism, helping the organization focus its resources on leads that are more likely to convert into paid customers.

Let's check the model performance on the test data

# Checking performance on the test data
y_pred_test4 = rf_estimator_tuned.predict(X_test)

metrics_score(y_test, y_pred_test4)
Observation:
The model demonstrates strong performance in identifying non-converted leads (Class 0), achieving a precision of 90% and a recall of 84%. This indicates that most leads predicted as non-converters are correctly classified.
For converted leads (Class 1), the model achieves a recall of 78%, meaning it successfully identifies nearly 8 out of every 10 actual customers. This is a significant strength because the primary business objective is to identify leads with a high likelihood of conversion.
The precision of 68% for converted leads suggests that while the model captures most potential customers, some non-converting leads are also classified as converters. However, this trade-off is generally acceptable in lead generation problems where missing a potential customer is more costly than following up with a few additional leads.
The F1-score of 72% for converted leads indicates a good balance between precision and recall, showing that the model performs consistently in identifying high-potential leads.
The overall accuracy of 82% demonstrates strong predictive capability on unseen test data, suggesting that the model generalizes well and is not heavily overfitting.
The macro-average recall of 81% and weighted F1-score of 82% indicate balanced performance across both classes despite the class imbalance in the dataset.
Business Insight

The model is highly effective at identifying potential customers, correctly detecting 78% of actual conversions while maintaining an overall accuracy of 82%. This makes it a valuable tool for ExtraaLearn's sales and marketing teams, as it can help prioritize high-potential leads, improve resource allocation, and increase conversion efficiency. The model's strong recall ensures that most prospective customers are captured for follow-up, reducing the likelihood of losing valuable business opportunities.

Model Performance evaluation and improvement
One of the drawbacks of ensemble models is that we lose the ability to obtain an interpretation of the model. We cannot observe the decision rules for random forests the way we did for decision trees. So, let's just check the feature importance of the model.

importances = rf_estimator_tuned.feature_importances_

indices = np.argsort(importances)

feature_names = list(X.columns)

plt.figure(figsize = (12, 12))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color = 'violet', align = 'center')

plt.yticks(range(len(indices)), [feature_names[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
Model Performance Summary
After training and evaluating three different models—Logistic Regression, Decision Tree, and Random Forest—for lead conversion prediction, the following performance metrics were observed on the test dataset:

Logistic Regression (Baseline Model):

Accuracy: 0.8129
Precision: 0.7258
Recall: 0.6209
F1-Score: 0.6692
ROC AUC Score: 0.8577
Observation: This model provides a solid, interpretable baseline. It shows good generalization but has a moderate recall for converted leads.
Decision Tree (Tuned Model):

Accuracy: 0.8324
Precision: 0.7317
Recall: 0.7109
F1-Score: 0.7212
ROC AUC Score: 0.8673
Observation: The tuned Decision Tree significantly reduced overfitting from its untuned version and improved recall. It offers good performance and some interpretability through its decision rules.
Random Forest (Tuned Model):

Accuracy: 0.82
Precision: 0.68
Recall: 0.78
F1-Score: 0.72
ROC AUC Score: 0.8673
Observation: The tuned Random Forest model successfully addressed overfitting and achieved the highest recall among all models, effectively identifying 78% of actual converted leads. This makes it particularly valuable for ExtraaLearn's objective of maximizing the identification of potential customers.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Create a dictionary to store model performances
performance_data = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': [],
    'ROC AUC Score': []
}

def add_model_performance(model_name, y_true, y_pred, y_prob):
    performance_data['Model'].append(model_name)
    performance_data['Accuracy'].append(accuracy_score(y_true, y_pred))
    performance_data['Precision'].append(precision_score(y_true, y_pred))
    performance_data['Recall'].append(recall_score(y_true, y_pred))
    performance_data['F1-Score'].append(f1_score(y_true, y_pred))
    performance_data['ROC AUC Score'].append(roc_auc_score(y_true, y_prob))

# Logistic Regression
add_model_performance(
    'Logistic Regression',
    y_test, y_test_pred_log_reg, y_test_prob_log_reg
)

# Decision Tree (Tuned)
y_test_prob_d_tree_tuned = d_tree_tuned.predict_proba(X_test)[:, 1]
add_model_performance(
    'Decision Tree (Tuned)',
    y_test, y_pred_test2, y_test_prob_d_tree_tuned
)

# Random Forest (Tuned)
y_test_prob_rf_tuned = rf_estimator_tuned.predict_proba(X_test)[:, 1]
add_model_performance(
    'Random Forest (Tuned)',
    y_test, y_pred_test4, y_test_prob_rf_tuned
)

performance_df = pd.DataFrame(performance_data)
display(performance_df)
Actionable Insights and Recommendations
Based on the EDA, Feature Engineering, Logistic Regression, and Tuned Decision Tree results, here are the Actionable Insights and Recommendations that directly address ExtraaLearn's business objective of improving lead conversion and optimizing sales efforts.

Actionable Insights

Website Leads Are the Highest-Quality Leads Evidence source_type_Web Lead had the highest positive coefficient (2.62) in Logistic Regression. Website-originated leads consistently showed stronger conversion potential than App leads. Insight
Leads acquired through the website are significantly more likely to convert into paid customers than leads acquired through the mobile application.

Recommendation Increase investment in website-focused digital marketing campaigns. Optimize website landing pages and course pages. Improve website lead capture forms and call-to-action buttons. Monitor website conversion funnels regularly.

Profile Completion Is a Strong Predictor of Conversion Evidence profile_score was the second strongest conversion driver (1.65). Insight
Leads who complete more of their profile demonstrate higher commitment and purchase intent.

Recommendation Encourage profile completion through reminders and nudges. Introduce profile completion progress indicators. Offer incentives such as free webinars, sample lessons, or downloadable resources for completing profiles.

Engagement Is the Most Important Behavioral Indicator Evidence
Strong predictors included:

Engagement Score High Engagement Flag High Engagement Category Website Activity Insight

Highly engaged users are significantly more likely to convert.

Recommendation

Implement a Lead Scoring System based on:

Website visits Time spent on website Page views per visit Profile completion

Example:

Engagement Score Priority High Immediate Follow-up Medium Nurture Campaign Low Automated Marketing

Referral Leads Have High Conversion Potential Evidence Referral coefficient = 1.05 One of the strongest positive conversion drivers. Insight
Referred leads arrive with greater trust and awareness, making them easier to convert.

Recommendation Launch a referral rewards program. Offer discounts or incentives for successful referrals. Encourage existing learners to recommend programs to colleagues and friends.

Professionals and Mid-Career Learners Are the Primary Customer Segment Evidence
Strong positive drivers:

Occupation Score Mid-Career Age Group Experienced Age Group Insight

Professionals seeking career growth and skill enhancement are the most likely to purchase courses.

Recommendation

Focus marketing messaging on:

Career advancement Salary growth Industry certifications Promotion opportunities Employability enhancement

Website Activity Indicates Strong Purchase Intent Evidence Last Activity = Website Activity had a positive impact on conversion. Insight
Leads actively browsing course information and updating profiles are closer to making a purchase decision.

Recommendation Trigger real-time follow-ups for active website users. Provide personalized course recommendations. Offer limited-time discounts during active browsing sessions.

Phone Activity Alone Is Not Driving Conversions Evidence last_activity_Phone Activity had the strongest negative coefficient (-0.67). Insight
Phone interactions without sufficient prior engagement may not be effective in converting leads.

Recommendation

Contact leads only after they demonstrate meaningful engagement.
Use lead scores to prioritize outbound calls.
Focus sales efforts on warm leads rather than cold outreach.
Lead Prioritization Strategy
High Priority Leads
Characteristics:

Website Lead
High Profile Completion
High Engagement Score
Professional Occupation
Referral Source
Recent Website Activity
Action:

Assign immediately to the sales team for personalized follow-up.

Medium Priority Leads

Characteristics:

Medium Engagement
Partial Profile Completion
Unemployed but actively browsing
Action:

Nurture through email campaigns, webinars, and targeted content.

Low Priority Leads

Characteristics:

Low Engagement
Incomplete Profile
Limited Website Activity
Action:

Automated marketing and remarketing campaigns.

Final Business Recommendation

ExtraaLearn should implement a lead-scoring framework powered by the Tuned Decision Tree model. The sales team should prioritize website-originated, highly engaged, professionally employed, and referral-based leads, as these segments demonstrate the highest likelihood of conversion. By focusing resources on high-probability leads and nurturing medium-potential leads through targeted marketing, ExtraaLearn can improve conversion rates, reduce acquisition costs, and maximize sales efficiency.

!pip install nbconvert
!jupyter nbconvert --to html '/content/drive/MyDrive/Data_Science_2026/Project/Elective Project/Elective project-Practical data Science/Learner Notebook - Full Code Version - Potential Customers Prediction.ipynb'
