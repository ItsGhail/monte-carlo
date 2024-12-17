import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",  # Menu title
        options=["Home", "About", "Chart", "Contact"],  # Options in the menu
        icons=["house-heart-fill", "calendar2-heart-fill", "bar-chart-fill", "envelope-heart-fill"],  # Icons for the options
        menu_icon="heart-eyes-fill",  # Icon for the menu title
        default_index=0,  # Default selected option
    )

# Use 'selected' instead of 'select' in the conditional statements
if selected == "Home":
    st.title("Welcome to Monte Carlo Simulation for Risk Analysis! 👋")
    st.write("# MEMBERS: ")
    st.write("Abegail Repia")
    st.write("Ma. Teresa Saguit")
    st.write("Jesrel Pizzaro")
    
    st.markdown(
        """
        ### Want to learn more?
        - Jump into our [documentation](https://docs.google.com/document/d/1ug2qp4cOOrZnAm5UwRgHmMflDQrRTfqs/edit?usp=sharing&ouid=100564633130393337897&rtpof=true&sd=true)
        - Ask a question in my [Facebook account](https://www.facebook.com/liageba.aiper)
        ### See more complex demos
        - Risk Analysis using [monte carlo simulation](https://www.riskamp.com/files/Risk%20Analysis%20using%20Monte%20Carlo%20Simulation.pdf)
        - What is [Monte Carlo Simulation?](https://lumivero.com/software-features/monte-carlo-simulation/)
        - Monte Carlo Simulation: What It Is, How It Works, History, [4 Key Steps](https://www.investopedia.com/terms/m/montecarlosimulation.asp)
        - Probabilistic Risk Analysis Demo[Tool](https://www.riskcon.at/software/monte-carlo-demo-tool)
        - Introduction to Monte Carlo simulation in[Excel](https://support.microsoft.com/en-us/office/introduction-to-monte-carlo-simulation-in-excel-64c0ba99-752a-4fa8-bbd3-4450d8db16f1)
    """
    )
    
elif selected == "About":
    st.write("Provide a detailed explanation of the Monte Carlo simulation process and its application in risk analysis. You can add a section on the significance of the project and its learning objectives.")
    st.title("Monte Carlo Simulation for Risk Analysis")
    st.write("""
Introduction:
 Introduction:
This project focuses on modeling and simulation using Python, with an emphasis on Monte Carlo Simulation for risk analysis. The primary goal is to gain hands-on experience with Python libraries and tools widely used for these tasks. By working through this project, you will understand the steps involved in generating data, analyzing it, building models, and simulating outcomes to evaluate risk.

2. Project Overview:
The project is structured into several steps. Each step builds upon the previous one, starting from data
generation to simulation and model evaluation. Let’s break down the steps in more detail:
- Data Generation:
First, we need to create synthetic data that represents the system we want to model. This data will serve
as the input for the models and simulations. The synthetic data can be generated based on assumptions
about the real-world scenario being modeled (e.g., a population growth model, a stock price model, etc.).
- Exploratory Data Analysis (EDA):
Once we have the synthetic data, we perform exploratory analysis to understand its characteristics. This
involves summarizing statistical properties, identifying patterns, and visualizing the data to gain insights.
- Modeling:
After understanding the data, we apply a modeling technique that best fits the type of data we have. The
goal of modeling is to create a mathematical representation of the system that we can use to simulate its
behavior.
- Simulation:
The simulation phase uses the model to generate outcomes based on different inputs or scenarios. This is
often done using Monte Carlo simulations or other stochastic techniques, where we repeatedly simulate the
system's behavior to account for uncertainty and variability.
- Evaluation and Analysis:
Finally, the model's performance is evaluated by comparing the simulated outcomes with the actual data.
We will use various metrics and visualization techniques to assess the accuracy and robustness of the
model.
3. Data Generation:
Data generation is the first step in the project. The goal is to create synthetic data that mimics real-world
systems or processes. This data can be generated using different probability distributions or mathematical
models. Some common techniques include:
- Random Sampling with Numpy:
The NumPy library can generate random numbers from various distributions (e.g., uniform, normal,
exponential). For example, if you want to simulate a financial system with normally distributed returns, you
could use:
``python
import numpy as np
simulated_data = np.random.normal(loc=100, scale=20, size=1000)
Here, `loc` is the mean (100), `scale` is the standard deviation (20), and `size` is the number of data points
(1000).
- Using Scikit-learn for More Complex Data:
scikit-learn provides tools for generating more complex datasets, such as classification datasets or
regression datasets, using `make_classification()` or `make_regression()`. This allows you to generate
synthetic datasets with known underlying structures that can be used to model and test algorithms.
4. Exploratory Data Analysis (EDA):
EDA is a crucial step to understand the characteristics and relationships in your data. By visualizing the
data and calculating summary statistics, you can identify patterns, detect outliers, and get a sense of the
data's distribution. Key activities during EDA include:
- Statistical Summary:
Use pandas to calculate basic statistics (mean, median, standard deviation) of your data:
```python
import pandas as pd
data = pd.DataFrame(simulated_data, columns=["Outcome"])
data.describe()
- Visualization
Matplotlib and seaborn are useful for creating plots such as histograms, scatter plots, and box plots. For
example, a histogram can show the distribution of your data:
```python
import matplotlib.pyplot as plt
plt.hist(simulated_data, bins=50, color='blue', edgecolor='black')
plt.title('Distribution of Simulated Data')
plt.xlabel('Outcome')
plt.ylabel('Frequency')
plt.show()
These visualizations help in identifying trends, correlations, and outliers.
- Correlation Analysis:
If you have multiple variables, understanding the relationships between them is essential. You can use
correlation matrices and heatmaps (via seaborn ) to visualize the strength and direction of relationships:
```python
import seaborn as sns
sns.heatmap(data.corr(), annot=True)
5. Modeling:
Once you understand the data, the next step is to build a model that can describe or predict the behavior of
the system. Depending on the nature of the problem, you can choose from a variety of modeling
techniques:
- Linear Regression:
For continuous outcomes that are linearly dependent on one or more input features, you can use
scikit-learn's , `LinearRegression` model:
``python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
- Logistic Regression
If you’re dealing with classification tasks (binary outcomes), you could use logistic regression
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
- Time Series Models:
For time-dependent data (e.g., stock prices), time series models such as ARIMA or LSTM (Long
Short-Term Memory) networks are suitable.
- Neural Networks:
For more complex problems, deep learning with frameworks like TensorFlow or Keras allows you to
build neural networks that can handle intricate patterns in data.
6. Simulation:
In the simulation step, the goal is to use the model to generate potential outcomes under different
conditions. You might simulate multiple scenarios by varying input parameters and observing how the model
behaves.
- Monte Carlo Simulations:
Monte Carlo methods are widely used for simulations that involve uncertainty. In this approach, the model
is run many times using random inputs (based on predefined distributions) to assess the range of possible
outcomes. The NumPy library can help with this:
```python
# Monte Carlo Simulation with normal distribution
num_simulations = 10000
simulated_results = np.random.normal(loc=100, scale=20, size=num_simulations)
- Sensitivity Analysis:
This involves testing how sensitive your model is to changes in input parameters. By varying key
assumptions, you can observe how the output changes and identify the most influential factors in the model.
7. Evaluation and Analysis
Once the simulation is complete, the next step is to evaluate the model’s performance. This involves
comparing the simulated outcomes with known or observed data to see how well the model fits.
- Evaluation Metrics:
For regression tasks, metrics like Mean Absolute Error (MAE) Root Mean Squared Error (RMSE) and
R-squared are commonly used. For classification tasks, you would use metrics like accuracy , precision,
recall, and F1-score.
- Visual Analysis:
Visualizing the difference between the observed data and the simulated data helps assess the model's
accuracy. For example, you might plot the residuals (differences between predicted and actual values) to
check for any patterns that suggest model improvements.
8. Conclusion:
In this final phase, you summarize the findings from the project. The focus is on discussing the importance
of modeling and simulation in decision-making, risk analysis, and understanding complex systems. The
project should encourage further exploration of Python-based modeling techniques, highlighting the
flexibility and power of Python libraries for solving real-world problems. You can suggest applying the
knowledge gained to different domains such as finance (for predicting stock prices), healthcare (for patient
outcome modeling), or engineering (for simulating physical systems), and emphasize the potential impact of
simulation in improving decision-making processes.
This project allows for a comprehensive understanding of modeling and simulation in Python, with an
emphasis on practical applications. By following these steps, you’ll gain valuable insights into both the
theoretical and practical
    """)

elif selected == "Chart":
    st.sidebar.header("Simulation Parameters")
    num_simulations = st.sidebar.number_input("Number of Simulations", min_value=100, max_value=100000, value=1000, step=100)
    mean = st.sidebar.number_input("Mean of Distribution", value=100.0)
    std_dev = st.sidebar.number_input("Standard Deviation of Distribution", value=20.0)
    
    st.header("Simulating Outcomes")
    st.write("Running the Monte Carlo simulation with the following parameters:")
    st.write(f"Number of simulations: {num_simulations}")
    st.write(f"Mean: {mean}, Standard Deviation: {std_dev}")
    
    
    # Simulated data for illustration
    simulated_data = np.random.normal(loc=0, scale=1, size=1000)

    # Improved Histogram
    st.subheader("Histogram of Simulation Results")
    fig, ax = plt.subplots(figsize=(10, 6))  # Increase the figure size for better clarity

    # Plot histogram with enhancements
    ax.hist(simulated_data, bins=50, color='lightcoral', edgecolor='black', alpha=0.7, density=True)

    # Add a title and labels with larger font
    ax.set_title("Distribution of Simulation Outcomes", fontsize=16, fontweight='bold')
    ax.set_xlabel("Outcome", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)

    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.6)

    # Optionally, add a normal distribution curve (for comparison)
    from scipy.stats import norm
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, np.mean(simulated_data), np.std(simulated_data))
    ax.plot(x, p, 'k', linewidth=2, label="Normal Distribution")

    # Add a legend
    ax.legend()

    # Display the plot in Streamlit
    st.pyplot(fig) 
    # Download option
    st.subheader("Download Simulated Data")
    dataframe = pd.DataFrame(simulated_data, columns=["Simulated Outcome"])
    
        
    # Add a title and labels with larger font
    ax.set_title("Distribution of Simulation Outcomes", fontsize=16, fontweight='bold')
    ax.set_xlabel("Outcome", fontsize=14)
    ax.set_ylabel("Frequency (Density)", fontsize=14)

    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.6)
        
    csv = dataframe.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="simulated_data.csv",
        mime="text/csv"
    )

elif selected == "Contact":
    st.title("Contact Us")
    st.write("""
        If you have any questions, feel free to contact us at:
        - Email: abrepia@my.cspc.edu.ph,
        jepizzaro@my.cspc.edu.ph,
        masaguit@my.cspc.edu.ph
        
        - Facebook: https://www.facebook.com/liageba.aiper, https://www.facebook.com/iamyhegss2, https://www.facebook.com/jhess.pentecostes
    """)
