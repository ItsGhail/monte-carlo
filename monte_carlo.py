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
        In this project, we will focus on Monte Carlo simulations to model and analyze risk in various scenarios. Monte Carlo methods use random sampling to obtain numerical results, particularly useful for understanding uncertainty in complex systems.
        ### Steps:
        1. Data Generation
        2. Exploratory Data Analysis (EDA)
        3. Modeling
        4. Simulation
        5. Evaluation and Analysis
        6. Conclusion
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