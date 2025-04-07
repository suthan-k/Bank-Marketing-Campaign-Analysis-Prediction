# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

#### -------------------------------------------------------------------------- ####


# Load saved models
with open("models/logistics.pkl", "rb") as file:
    logistic_model = pickle.load(file)

with open("models/rf.pkl", "rb") as file:
    random_forest_model = pickle.load(file)

with open("models/xgb.pkl", "rb") as file:
    xgboost_model = pickle.load(file)


#### -------------------------------------------------------------------------- ####


# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/bank.csv")  # Replace with your dataset path


data = load_data()

# changing data types of categorical colomuns
cat_cols = data.select_dtypes(include=["object"])

for col in cat_cols:
    data[col] = data[col].astype("category")

# Ordering the month
month_order = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]

# Convert 'month' column to categorical type with the defined order
data["month"] = pd.Categorical(data["month"], categories=month_order, ordered=True)


#### -------------------------------------------------------------------------- ####


# Crete a copy of the dataset for EDA
bank_df = data.copy()

# Creating age segmentation
age_bins = [17, 25, 35, 50, 65, 100]
age_label = ["Age <=25", "Age 26-35", "Age 36-50", "Age 51-65", "Age >65"]
bank_df["age_segmentation"] = pd.cut(bank_df["age"], bins=age_bins, labels=age_label)
bank_df["age_segmentation"] = bank_df["age_segmentation"].astype("category")

# Creating balance segmentation
balance_bins = [-990, 200, 1000, 5000, 15000, 100000]
balance_label = [
    "Low Balance",
    "Moderate Balance",
    "Comfortable Savers",
    "Wealthy Clients",
    "High Value Clients",
]
bank_df["balance_segmentation"] = pd.cut(
    bank_df["balance"], bins=balance_bins, labels=balance_label
)
bank_df["balance_segmentation"] = bank_df["balance_segmentation"].astype("category")

# Creating duration segmentation
median_duration = bank_df["duration"].median()
bank_df["duration_segmentation"] = np.where(
    bank_df["duration"] > median_duration,
    "duration_above_median",
    "duration_below_median",
)

#### -------------------------------------------------------------------------- ####

# Set Header
html_temp = """
    		<div style="background-color:{};padding:10px;border-radius:15px">
    		<h1 style="color:{};text-align:center;"> Bank Marketing Campaign Analysis & Prediction </h1>
    		</div>
    		"""
st.markdown(html_temp.format("#17202a", "white"), unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose an option", ["EDA", "Predictions"])

if option == "EDA":
    # Display an image
    img = Image.open("pictures/eda.jpg")
    st.image(img, width=700)

    # Title for EDA section
    st.title("Exploratory Data Analysis (EDA)")

    # Overview of the dataset
    st.subheader("Dataset Overview")
    st.write(data.head())
    st.write(f"Shape of the dataset: {data.shape[0]} rows and {data.shape[1]} columns")
    st.write("Summary statistics of Numerical Features:")
    st.write(data.describe().T)
    st.write("Summary statistics of Categorical Features:")
    st.write(data.describe(include=["category"]).T)

    # Graph options
    st.subheader("Visualizations")
    graph_option = st.selectbox(
        "Choose a graph type",
        [
            "Target Variable Distribution",
            "Bar Plots",
            "Histograms",
            "Box Plots",
            "Compare Features with Target",
        ],
    )

    if graph_option == "Target Variable Distribution":
        st.subheader("Target Variable Distribution")
        target_counts = data["deposit"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(
            target_counts, labels=target_counts.index, autopct="%1.1f%%", startangle=90
        )
        ax.axis("equal")
        st.pyplot(fig)

    elif graph_option == "Bar Plots":
        st.subheader("Bar Plots of Categorical Features")
        categorical_features = data.select_dtypes(include=["category"]).columns
        feature = st.selectbox("Select a feature", categorical_features)
        fig, ax = plt.subplots()
        if feature == "job":
            sns.countplot(x=feature, data=bank_df, ax=ax)
            plt.xticks(rotation=75)
            for container in ax.containers:
                ax.bar_label(container, fontsize=8)
            st.pyplot(fig)
        else:
            sns.countplot(x=feature, data=bank_df, ax=ax)
            for container in ax.containers:
                ax.bar_label(container, fontsize=8)
            st.pyplot(fig)

    elif graph_option == "Histograms":
        st.subheader("Histograms of Numerical Features")
        numerical_features = data.select_dtypes(include=[np.number]).columns
        feature = st.selectbox("Select a feature", numerical_features)
        fig, ax = plt.subplots()
        sns.histplot(data[feature], kde=True, ax=ax)
        st.pyplot(fig)

    elif graph_option == "Box Plots":
        st.subheader("Box Plots of Numerical Features")
        numerical_features = data.select_dtypes(include=[np.number]).columns
        feature = st.selectbox("Select a feature", numerical_features)
        fig, ax = plt.subplots()
        sns.boxplot(y=data[feature], ax=ax)
        st.pyplot(fig)

    elif graph_option == "Compare Features with Target":
        st.subheader("Compare Features with Target Variable")
        cols = [
            "age_segmentation",
            "balance_segmentation",
            "duration_segmentation",
            "job",
            "marital",
            "education",
            "contact",
            "month",
            "poutcome",
            "housing",
            "loan",
            "default",
        ]
        feature = st.selectbox("Select a feature", cols)
        fig, ax = plt.subplots()
        if feature == "balance_segmentation" or feature == "job":
            sns.histplot(
                data=bank_df,
                x=feature,
                multiple="dodge",
                hue="deposit",
                shrink=0.9,
                stat="probability",
                ax=ax,
            )
            plt.xticks(fontsize=10, rotation=75)
            st.pyplot(fig)
        elif feature == "poutcome":
            data = bank_df[bank_df["poutcome"].isin(["failure", "success"])]
            sns.histplot(
                data=data,
                x=feature,
                multiple="dodge",
                hue="deposit",
                shrink=0.9,
                stat="probability",
                ax=ax,
            )
            st.pyplot(fig)
        else:
            sns.histplot(
                data=bank_df,
                x=feature,
                multiple="dodge",
                hue="deposit",
                shrink=0.9,
                stat="probability",
                ax=ax,
            )
            st.pyplot(fig)

## ------------------------------------------------------------- ##

elif option == "Predictions":
    # Display an image
    img = Image.open("pictures/prediction.jpg")
    st.image(img, width=700)

    # Title for Predictions section
    st.title("Model Predictions")

    # Input features
    st.subheader("Select Input Features")

    # Collects user input features into dataframe
    def user_input_features():
        job = st.selectbox(
            "Job",
            (
                "student",
                "admin.",
                "technician",
                "services",
                "management",
                "blue-collar",
                "entrepreneur",
                "housemaid",
                "self-employed",
                "unemployed",
                "retired",
            ),
        )
        education = st.selectbox("Education", ("primary", "secondary", "tertiary"))
        marital = st.selectbox("Marital Status", ("single", "married", "divorced"))
        month = st.selectbox(
            "Contacted Month",
            (
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec",
            ),
        )
        housing = st.selectbox("Has a Housing Loan?", ("yes", "no"))
        loan = st.selectbox("Has a Personal Loan?", ("yes", "no"))
        default = st.selectbox("Has Credit in Default?", ("yes", "no"))
        contact = st.selectbox("Contact Type", ("cellular", "telephone"))
        poutcome = st.selectbox(
            "Previous outcome of a campaign", ("success", "failure", "unknown")
        )
        age = st.slider("Age", 17, 100, 30)
        balance = st.slider("Balance", -7000, 100000, 5000)
        day = st.slider("Last Contact Day of the Month", 1, 31, 15)
        duration = st.slider("Last contact duration in seconds", 0, 4000, 100)
        campaign = st.slider("No of contatacts performed during the campaign", 1, 70, 1)
        previous = st.slider(
            "No of contatacts performed before this campaign", 0, 60, 0
        )
        pdays = st.slider(" Days since the client was last contacted", -1, 1000, 5)

        # Create a dictionary with the input values
        data = {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome,
        }

        # Convert the dictionary to a DataFrame
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Displays the user input features
    st.subheader("User Input Features")
    input_df_transposed = input_df.T
    input_df_transposed.columns = ["User Input"]
    input_df_transposed.index.name = "Feature"
    st.write(input_df_transposed)

    # Converting user input features to the same data types as the dataset
    input_cat_cols = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
    ]
    for col in input_cat_cols:
        input_df[col] = input_df[col].astype("category")

    ## ------------------------------------------------------------- ##

    # Combines user input features with entire shoppers dataset
    # This will be useful for the encoding phase
    int_data = data.drop(columns=["deposit"])
    df = pd.concat([input_df, int_data], axis=0)

    # Changing data types of categorical columns
    for col in input_cat_cols:
        df[col] = df[col].astype("category")

    ### --------------------- Data encoding --------------------- ###

    ## Changing binary columns to integer ##
    binary_cols = ["default", "housing", "loan"]
    for col in binary_cols:
        df[col] = df[col].map({"yes": 1, "no": 0}).astype("int")

    ## Encoding 'month' and 'day' columns using cyclic encoding ##
    # Assigning numbers to months
    month_to_num = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    # Mapping months to month numbers
    df["month"] = df["month"].map(month_to_num).astype("int")

    # Function for cyclic encoding
    def cyclic_encoding(data, cols):
        for col in cols:
            data[f"sin_{col}"] = np.sin(2 * np.pi * data[col] / max(data[col]))
            data[f"cos_{col}"] = np.cos(2 * np.pi * data[col] / max(data[col]))
        return data

    # Performing cyclic encoding for the variables 'month' and 'day'
    cyclic_encoding(df, ["month", "day"])

    # Dropping the unwanted columns
    df.drop(columns=["month", "day"], inplace=True)

    ## Encoding other categorical variables with one-hot encoding ##
    cat_cols = df.select_dtypes(include=["category"]).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Selects only the first row (the user input data)
    user_data = df[:1]

    ## ------------------------------------------------------------- ##

    # Select model
    st.subheader("Select Model")
    model_option = st.selectbox(
        "Choose a model", ["Logistic Regression", "Random Forest", "XGBoost"]
    )

    if model_option == "Logistic Regression":
        model = logistic_model
    elif model_option == "Random Forest":
        model = random_forest_model
    elif model_option == "XGBoost":
        model = xgboost_model

    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(user_data)
        st.subheader("Prediction")
        output = np.array(["No", "Yes"])
        result = output[prediction[0]]
        st.info(result)
        if result == "Yes":
            st.write("The Client is likely to open a term deposit.")
        else:
            st.write("The Client is unlikely to open a term deposit.")
