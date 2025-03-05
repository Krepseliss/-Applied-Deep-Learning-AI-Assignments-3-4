import pandas as pd
import os
from crewai import Agent, Task, Crew, LLM
from litellm import completion
import streamlit as st
import re
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords


# 🚀 Set OpenAI API Key
api_key = "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"
os.environ["OPENAI_API_KEY"] = api_key

llm = LLM(
    model="gpt-3.5-turbo",
    temperature=0.7,
    base_url="https://api.openai.com/v1",
    api_key=api_key
)


def analyze_sentiment(text):
    """Uses OpenAI GPT to classify sentiment and return standardized labels."""
    prompt = (
        "Classify the sentiment of this review strictly as 'Positive', 'Negative', or 'Neutral'. "
        "Do not include extra words or explanations.\n\n"
        f"Review: {text}"
    )
    
    response = completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    sentiment = response["choices"][0]["message"]["content"].strip()

    # ✅ Standardize outputs
    if "positive" in sentiment.lower():
        return "Positive"
    elif "negative" in sentiment.lower():
        return "Negative"
    elif "neutral" in sentiment.lower():
        return "Neutral"
    else:
        return "Neutral"  # Default fallback if GPT returns unexpected text

# Load cleaned dataset
df = pd.read_csv('C:/Users/alex/Documents/Applied Deep Learning & Ai/Assignment 3/data1/redmi6_clean.csv', encoding='latin-1')
df.drop(columns=["Unnamed: 0"], inplace=True)

# Ensure dataset is correctly formatted
#print(df.head()) 

#================================AGENTS=========================================

# ✅ OpenAI-Based Sentiment Analysis Function
data_processing_agent = Agent(
    role="Data Processor",
    goal="Clean and prepare the dataset for analysis",
    backstory="An expert in structuring and formatting data for NLP tasks.",
    llm=llm
)

sentiment_agent = Agent(
    role="Sentiment Analyst",
    goal="Analyze customer review sentiments using AI",
    backstory="A machine learning specialist trained to classify text sentiment.",
    llm=llm
)

# 🕵️ Research Assistant Agent for Q&A
research_assistant = Agent(
    role="Research Assistant",
    goal="Answer user questions about the dataset",
    backstory="An AI-powered assistant that can answer questions based on sentiment analysis results.",
    llm=llm
)

#================================TASKS=========================================

# ✅ CrewAI Tasks
data_processing_task = Task(
    description="Load and clean the dataset for NLP processing.",
    agent=data_processing_agent,
    expected_output="A structured dataset with necessary columns for sentiment analysis."
)

sentiment_task = Task(
    description="Analyze customer review sentiment using OpenAI GPT.",
    agent=sentiment_agent,
    expected_output="A dataset with a new Sentiment column indicating customer emotions."
)

# ✅ Research Task: Answer User Queries About the Dataset
research_task = Task(
    description="Answer user questions about the sentiment analysis results.",
    agent=research_assistant,
    expected_output="A well-formatted response based on the dataset insights."
)

# ✅ Define Crew
crew = Crew(
    agents=[data_processing_agent, sentiment_agent, research_assistant],
    tasks=[data_processing_task, sentiment_task, research_task]
)

crew.kickoff()

# ✅ Function to Answer Questions Using OpenAI
def answer_question(question):
    """Uses OpenAI GPT to answer questions based on the dataset insights."""
    prompt = f"Explain your reasoning step-by-step before answering this question based on the dataset:\n\n{question}\n\nHere is a sample of the data:\n\n{df.head().to_string()}"
    
    response = completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response["choices"][0]["message"]["content"].strip()

# ✅ Apply Sentiment Analysis to Dataset
df["Sentiment"] = df["Comments"].apply(analyze_sentiment)


# Save results
#df.to_csv("C:/Users/alex/Documents/Applied Deep Learning & Ai/Assignment 3/Results/redmi6_reviews_with_sentiment.csv", index=False)

#print("Sentiment Analysis Completed. Results saved to 'redmi6_reviews_with_sentiment.csv'.")

#================================STREAMLIT Q&A SYSTEM=========================================

# 🤖 Function to Answer User Questions
def answer_question(question):
    """Uses OpenAI GPT to answer questions intelligently based on the dataset insights."""
    
    # ✅ Check if the question is related to the dataset
    dataset_keywords = ["sentiment", "reviews", "data", "positive", "negative", "neutral",
                        "Xiaomi", "Redmi6", "customers", "feedback", "opinion"]
    
    # ✅ If the question contains dataset-related keywords, analyze the dataset
    if any(re.search(rf"\b{kw}\b", question, re.IGNORECASE) for kw in dataset_keywords):
        prompt = f"If the following question is related to customer sentiment analysis, answer it using dataset insights. Otherwise, provide a general response:\n\n{question}\n\nHere is a sample of the data:\n\n{df.head().to_string()}"
    else:
        # ✅ If the question is unrelated, use GPT normally without dataset context
        prompt = f"Answer this question conversationally without referring to the dataset:\n\n{question}"

    # ✅ Send request to OpenAI GPT
    response = completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response["choices"][0]["message"]["content"].strip()

# 🔮 Streamlit UI for Q&A System with Tabs
st.title("🔍 Research Assistant Q&A System")

# Create tabs
tab1, tab2 = st.tabs(["Q&A System", "Visualizations"])

# Content for Tab 1: Q&A System
with tab1:
    st.write("Ask any question about the sentiment analysis dataset!")
    
    # ✅ User Input for Q&A
    question = st.text_input("Enter your question:")
    if st.button("Submit"):
        if question.strip():
            answer = answer_question(question)
            st.write(f"**Answer:** {answer}")
        else:
            st.write("⚠️ Please enter a question.")
    
    # ✅ Show Dataset Sample
    st.subheader("📊 Sentiment Analysis Data Sample")
    st.dataframe(df.head())

    # ✅ Download stopwords
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

    # ✅ Add Custom Stopwords
    custom_stopwords={"phone", "mobile", "one", "nice", "dont", "pro", "good", "redmi", "xiaomi", "like", "nice", "product", "mi",
    "buy", "using", "also", "get", "please", "well", "great"
    }
    stop_words.update(custom_stopwords) # updating the existing stopword list

    # ✅ Improved text cleaning function
    def clean_text(text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters & numbers
        words = text.split()
        words = [word for word in words if word not in stop_words]  # Remove stopwords
        return " ".join(words)

    # ✅ Combine all reviews into a single text block
    all_reviews = " ".join(df["Comments"].dropna().astype(str).apply(clean_text))
    
    # ✅ Tokenize and count word frequencies
    word_list = all_reviews.split()
    word_counts = Counter(word_list)

    # ✅ Convert to DataFrame
    common_words_df = pd.DataFrame(word_counts.items(), columns=["Word", "Count"])
    common_words_df = common_words_df.sort_values(by="Count", ascending=False).head(20)  # Top 20 words

    # ✅ Create Bar Chart
    fig1 = px.bar(
        common_words_df,
        x="Word",
        y="Count",
        title="Top 20 Most Frequent Words in Customer Reviews",
        text="Count",
        color="Count",
        color_continuous_scale="viridis"
    )

# Content for Tab 2: Visualizations (empty for now)
with tab2:
    st.title("📊 Sentiment Analysis Visualizations")
    st.write("This section contains visualizations of sentiment analysis data.")

    # ✅ Sentiment Proportion (Pie Chart)
    st.subheader("🥧 Sentiment Distribution")

    # Count occurrences of each sentiment
    sentiment_counts = df["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig = px.pie(
    sentiment_counts,
    names="Sentiment",
    values="Count",
    title="Sentiment Distribution",
    color="Sentiment",  # Color coding for each sentiment
    color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"},
    hole=0.3  # Creates a donut-style chart
    )

    # Display interactive Pie Chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # ✅ Interactive Filtering of Sentiments
    st.subheader("🔍 Filter Reviews by Sentiment")

    # Create a dropdown select box
    selected_sentiment = st.selectbox(
        "Choose a sentiment to filter reviews:",
        ["All", "Positive", "Neutral", "Negative"]
    )

    # Filter DataFrame based on user selection
    if selected_sentiment != "All":
        filtered_df = df[df["Sentiment"] == selected_sentiment]
    else:
        filtered_df = df  # Show all data if "All" is selected

    # Display filtered dataset
    st.write(f"Showing reviews for sentiment: **{selected_sentiment}**")
    st.dataframe(filtered_df[["Comments", "Sentiment"]])

    st.subheader("☁️ Word Cloud of Customer Reviews")

    # ✅ Display Word Cloud Chart in Streamlit
    st.plotly_chart(fig1, use_container_width=True)

touch requirements.txt