import streamlit as st
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

FILE = "tasks.csv"

# ---------------- LOAD DATA ----------------
if os.path.exists(FILE):
    tasks = pd.read_csv(FILE)
else:
    tasks = pd.DataFrame(columns=["description", "priority", "deadline", "status"])

# Ensure status column exists (for old files)
if "status" not in tasks.columns:
    tasks["status"] = "Pending"

# ---------------- ML MODEL ----------------
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
le = LabelEncoder()

if not tasks.empty and tasks["priority"].nunique() > 1:
    y = le.fit_transform(tasks["priority"])
    model.fit(tasks["description"], y)
    trained = True
else:
    trained = False

# ---------------- RULE-BASED FUNCTIONS ----------------
def auto_priority(text):
    text = str(text).lower()
    if any(w in text for w in ["urgent", "asap", "immediately", "deadline", "exam"]):
        return "High"
    elif any(w in text for w in ["later", "optional", "whenever"]):
        return "Low"
    else:
        return "Medium"

def deadline_priority(deadline):
    days_left = (deadline - datetime.date.today()).days
    if days_left <= 1:
        return "High"
    elif days_left <= 3:
        return "Medium"
    else:
        return "Low"

# ---------------- UI ----------------
st.title("📊 Smart Task Manager")

menu = st.sidebar.selectbox(
    "Menu",
    ["Add", "View", "Delete", "Edit", "Update Status", "Predict", "Analytics"]
)

# ---------------- TASK TEMPLATES ----------------
templates = [
    "Submit assignment before deadline",
    "Prepare for exam",
    "Attend meeting",
    "Buy groceries",
    "Complete project report"
]

# ---------------- ADD ----------------
if menu == "Add":

    st.subheader("Add Task")

    use_template = st.checkbox("Use template")

    if use_template:
        d = st.selectbox("Choose task", templates)
    else:
        d = st.text_input("Enter task")

    deadline = st.date_input("Deadline")

    manual_priority = st.selectbox("Manual Priority", ["Auto", "Low", "Medium", "High"])

    if st.button("Add Task"):

        rule_p = auto_priority(d)
        date_p = deadline_priority(deadline)

        if trained:
            ml_pred = le.inverse_transform(model.predict([d]))[0]
        else:
            ml_pred = "Medium"

        if manual_priority != "Auto":
            final_p = manual_priority
        else:
            if "High" in [rule_p, date_p, ml_pred]:
                final_p = "High"
            elif "Medium" in [rule_p, date_p, ml_pred]:
                final_p = "Medium"
            else:
                final_p = "Low"

        new = pd.DataFrame([[d, final_p, deadline, "Pending"]],
                           columns=["description", "priority", "deadline", "status"])

        tasks = pd.concat([tasks, new], ignore_index=True)
        tasks.to_csv(FILE, index=False)

        st.success(f"Added with Priority: {final_p}")

# ---------------- VIEW ----------------
elif menu == "View":

    st.subheader("All Tasks")

    if tasks.empty:
        st.warning("No tasks available")
    else:
        search = st.text_input("Search task")

        filter_priority = st.selectbox(
            "Filter by Priority", ["All", "High", "Medium", "Low"]
        )

        df = tasks.copy()

        if search:
            df = df[df["description"].str.contains(search, case=False)]

        if filter_priority != "All":
            df = df[df["priority"] == filter_priority]

        st.dataframe(df)

# ---------------- DELETE ----------------
elif menu == "Delete":

    if not tasks.empty:
        d = st.selectbox("Select task", tasks["description"])

        if st.button("Delete"):
            tasks = tasks[tasks["description"] != d]
            tasks.to_csv(FILE, index=False)
            st.success("Deleted")

# ---------------- EDIT ----------------
elif menu == "Edit":

    st.subheader("Edit Task")

    if not tasks.empty:
        selected = st.selectbox("Select Task", tasks["description"])

        new_desc = st.text_input("New Description", selected)
        new_priority = st.selectbox("New Priority", ["Low", "Medium", "High"])

        if st.button("Update Task"):
            tasks.loc[tasks["description"] == selected, "description"] = new_desc
            tasks.loc[tasks["description"] == selected, "priority"] = new_priority
            tasks.to_csv(FILE, index=False)
            st.success("Task Updated")

# ---------------- UPDATE STATUS ----------------
elif menu == "Update Status":

    st.subheader("Update Task Status")

    if not tasks.empty:
        task_name = st.selectbox("Select Task", tasks["description"])

        new_status = st.selectbox("Status", ["Pending", "Completed"])

        if st.button("Update"):
            tasks.loc[tasks["description"] == task_name, "status"] = new_status
            tasks.to_csv(FILE, index=False)
            st.success("Status Updated")

# ---------------- PREDICT ----------------
elif menu == "Predict":

    text = st.text_input("Enter task")

    if st.button("Predict"):

        rule_p = auto_priority(text)

        if trained:
            ml_p = le.inverse_transform(model.predict([text]))[0]
        else:
            ml_p = "Medium"

        st.write(f"Rule-Based: {rule_p}")
        st.write(f"ML Prediction: {ml_p}")

# ---------------- ANALYTICS ----------------
elif menu == "Analytics":

    st.subheader("📊 Task Analytics Dashboard")

    if tasks.empty:
        st.warning("No data available")
    else:
        tasks["deadline"] = pd.to_datetime(tasks["deadline"])

        # Priority Chart
        priority_counts = tasks["priority"].value_counts()

        st.write("### Priority Distribution")

        fig1, ax1 = plt.subplots()
        ax1.pie(priority_counts, labels=priority_counts.index, autopct='%1.1f%%')
        st.pyplot(fig1)

        st.write("### Priority Count")

        fig2, ax2 = plt.subplots()
        priority_counts.plot(kind='bar', ax=ax2)
        st.pyplot(fig2)

        # Deadline Chart
        st.write("### Tasks by Deadline")

        deadline_counts = tasks.groupby(tasks["deadline"].dt.date).size()

        fig3, ax3 = plt.subplots()
        deadline_counts.plot(kind='bar', ax=ax3)
        st.pyplot(fig3)

        # Status Chart
        st.write("### Task Completion Status")

        status_counts = tasks["status"].value_counts()

        fig4, ax4 = plt.subplots()
        status_counts.plot(kind='bar', ax=ax4)
        st.pyplot(fig4)

        # Insights
        st.write("### 📌 Insights")

        total = len(tasks)
        completed = len(tasks[tasks["status"] == "Completed"])
        pending = len(tasks[tasks["status"] == "Pending"])

        st.write(f"Total Tasks: {total}")
        st.write(f"Completed Tasks: {completed}")
        st.write(f"Pending Tasks: {pending}")

        if pending > completed:
            st.warning("⚠️ You have many pending tasks!")
        else:
            st.success("✅ Good job! You're completing tasks well!")