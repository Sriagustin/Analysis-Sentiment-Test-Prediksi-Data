import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sentiment_analysis import preprocess_data, train_sentiment_models, predict_sentiment
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Sidebar: Title and File Upload
st.sidebar.title("Sentiment Analysis and Data Prediction Test")
st.sidebar.markdown("""
<div style="text-align: justify;">This application can only be used with datasets containing the columns <b> Text Tweet </b> 
                    and <b> Sentiment </b>, with content in either Indonesian or English. It can also be used for predicting 
                    new data by entering sentences or reviews from various sources.</div>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
model_view = st.sidebar.radio("Select Model to Display:", ["Naive Bayes", "SVM"])

if uploaded_file:
    # Read and preview dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

    # Check if required columns exist in the dataset
    if 'Text Tweet' not in data.columns or 'Sentiment' not in data.columns:
        st.error("Dataset harus memiliki kolom 'Text Tweet' dan 'Sentiment'.")
    else:
        # Remove sentiment categories with fewer than 7 entries
        data = data.groupby('Sentiment').filter(lambda x: len(x) >= 7)

        if data.empty:
            st.error("Tidak ada cukup data setelah menghapus kategori dengan kurang dari 7 entri.")
        else:
            # Preprocess data
            data = preprocess_data(data)

            # WordCloud for each sentiment
            st.write("### WordClouds by Sentiment")
            sentiments = data['Sentiment'].unique()
            cols = st.columns(len(sentiments))

            for i, sentiment in enumerate(sentiments):
                with cols[i]:
                    st.write(f"#### {sentiment} Sentiment")
                    sentiment_data = data[data['Sentiment'] == sentiment]
                    text = " ".join(sentiment_data['Text Tweet'])

                    # Generate WordCloud
                    wordcloud = WordCloud(width=400, height=400, background_color="white").generate(text)

                    plt.figure(figsize=(4, 4))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(plt)

            # Sentiment distribution pie chart
            st.write("### Sentiment Distribution")
            sentiment_counts = data['Sentiment'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

            # Train models
            nb_model, svm_model, vectorizer, X_test, y_test = train_sentiment_models(data)

            # K-Fold Cross Validation
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

            if model_view == "Naive Bayes":
                st.write("### Naive Bayes Model Evaluation Metrics")
                nb_scores = cross_val_score(nb_model, X_test, y_test, cv=skf, scoring='accuracy')
                st.write(f"10-Fold Accuracy: {nb_scores.mean():.2f} (\u00B1 {nb_scores.std():.2f})")

                nb_y_pred = nb_model.predict(X_test)
                nb_accuracy = accuracy_score(y_test, nb_y_pred)
                nb_precision = precision_score(y_test, nb_y_pred, average='macro')
                nb_recall = recall_score(y_test, nb_y_pred, average='macro')
                nb_f1 = f1_score(y_test, nb_y_pred, average='macro')

                # Display metrics
                st.write(f"Accuracy: {nb_accuracy:.2f}")
                st.write(f"Precision (Macro avg): {nb_precision:.2f}")
                st.write(f"Recall (Macro avg): {nb_recall:.2f}")
                st.write(f"F1-Score (Macro avg): {nb_f1:.2f}")

                # Confusion Matrix
                st.write("#### Naive Bayes Confusion Matrix")
                nb_cm = confusion_matrix(y_test, nb_y_pred)
                fig, ax = plt.subplots(figsize=(6, 6))
                sns.heatmap(nb_cm, annot=True, fmt="d", cmap="Blues", xticklabels=nb_model.classes_, yticklabels=nb_model.classes_)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Naive Bayes Confusion Matrix')
                st.pyplot(fig)

            elif model_view == "SVM":
                st.write("### SVM Model Evaluation Metrics")
                svm_scores = cross_val_score(svm_model, X_test, y_test, cv=skf, scoring='accuracy')
                st.write(f"10-Fold Accuracy: {svm_scores.mean():.2f} (\u00B1 {svm_scores.std():.2f})")

                svm_y_pred = svm_model.predict(X_test)
                svm_accuracy = accuracy_score(y_test, svm_y_pred)
                svm_precision = precision_score(y_test, svm_y_pred, average='macro')
                svm_recall = recall_score(y_test, svm_y_pred, average='macro')
                svm_f1 = f1_score(y_test, svm_y_pred, average='macro')

                # Display metrics
                st.write(f"Accuracy: {svm_accuracy:.2f}")
                st.write(f"Precision (Macro avg): {svm_precision:.2f}")
                st.write(f"Recall (Macro avg): {svm_recall:.2f}")
                st.write(f"F1-Score (Macro avg): {svm_f1:.2f}")

                # Confusion Matrix
                st.write("#### SVM Confusion Matrix")
                svm_cm = confusion_matrix(y_test, svm_y_pred)
                fig, ax = plt.subplots(figsize=(6, 6))
                sns.heatmap(svm_cm, annot=True, fmt="d", cmap="Blues", xticklabels=svm_model.classes_, yticklabels=svm_model.classes_)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('SVM Confusion Matrix')
                st.pyplot(fig)

            # Real-time sentiment prediction
            st.write("### Real-Time Sentiment Prediction")
            user_input = st.text_input("Enter text to analyze sentiment:")
            if user_input:
                if model_view == 'Naive Bayes':
                    sentiment = predict_sentiment(nb_model, vectorizer, user_input)
                else:
                    sentiment = predict_sentiment(svm_model, vectorizer, user_input)

                st.write(f"Predicted Sentiment: {sentiment}")
else:
    st.info("Please upload a CSV file to get started.")

st.sidebar.markdown("""
<hr>
<h4 style="text-align: center;">Created by [SRI AGUSTIN]</h4>
<p style="text-align: center;">
    Check out the source code on <a href="https://github.com/Sriagustin/Analysis-Sentiment-Test-Prediksi-Data.git" target="_blank">GitHub</a>
</p>
""", unsafe_allow_html=True)