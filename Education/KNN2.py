from docx import Document
import networkx as nx
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Function to preprocess text
def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Initialize Porter Stemmer
    porter = PorterStemmer()
    
    # Stem words
    stemmed_tokens = [porter.stem(word) for word in filtered_tokens]
    
    # Return preprocessed text
    return stemmed_tokens

# Function to read preprocessed data from a document file
def read_preprocessed_data(file_path):
    try:
        document = Document(file_path)
        # Extract text from paragraphs
        text = '\n'.join([paragraph.text for paragraph in document.paragraphs])
        return preprocess_text(text)
    except Exception as e:
        print(f"Error reading preprocessed data from document: {e}")
        return None

# Function to create a directed graph from the list of tokens
def create_directed_graph(tokens):
    try:
        G = nx.DiGraph()
        for i in range(len(tokens) - 1):
            source = tokens[i]
            target = tokens[i + 1]
            G.add_edge(source, target)
        return G
    except Exception as e:
        print(f"Error creating directed graph: {e}")
        return None
    
# Function to calculate the maximal common subgraph size between two graphs
def calculate_maximal_common_subgraph_size(graph_1, graph_2):
    num_nodes_g1 = len(graph_1)
    num_nodes_g2 = len(graph_2)
    common_nodes = set(graph_1.nodes()) & set(graph_2.nodes())
    common_subgraph_size = len(graph_1.subgraph(common_nodes))
    max_size = max(num_nodes_g1, num_nodes_g2)
    distance = 1 - (common_subgraph_size / max_size)
    return distance



# Function to classify test documents using KNN algorithm
def classify_knn(test_graph, training_graphs, labels, k):
    distances = []
    for i, train_graph in enumerate(training_graphs):
        distance = calculate_maximal_common_subgraph_size(test_graph, train_graph)
        distances.append((distance, labels[i]))  # Store distance along with label

    # Sort training graphs based on distances
    distances.sort(key=lambda x: x[0])

    # Select k-nearest neighbors
    neighbors = distances[:k]

    # Get the class labels of the k-nearest neighbors
    labels_of_neighbors = [label for _, label in neighbors]

    # Find the majority class label
    majority_label = Counter(labels_of_neighbors).most_common(1)[0][0]

    return majority_label

def annotate_documents(test_file_paths):
    true_labels = []
    for file_path in test_file_paths:
        print(f"Document: {file_path}")
        label = input("Enter the label for this document: ")
        true_labels.append(label)
    return true_labels

# Function to create TF-IDF vectors from text data
def create_tfidf_vectors(file_paths, vectorizer=None):
    documents = []
    for file_path in file_paths:
        tokens = read_preprocessed_data(file_path)
        if tokens is not None:
            documents.append(' '.join(tokens))
    
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
    else:
        tfidf_matrix = vectorizer.transform(documents)
    
    return tfidf_matrix, vectorizer

# Function to classify test documents using SVM
def classify_svm(test_file_paths, training_tfidf_matrix, labels, vectorizer):
    test_tfidf_matrix, _ = create_tfidf_vectors(test_file_paths, vectorizer)
    X_train, X_test, y_train, y_test = train_test_split(training_tfidf_matrix, labels, test_size=0.2, random_state=42)
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    predicted_labels = svm_model.predict(test_tfidf_matrix)
    return predicted_labels

def extract_graph_features(graph):
    # Extract features from the graph
    features = []
    # Example feature: Average node degree
    avg_degree = sum(dict(graph.degree()).values()) / len(graph)
    features.append(avg_degree)
    return features

# Main function
def main():
    # File paths of the documents containing preprocessed data
    '''file_paths = [
        "d:\\6 semester\\GT project\\educationdata\\scraped_data_1.docx",
        "d:\\6 semester\\GT project\\educationdata\\scraped_data_2.docx",
        "d:\\6 semester\\GT project\\educationdata\\scraped_data_3.docx",
        "d:\\6 semester\\GT project\\educationdata\\scraped_data_4.docx",
        "d:\\6 semester\\GT project\\educationdata\\scraped_data_5.docx",
        "d:\\6 semester\\GT project\\educationdata\\scraped_data_6.docx",
        "d:\\6 semester\\GT project\\educationdata\\scraped_data_7.docx",
        "d:\\6 semester\\GT project\\educationdata\\scraped_data_8.docx",
        "d:\\6 semester\\GT project\\educationdata\\scraped_data_9.docx",
        "d:\\6 semester\\GT project\\educationdata\\scraped_data_10.docx",
        "d:\\6 semester\\GT project\\educationdata\\scraped_data_11.docx",
        "d:\\6 semester\\GT project\\educationdata\\scraped_data_12.docx"
    ]

    # Labels for each document (assuming binary classification, modify as needed)
    # Assign arbitrary labels to each document
    labels = [f"class{i}" for i in range(len(file_paths))]

    # Create graphs and extract labels for each file
    graphs = []
    for file_path in file_paths:
        tokens = read_preprocessed_data(file_path)
        if tokens is not None:
            graph = create_directed_graph(tokens)
            if graph is not None:
                graphs.append(graph)

    # File paths of the test documents
    test_file_paths = [
        "d:\\6 semester\\GT project\\educationdata\\scraped_data_13.docx",
        "d:\\6 semester\\GT project\\educationdata\\scraped_data_14.docx",
        "d:\\6 semester\\GT project\\educationdata\\scraped_data_15.docx"
    ]'''

    topics = ['finance', 'food', 'education']
    features = []
    labels = []
    features2=[]
    labels2=[]

    # Compute distances between graphs within each topic
    for topic in topics:
        for doc_num1 in range(1, 13):  # 12 training documents per topic
            file_path1 = f"D:\\6 semester\\GT project\\{topic}data\\scraped_data_{doc_num1}.docx"
            words1 = read_preprocessed_data(file_path1)
            graph1 = create_directed_graph(words1)
            if graph1 is not None:
                for doc_num2 in range(13, 16):  # 3 testing documents per topic
                    file_path2 = f"D:\\6 semester\\GT project\\{topic}data\\scraped_data_{doc_num2}.docx"  # Corrected file path
                    words2 = read_preprocessed_data(file_path2)
                    graph2 = create_directed_graph(words2)
                    if graph2 is not None:  
                        distance = calculate_maximal_common_subgraph_size(graph1, graph2)
                        features.append([distance])
                        labels.append(topic)

    desired_test_samples = 9  # Update this with the desired number of samples

    # Calculate the test size based on the desired number of samples
    test_size = desired_test_samples / len(features)

    # Perform the train-test split with the updated test size
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    print("X_train:", X_train)
    print("X_test:", X_test)
    print("y_train:", y_train)
    print("y_test:", y_test)

    # Train k-NN model
    k = 3  # Choose the number of neighbors
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)

    # Predict labels for the test set
    y_pred = knn_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=topics)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=topics, yticklabels=topics)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

        # Extract TP, FP, TN, FN values
    for i, topic in enumerate(topics):
        TP = conf_matrix[i, i]
        FP = sum(conf_matrix[:, i]) - TP
        TN = np.sum(conf_matrix) - (sum(conf_matrix[i, :]) + sum(conf_matrix[:, i]) - TP)
        FN = sum(conf_matrix[i, :]) - TP
    
    print(f"Class: {topic}")
    print("True Positives (TP):", TP)
    print("False Positives (FP):", FP)
    print("True Negatives (TN):", TN)
    print("False Negatives (FN):", FN)

    # Calculate precision, recall, and F1 score for k-NN
    precision_knn = precision_score(y_test, y_pred, average='weighted')
    recall_knn = recall_score(y_test, y_pred, average='weighted')
    f1_score_knn = f1_score(y_test, y_pred, average='weighted')

    print("Precision (k-NN):", precision_knn)
    print("Recall (k-NN):", recall_knn)
    print("F1 Score (k-NN):", f1_score_knn)

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train KNN model
    knn_classifier = KNeighborsClassifier(n_neighbors=5)  # Example: 5 neighbors
    knn_classifier.fit(X_train_tfidf, y_train)

    # Predict with KNN
    knn_predictions = knn_classifier.predict(X_test_tfidf)

    # Evaluate KNN performance
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    knn_precision = precision_score(y_test, knn_predictions, average='weighted')
    knn_recall = recall_score(y_test, knn_predictions, average='weighted')
    knn_f1 = f1_score(y_test, knn_predictions, average='weighted')

    # Train SVM model
    svm_classifier = SVC(kernel='linear')  # Example: Linear SVM
    svm_classifier.fit(X_train_tfidf, y_train)

    # Predict with SVM
    svm_predictions = svm_classifier.predict(X_test_tfidf)

    # Evaluate SVM performance
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    svm_precision = precision_score(y_test, svm_predictions, average='weighted')
    svm_recall = recall_score(y_test, svm_predictions, average='weighted')
    svm_f1 = f1_score(y_test, svm_predictions, average='weighted')

    # Print results
    print("KNN Metrics:")
    print("Accuracy:", knn_accuracy)
    print("Precision:", knn_precision)
    print("Recall:", knn_recall)
    print("F1 Score:", knn_f1)
    print("\nSVM Metrics:")
    print("Accuracy:", svm_accuracy)
    print("Precision:", svm_precision)
    print("Recall:", svm_recall)
    print("F1 Score:", svm_f1)


    '''# Lists to store predicted labels
    predicted_labels = []

    # Iterate over each test document
    for test_document_path in test_file_paths:
        # Load and preprocess test document
        test_tokens = read_preprocessed_data(test_document_path)
        if test_tokens is None:
            continue

        # Create graph for test document
        test_graph = create_directed_graph(test_tokens)
        if test_graph is None:
            continue

        # Classify test document using KNN
        k = 3  # Number of neighbors to consider
        predicted_label = classify_knn(test_graph, graphs, labels, k)
        predicted_labels.append(predicted_label)
        print(f"Predicted label for test document {test_document_path} is: {predicted_label}")

    true_labels = ["class3", "class1", "class1"]  

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Display confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

        # Extract TP, FP, TN, FN values
    TP = conf_matrix[1, 1]  # True Positives
    FP = conf_matrix[0, 1]  # False Positives
    TN = conf_matrix[0, 0]  # True Negatives
    FN = conf_matrix[1, 0]  # False Negatives

    print("True Positives (TP):", TP)
    print("False Positives (FP):", FP)
    print("True Negatives (TN):", TN)
    print("False Negatives (FN):", FN)

# Calculate accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    print("Accuracy:", accuracy)

    # Calculate precision
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    print("Precision:", precision)

    # Calculate recall
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    print("Recall:", recall)

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    print("F1 Score:", f1_score)

    # Compute TF-IDF vectors for SVM
    training_tfidf_matrix, vectorizer = create_tfidf_vectors(file_paths)

    # Classify test documents using SVM
    predicted_labels_svm = classify_svm(test_file_paths, training_tfidf_matrix, labels, vectorizer)

    # Compute confusion matrix for SVM
    conf_matrix_svm = confusion_matrix(true_labels, predicted_labels_svm)

    # Display confusion matrix for SVM
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_svm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (SVM)")
    plt.show()

    # Extract TP, FP, TN, FN values for SVM
    TP_svm = conf_matrix_svm[1, 1]  # True Positives
    FP_svm = conf_matrix_svm[0, 1]  # False Positives
    TN_svm = conf_matrix_svm[0, 0]  # True Negatives
    FN_svm = conf_matrix_svm[1, 0]  # False Negatives

    print("True Positives (TP) for SVM:", TP_svm)
    print("False Positives (FP) for SVM:", FP_svm)
    print("True Negatives (TN) for SVM:", TN_svm)
    print("False Negatives (FN) for SVM:", FN_svm)

    # Calculate accuracy for SVM
    accuracy_svm = (TP_svm + TN_svm) / (TP_svm + TN_svm + FP_svm + FN_svm) if (TP_svm + TN_svm + FP_svm + FN_svm) != 0 else 0
    print("Accuracy for SVM:", accuracy_svm)

    # Calculate precision for SVM
    precision_svm = TP_svm / (TP_svm + FP_svm) if (TP_svm + FP_svm) != 0 else 0
    print("Precision for SVM:", precision_svm)

    # Calculate recall for SVM
    recall_svm = TP_svm / (TP_svm + FN_svm) if (TP_svm + FN_svm) != 0 else 0
    print("Recall for SVM:", recall_svm)

    # Calculate F1 score for SVM
    f1_svm = 2 * (precision_svm * recall_svm) / (precision_svm + recall_svm) if (precision_svm + recall_svm) != 0 else 0
    print("F1 Score for SVM:", f1_svm) '''

if __name__ == "__main__":
    main()
