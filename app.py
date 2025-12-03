import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# --------------------------
# Classification Module
# --------------------------
def classification_models():
    st.header("📘 Classification Models")

    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("### Preview of Data")
        st.dataframe(df.head())

        target = st.selectbox("Select Target Column", df.columns)

        st.write("### Feature Selection")
        use_all = st.checkbox("Use All Features (except target)", value=True)

        if use_all:
            features = df.columns.drop(target).tolist()
        else:
            features = st.multiselect("Select Feature Columns", df.columns.drop(target))

        if features:

            X = df[features]
            y = df[target]

            # CLEANING
            X = X.replace(["?", "NA", "nan", "missing", ""], np.nan)

            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = X[col].fillna(X[col].mode()[0])
                else:
                    X[col] = X[col].fillna(X[col].mean())

            # Encode categorical columns
            X = pd.get_dummies(X, drop_first=True)

            # Encode target
            if y.isnull().sum() > 0:
                y = y.fillna(y.mode()[0])

            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)

            st.write("---")
            model_choice = st.selectbox(
                "Choose Classification Model",
                [
                    "Logistic Regression",
                    "Decision Tree Classifier",
                    "Random Forest Classifier",
                    "Support Vector Machine (SVM)",
                    "K-Nearest Neighbors (KNN)"
                ]
            )

            test_size = st.slider("Test Size (%)", 10, 50, 20)

            # PARAMETERS
            if model_choice == "Decision Tree Classifier":
                max_depth = st.slider("Maximum Depth of the Tree", 1, 20, 5)

            if model_choice == "Random Forest Classifier":
                n_estimators = st.slider("Number of Trees", 10, 200, 100)

            if model_choice == "Support Vector Machine (SVM)":
                kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])

            if model_choice == "K-Nearest Neighbors (KNN)":
                k_value = st.slider("K Value (Neighbors)", 1, 25, 5)

            # Train button
            if st.button("Train Model"):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42
                )

                # MODEL INITIALIZATION
                if model_choice == "Logistic Regression":
                    model = LogisticRegression(max_iter=3000)

                elif model_choice == "Decision Tree Classifier":
                    model = DecisionTreeClassifier(
                        random_state=42,
                        max_depth=max_depth
                    )

                elif model_choice == "Random Forest Classifier":
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        random_state=42
                    )

                elif model_choice == "Support Vector Machine (SVM)":
                    model = SVC(kernel=kernel)

                else:  # KNN
                    model = KNeighborsClassifier(n_neighbors=k_value)

                # TRAIN
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                # SAVE MODEL INTO SESSION STATE FOR TREE VISUALIZATION
                if model_choice == "Decision Tree Classifier":
                    st.session_state['trained_model'] = model
                    st.session_state['X_columns'] = X.columns.tolist()

                # METRICS
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average='weighted')

                st.success(f"Accuracy: {acc:.2f}")
                st.write(f"F1 Score (Weighted): {f1:.2f}")

                # CONFUSION MATRIX
                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test, preds)

                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(fig)

                # REPORT
                st.write("### Classification Report")
                report = classification_report(y_test, preds, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                # FEATURE IMPORTANCE (Decision Tree / RF)
                if model_choice in ["Decision Tree Classifier", "Random Forest Classifier"]:
                    st.subheader("Feature Importance")

                    importance_df = pd.DataFrame({
                        "Feature": X.columns,
                        "Importance": model.feature_importances_
                    }).sort_values(by="Importance", ascending=False)

                    st.dataframe(importance_df)

                # PREDICTIONS SUMMARY
                st.write("---")
                st.subheader("Predictions (Sample)")
                st.dataframe(pd.DataFrame({"Actual": y_test, "Predicted": preds}).head(20))

                st.write("### Features Used")
                st.write(", ".join(features))

                st.info("Model training complete. Scroll up to explore the results.")

            # DECISION TREE VISUALIZATION
            if model_choice == "Decision Tree Classifier":
                st.write("---")
                st.subheader("Decision Tree Visualization")

                show_tree = st.checkbox(
                    "Show Full Decision Tree (Large & Clear Display)",
                    key="show_dt"
                )

                if show_tree:
                    if 'trained_model' in st.session_state and 'X_columns' in st.session_state:

                        st.write("### 🌳 Full Decision Tree Structure")

                        fig, ax = plt.subplots(figsize=(32, 24))
                        plot_tree(
                            st.session_state['trained_model'],
                            feature_names=st.session_state['X_columns'],
                            filled=True,
                            rounded=True,
                            fontsize=9
                        )
                        st.pyplot(fig)

                    else:
                        st.warning("⚠️ Please train the Decision Tree model first.")

# --------------------------
# Regression Module
# --------------------------
def regression_models():
    st.header("📉 Regression Models")

    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"], key="reg_upload")
    
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("### Preview of Data")
        st.dataframe(df.head())

        target = st.selectbox("Select Target Column", df.columns)

        use_all = st.checkbox("Use All Features (except target)", value=True)

        if use_all:
            features = df.columns.drop(target).tolist()
        else:
            features = st.multiselect("Select Feature Columns", df.columns.drop(target))
            if not features:
                st.warning("Please select at least one feature.")
                return

        X = df[features].copy()
        y = df[target].copy()

        # CLEANING & ENCODING
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].fillna(X[col].mode()[0])
                dummies = pd.get_dummies(X[col], drop_first=True)
                X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
            else:
                X[col] = X[col].fillna(X[col].mean())

        # If target is categorical
        if y.dtype == 'object':
            y = y.fillna(y.mode()[0])
            y = pd.factorize(y)[0]

        test_size = st.slider("Test Size (%)", 10, 50, 20)

        # MODEL SELECTION
        model_choice = st.selectbox(
            "Choose Regression Model",
            [
                "Linear Regression",
                "Decision Tree Regressor",
                "Random Forest Regressor",
                "Support Vector Regression (SVR)",
                "KNN Regression"
            ]
        )

        # Model parameters
        if model_choice == "Decision Tree Regressor":
            max_depth = st.slider("Maximum Depth", 1, 20, 5)

        if model_choice == "Random Forest Regressor":
            n_estimators = st.slider("Number of Trees", 10, 500, 100)
            max_depth = st.slider("Maximum Depth", 1, 20, 5)

        if model_choice == "Support Vector Regression (SVR)":
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])

        if model_choice == "KNN Regression":
            k_value = st.slider("Number of Neighbors (K)", 1, 25, 5)

        # TRAIN MODEL
        if st.button(f"Train {model_choice}", key="train_reg"):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=42
            )

            # Instantiate model
            if model_choice == "Linear Regression":
                model = LinearRegression()

            elif model_choice == "Decision Tree Regressor":
                model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)

            elif model_choice == "Random Forest Regressor":
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )

            elif model_choice == "Support Vector Regression (SVR)":
                model = SVR(kernel=kernel)

            elif model_choice == "KNN Regression":
                model = KNeighborsRegressor(n_neighbors=k_value)

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Save state
            st.session_state['reg_model'] = model
            st.session_state['preds'] = preds
            st.session_state['y_test'] = y_test
            st.session_state['X_columns'] = X.columns
            st.session_state['model_choice'] = model_choice

            # Metrics
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, preds)

            st.success(f"R² Score: {r2:.3f}")
            st.info(f"MAE: {mae:.3f} | MSE: {mse:.3f} | RMSE: {rmse:.3f}")

            # Linear Regression extra
            if model_choice == "Linear Regression":
                st.write("### Coefficients")
                coef_df = pd.DataFrame({
                    "Feature": X_train.columns,
                    "Coefficient": model.coef_
                })
                st.dataframe(coef_df)

                st.write("### Regression Equation")
                eq = f"y = {model.intercept_:.3f}"
                for i, f in enumerate(X_train.columns):
                    eq += f" + ({model.coef_[i]:.3f} * {f})"
                st.code(eq)

            # Feature importance
            if model_choice in ["Decision Tree Regressor", "Random Forest Regressor"]:
                st.write("### Feature Importance")
                imp = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": model.feature_importances_
                }).sort_values("Importance", ascending=False)
                st.dataframe(imp)

    # PERSIST OUTPUTS
    if "reg_model" in st.session_state:
        model = st.session_state['reg_model']
        preds = st.session_state['preds']
        y_test = st.session_state['y_test']
        model_choice = st.session_state['model_choice']

        st.write("### Actual vs Predicted Values")
        results_df = pd.DataFrame({
            "Actual": np.array(y_test).flatten(),
            "Predicted": np.array(preds).flatten()
        })
        st.dataframe(results_df)

        # Scatter plot
        st.write("### Scatter Plot")
        fig, ax = plt.subplots(figsize=(7,5))
        sns.scatterplot(x=y_test, y=preds)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

        # Tree visualization
        if model_choice == "Decision Tree Regressor":
            if st.checkbox("Show Full Decision Tree"):
                fig, ax = plt.subplots(figsize=(32,24))
                plot_tree(
                    model,
                    feature_names=st.session_state['X_columns'],
                    filled=True,
                    rounded=True
                )
                st.pyplot(fig)

# --------------------------
# Clustering Module
# --------------------------
def clustering_models():
    st.header("📊 Clustering Algorithms")

    # Upload dataset
    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"], key="cluster_upload")
    
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("### Preview of Data")
        st.dataframe(df.head())

        # Feature selection
        st.write("### Feature Selection")
        features = st.multiselect("Select Feature Columns", df.columns, default=df.columns.tolist())
        if not features:
            st.warning("Please select at least one feature.")
            return

        X = df[features].copy()

        # Handle missing values & encode categorical
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].fillna(X[col].mode()[0])
                dummies = pd.get_dummies(X[col], drop_first=True)
                X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
            else:
                X[col] = X[col].fillna(X[col].mean())

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.write("---")
        # Algorithm selection
        algo_choice = st.selectbox(
            "Choose Clustering Algorithm",
            ["K-Means", "Hierarchical Clustering", "DBSCAN"]
        )

        # Algorithm-specific parameters
        if algo_choice == "K-Means":
            st.write("### Elbow Method to Choose K")
            max_k = st.slider("Max K for Elbow Method", 2, 10, 6)
            wcss = []
            for k in range(1, max_k + 1):
                km = KMeans(n_clusters=k, random_state=42)
                km.fit(X_scaled)
                wcss.append(km.inertia_)

            # Plot Elbow
            plt.figure(figsize=(7,5))
            plt.plot(range(1, max_k+1), wcss, marker='o')
            plt.xlabel("Number of Clusters (K)")
            plt.ylabel("WCSS")
            plt.title("Elbow Method")
            st.pyplot(plt.gcf())

            n_clusters = st.slider("Select Number of Clusters (K)", 2, max_k, 3)

        elif algo_choice == "Hierarchical Clustering":
            n_clusters_h = st.slider("Number of Clusters (to cut dendrogram)", 2, 10, 3)

        elif algo_choice == "DBSCAN":
            eps = st.slider("Epsilon (Neighborhood Radius)", 0.1, 5.0, 0.5, step=0.1)
            min_samples = st.slider("Minimum Samples", 1, 10, 5)

        if st.button(f"Run {algo_choice}"):
            # K-Means
            if algo_choice == "K-Means":
                model = KMeans(n_clusters=n_clusters, random_state=42)
                labels = model.fit_predict(X_scaled)
                st.write("### Cluster Assignments")
                st.dataframe(pd.DataFrame({"Cluster": labels}))

                st.write("### Cluster Centers")
                st.dataframe(pd.DataFrame(model.cluster_centers_, columns=X.columns))

            # Hierarchical
            elif algo_choice == "Hierarchical Clustering":
                model = AgglomerativeClustering(n_clusters=n_clusters_h)
                labels = model.fit_predict(X_scaled)
                st.write("### Cluster Assignments")
                st.dataframe(pd.DataFrame({"Cluster": labels}))

                # Dendrogram
                st.write("### Dendrogram")
                linked = linkage(X_scaled, method='ward')
                plt.figure(figsize=(14, 7))
                dendrogram(linked, distance_sort='ascending', show_leaf_counts=True)
                st.pyplot(plt.gcf())

            # DBSCAN
            elif algo_choice == "DBSCAN":
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X_scaled)
                st.write("### Cluster Assignments (-1 = Outlier)")
                st.dataframe(pd.DataFrame({"Cluster": labels}))

            # 2D scatter plot of clusters (first two features)
            st.write("### Cluster Scatter Plot (First 2 Features)")
            plt.figure(figsize=(8,6))
            unique_labels = np.unique(labels)
            colors = sns.color_palette("hls", len(unique_labels))
            for lbl, color in zip(unique_labels, colors):
                mask = labels == lbl
                plt.scatter(
                    X_scaled[mask, 0], X_scaled[mask, 1],
                    label=f"Cluster {lbl}", color=color
                )
            plt.xlabel(features[0])
            plt.ylabel(features[1])
            plt.title(f"{algo_choice} Clusters")
            plt.legend()
            st.pyplot(plt.gcf())

            # Cluster sizes
            st.write("### Number of Points per Cluster")
            cluster_sizes = pd.Series(labels).value_counts().sort_index()
            st.dataframe(cluster_sizes)

# --------------------------
# Streamlit App
# --------------------------
# Page configuration
st.set_page_config(
    page_title="Unified ML Platform",
    layout="wide",
    page_icon="🌾"
)

# Styling
st.markdown("""
    <style>
        .main {
            background-color: #faf8f3;
        }
        .css-18e3th9 {
            padding: 2rem;
            border-radius: 12px;
            background: #ffffff;
            box-shadow: 0 0 8px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #4b3f2f;
        }
        .sidebar .sidebar-content {
            background-color: #f0ebe1;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🌾 Unified Machine Learning Platform")
st.write("A simple, elegant platform to try out various machine learning algorithms.")

# Sidebar Menu
option = st.sidebar.selectbox(
    "Choose a category",
    ["Home", "Classification", "Regression", "Clustering"]
)

# Routing
if option == "Home":
    st.subheader("Welcome!")
    st.write("""
    This platform allows you to:
    - Upload datasets  
    - Explore data visually  
    - Train ML models  
    - Download trained models  
    - Visualize results  

    Select a category from the left sidebar to begin.
    """)

elif option == "Classification":
    classification_models()

elif option == "Regression":
    regression_models()

elif option == "Clustering":
    clustering_models()
