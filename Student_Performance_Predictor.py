import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load(filename):
    df = pd.read_csv(filename)

    # calculate average score and define academic success
    df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
    df['Label'] = (df['average_score'] >= 70).astype(int)
    # dropping to get average score
    df.drop(['math score', 'reading score', 'writing score', 'average_score'], axis=1, inplace=True)
    # perform one-hot encoding for 'race/ethnicity' and 'parental level of education'
    df = pd.get_dummies(df, columns=['race/ethnicity', 'parental level of education'])
    # converting rest of features to binary
    df['lunch'] = df['lunch'].map({'free/reduced': 1, 'standard': 0})
    df['test preparation course'] = df['test preparation course'].map({'completed': 1, 'none': 0})
    df['gender'] = df['gender'].map({'male': 1, 'female': 0})
    return df

def get_p_x_given_y(x_column, y_column, df):
    probabilities = {}
    x_classes = df[x_column].unique()

    for x_class in x_classes:
        probabilities[x_class] = []
        for y_class in [0, 1]:
            count_x_and_y = len(df[(df[x_column] == x_class) & (df[y_column] == y_class)])
            count_y = len(df[df[y_column] == y_class])
            probability = (count_x_and_y + 1) / (count_y + len(x_classes))
            probabilities[x_class].append(probability)

    return probabilities

def get_all_p_x_given_y(y_column, df):
    all_p_x_given_y = {}

    for column in df.columns:
        if column != y_column:
            all_p_x_given_y[column] = get_p_x_given_y(column, y_column, df)

    return all_p_x_given_y

def get_p_y(y_column, df):
    all_y_1 = len(df[df[y_column] == 1])
    p_y = all_y_1 / len(df[y_column])
    return p_y

def joint_prob(xs, y, all_p_x_given_y, p_y):
    prob = p_y
    for feature_name in xs.index:
        feature_value = xs[feature_name]
        if feature_value in all_p_x_given_y[feature_name]:
            # multiply by prob of the feature given y
            prob *= all_p_x_given_y[feature_name][feature_value][y]
        else:
            # if feature not in dict, use small prob
            prob *= 1e-6
    return prob

def get_prob_y_given_x(y, xs, all_p_x_given_y, p_y):
    joint_probability_y = joint_prob(xs, y, all_p_x_given_y, p_y)
    #find total joint prob
    total_joint_probability = sum(
        joint_prob(xs, class_value, all_p_x_given_y, p_y if class_value == 1 else 1 - p_y) for class_value in [0, 1])
    # normalize and return
    prob_y_given_x = joint_probability_y / total_joint_probability if total_joint_probability > 0 else 0

    return prob_y_given_x

def compute_accuracy(all_p_x_given_y, p_y, df):
    y_test = df["Label"]
    X_test = df.drop(columns="Label")
    num_correct = 0
    total = len(y_test)
    for i, row in X_test.iterrows():
        #calculate prob of label being 1
        prob_y_equals_1 = get_prob_y_given_x(1, row, all_p_x_given_y, p_y)
        # if prob success >= 0.5, predict success, else predict not success
        predicted = 1 if prob_y_equals_1 >= 0.5 else 0
        # increment num_correct if prediction matches label
        if predicted == y_test.iloc[i]:
            num_correct += 1
    accuracy = num_correct / total
    return accuracy

def main():
    df = load("StudentsPerformance.csv")
    #split to train and testing
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
    #reset the indices
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    #compute model parameters (P(Y), P(X_i|Y))
    all_p_x_given_y = get_all_p_x_given_y("Label", df_train)
    p_y = get_p_y("Label", df_train)
    #print accuracy on training and test sets
    print(f"Training accuracy: {compute_accuracy(all_p_x_given_y, p_y, df_train)}")
    print(f"Test accuracy: {compute_accuracy(all_p_x_given_y, p_y, df_test)}")


if __name__ == "__main__":
    main()