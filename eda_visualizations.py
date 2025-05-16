import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(df, target_col='Autism_Diagnosis', save_path='class_distribution.png'):
    plt.figure(figsize=(8,6))
    sns.countplot(x=target_col, data=df)
    plt.title('Class Distribution of ASD Traits in Toddler Dataset')
    plt.xlabel('ASD Diagnosis')
    plt.ylabel('Count')
    plt.savefig(save_path)
    plt.close()

def plot_pairplot(df, features, hue='Autism_Diagnosis', save_path='pairplot.png'):
    plt.figure(figsize=(12, 10))
    sns.pairplot(df[features + [hue]], hue=hue, diag_kind='kde', plot_kws={'alpha':0.6})
    plt.suptitle('Pair Plot of Features with ASD Diagnosis Hue', y=1.02)
    plt.savefig(save_path)
    plt.close()

def main():
    # Load dataset
    df = pd.read_csv('data/autism_screening_dataset.csv')
    
    # Define features for pairplot (select numeric features excluding target)
    features = [col for col in df.columns if col != 'Autism_Diagnosis' and df[col].dtype in ['int64', 'float64']]
    
    # Plot and save class distribution
    plot_class_distribution(df)
    
    # Plot and save pairplot
    plot_pairplot(df, features)

if __name__ == '__main__':
    main()
