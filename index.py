import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print("Hello Akbota")

file_path = "datasets/abalone.data"
columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
           'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']


df = pd.read_csv(file_path, header=None, names=columns)


print(df.head())

def show_scat_plot_with_name_Rings(column_name,ff='mm'):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=column_name, y='Rings', hue='Sex', palette='Set2', s=100)
    plt.title(str(column_name) + ' vs Rings (Age)')
    plt.xlabel(str(column_name) + ', ' + ff)
    plt.ylabel('Rings (Age without 1.5)')
    plt.legend(title='Sex')
    plt.grid(True)
    plt.show()

show_scat_plot_with_name_Rings("Length")
show_scat_plot_with_name_Rings("Height")
show_scat_plot_with_name_Rings("Diameter")
show_scat_plot_with_name_Rings("Whole weight",'gg')
show_scat_plot_with_name_Rings("Viscera weight",'gg')
show_scat_plot_with_name_Rings("Shell weight",'gg')







corr_matrix = df[['Length', 'Diameter', 'Height', 'Whole weight',
                  'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
