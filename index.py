import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных из файла
file_path = "datasets/abalone.data"  # Укажите путь к вашему файлу
columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
           'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

# Загрузка данных в DataFrame
df = pd.read_csv(file_path, header=None, names=columns)

# Просмотр первых строк
print(df.head())

# Scatter Plot: Length vs Rings
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Length', y='Rings', hue='Sex', palette='Set2', s=100)
plt.title('Length vs Rings (Age)')
plt.xlabel('Length (mm)')
plt.ylabel('Rings (Age without 1.5)')
plt.legend(title='Sex')
plt.grid(True)
plt.show()

# Корреляционная матрица
corr_matrix = df[['Length', 'Diameter', 'Height', 'Whole weight',
                  'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
