import seaborn as sns
# Plot the confusion matrix
cm = confusion_matrix(predicted_label,actual_label)
sns.heatmap(cm,annot=True)
plt.show()