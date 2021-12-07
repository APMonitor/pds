# convert scaled values back to dataframe
s_train_df = pd.DataFrame(s_train, columns=train.columns.values)
s_test_df = pd.DataFrame(s_test, columns=test.columns.values)