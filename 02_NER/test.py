import pandas as pd

true = pd.read_csv('./rule_result.csv').values
index = true[:, 0]
true = true[:, 1:]
false = pd.read_csv('./l_result.csv', index_col=0).values

false[:, 2] = ['[]' if item == 'None' else item for item in false[:, 2]]
result = true.copy()
different = []
for i in range(true.shape[0]):
    t = true[i]
    f = false[i]
    if len(str(t)) != len(str(f)):
        different.append(
            {
                'id': index[i],
                'true': t,
                'false': f,
            }
        )
        result[i] = [str(list(set(eval(i)+eval(j)))) for i, j in zip(t, f)]

# pd.DataFrame([index, result[:, 0], result[:, 1], result[:, 2]]).T.\
#     to_csv('./combine_result.csv', header=['id', 'n_crop', 'n_disease', 'n_medicine'], index=False)
print(1)
