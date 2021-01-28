import pandas as pd
import os
import json

anno_folder = "train/annos/"
df_tot = pd.DataFrame(columns=['category', 'occlusion'])
i=1
for filename in os.listdir(anno_folder):
    with open(anno_folder + filename) as json_file:
        data = json.load(json_file)
        filtered_data = {k: v for k, v in data.items() if k.startswith('item')}

        for key, v in filtered_data.items():
            df = pd.DataFrame([[v['category_id'], v['occlusion']]], columns=['category', 'occlusion'])
            df_tot = df_tot.append(df)
    if i % 100:
        print(i)
    i+=1

df_tot = df_tot.reset_index(drop=True)
print(df_tot)

df.to_pickle("df300.pkl")
