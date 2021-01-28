import pandas as pd
import os
import json

anno_folder = "train/annos/"
df = pd.DataFrame(columns=['category', 'occlusion'])
i=1
for filename in os.listdir(anno_folder):
    with open(anno_folder + filename) as json_file:
        data = json.load(json_file)
        filtered_data = {k: v for k, v in data.items() if k.startswith('item')}

        for key, v in filtered_data.items():
            df['category'] = v['category_id']
            df['occlusion'] = v['occlusion']
    if i%100 == 0:
        print(i)
    i+=1
print(df)

df.to_pickle("df300.pkl")
