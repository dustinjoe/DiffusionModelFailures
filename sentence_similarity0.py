from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

# Two lists of sentences
sentences1 = ['An abandoned vehicle',
             'A camel',
             'A horse',
             'A boat',
             'A bus',
             'A bear in a red forest',
             'A bird',
             'A couch'
]

sentences2 = ['A forest covered with snow',
              'A forest',
              'A yellow flower field',
              'A desert',
              'A person',
              'A car stuck in the forest',
              'A flower',
              'A dog sitting in the living room'
]

#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings2)

#Output the pairs with their score
for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
    
'''
(base) xyzhou@xyzhou-HP-ZBook-17-G4:/media/xyzhou/extDisk2t1/DeepFake_Audio$ python sentence_similarity0.py 
An abandoned vehicle 		 A forest covered with snow 		 Score: 0.1767
A camel 		 A forest 		 Score: 0.3656
A horse 		 A yellow flower field 		 Score: 0.2314
A boat 		 A desert 		 Score: 0.1921
A bus 		 A person 		 Score: 0.2697
A bear in a red forest 		 A car stuck in the forest 		 Score: 0.5077
A bird 		 A flower 		 Score: 0.4528
A couch 		 A dog sitting in the living room 		 Score: 0.3366

'''
