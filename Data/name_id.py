import pickle
with open('disease_map.pkl', 'rb') as file:
    data = pickle.load(file)
    print(data)
    data=str(data)
    ft = open('disease_map_id.txt', 'w')  
    ft.write(data)  
with open('gene_map.pkl', 'rb') as file:
    data = pickle.load(file)
    print(data)
    data=str(data)
    ft = open('gene_map_id.txt', 'w')  
    ft.write(data)  
with open('microbe_map.pkl', 'rb') as file:
    data = pickle.load(file)
    print(data)
    data=str(data)
    ft = open('microbe_map_id.txt', 'w')  
    ft.write(data)  