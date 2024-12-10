# from model import POISet, EmbeddingModel
from POIEmbedding import PreProcess, POI2Vec
from model import POISet, EmbeddingModel
from torch_geometric.nn import Node2Vec
import torch
import torch.utils.data as tud

def preprocess():
    # Preprocess
    filename_boroughs = '/content/drive/MyDrive/Colab_Notebooks/region-embedding-benchmark/region-embedding/baselines/poi-encoder/Census_2020__Tracts_for_San_Francisco.csv'
    filename_pois = '/content/drive/MyDrive/Colab_Notebooks/region-embedding-benchmark/region-embedding/util/data/san-francisco-pois.csv.gz'
    PreProcess(filename_pois, filename_boroughs, h3=False).run() # Add the path to the POI and boroughs data here

def poi2vec_train():
    poi2vec = POI2Vec()
    poi2vec.train()
    poi2vec.save_walks()

def poi_category_embedding():
    poi2vec = POI2Vec()
    poi2vec.read_walks()
    poi2vec.get_global_second_class_walks()

    le_lambda = 1e-8
    second_class_hierarchy_pairs = list(set([tuple(x) for x in poi2vec.pois[["category", "fclass"]].to_numpy()]))
    embedding_size=64
    k = 5

    data = POISet(poi2vec.second_class_number, poi2vec.second_class_walks, poi2vec.global_second_class_walks, k)
    emb_model = EmbeddingModel(embed_size=embedding_size, vocab_size=poi2vec.second_class_number, second_class_hierarchy_pairs=second_class_hierarchy_pairs, le_lambda=le_lambda)

    return data, emb_model

def main():

    batch_size = 2048

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    dataset, model = poi_category_embedding()

    dataloader = tud.DataLoader(dataset, batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)

    for e in range(5):
        print(e, len(dataloader))
        for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
            # print(i)
            input_labels = input_labels.long().to(device)
            pos_labels = pos_labels.long().to(device)
            neg_labels = neg_labels.long().to(device)

            optimizer.zero_grad()
            loss, loss_le = model(input_labels, pos_labels, neg_labels)
            loss.backward()

            optimizer.step()
            if i % 100 == 0:
                print('epoch', e, 'iteration', i, loss.item(), 'loss_le', loss_le.item())

<<<<<<< Updated upstream
    # embedding_weights = model.input_embedding()
    # torch.save(model.state_dict(), "./data/poi-encoder-chicago.tensor")
=======
    embedding_weights = model.input_embedding()
    torch.save(model.state_dict(), "/content/drive/MyDrive/Colab_Notebooks/region-embedding-benchmark/region-embedding/baselines/poi-encoder/data/SF-poi-encoder.tensor")
>>>>>>> Stashed changes


if __name__ == '__main__':
    # preprocess()
    poi2vec_train()
    main()