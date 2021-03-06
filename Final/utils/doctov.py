import sys
import gensim
import load
import getopt

data_dir = "data"   # directory contains text documents
model_size = 200    # length of output vectors
epochs = 50    # number of training epochs
label_file = "label.txt"
out_filename = "doc2vec.out"

try:
    opts, args = getopt.getopt(sys.argv[1:],"hd:model_size:epoch:lb:",["data_dir=","model_size=","epoch=","label_file=","out_file="])
except getopt.GetoptError as e:
    print("Error of parameters")
    print(e)
    print(sys.argv[0] + " -h for help")
for opt, arg in opts:
    if opt == '-h':
        print ('Convert text documents to vectors by Doc2Vec')
        print (sys.argv[0] + " -h for help")
        sys.exit ()
    elif opt in ("-d","--data_dir"):
        data_dir = arg
    elif opt in ("-model_size","--model_size"):
        model_size = int (arg)
    elif opt in ("-epoch","--epoch"):
        nb_epochs = int (arg)
    elif opt in ("-lb","--label_file"):
        label_file = arg
    elif opt in ("--out_file"):
        out_filename = arg

#load documents
documents = load.get_doc(data_dir, label_file)

print('Data Loading finished')

print(len(documents),type(documents))

# build the model
# model = gensim.models.Doc2Vec(documents, dm=1, alpha=0.025, size=model_size, min_alpha=0.025, min_count=0, workers=8)
#
# # start training
# for epoch in range(epochs):
#     if epoch % 5 == 0:
#         print ('Now training epoch %s'%epoch)
#     model.train(documents)
#     model.alpha -= 0.002  # decrease the learning rate
#     model.min_alpha = model.alpha  # fix the learning rate, no decay

model = gensim.models.Doc2Vec(vector_size=model_size, window=10, min_count=0, workers=8, alpha=0.025, min_alpha=0.025, epochs=50)
model.build_vocab(documents)
model.train(documents, epochs=model.iter, total_examples=model.corpus_count)

# # shows the similar words
# print (model.most_similar('preserve'))

# # shows the learnt embedding
# print (model['preserve'])

# shows the similar docs with id = 2
print(model.docvecs.most_similar(str(2)))
