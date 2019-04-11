"""
Author: Aleix Cambray Roma
Work done while at Imperial College London.

This code takes in the Flickr30k Entities dataset and generates the data structures necessary for the GroundeR implementation
Output data structures (for each image in dataset)
 - Annotation dictionary    annotation_data/<FILE_ID>.pkl           - A dictionary containing ids, bboxes and token_index_sequences for each phrase
 - Visual features matrix   visualfeatures_data/<FILE_ID>.npy       - A matrix in which each column is a visual feature vector for each region
 - Link id2idx dict         id2idx/<FILE_ID>.pkl                    - Matches object id for each phrase to the idx of the correct region (bbox)
"""

from utils import *
import PIL
from PIL import Image, ImageDraw
import time
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import os
import itertools
import glob
import pickle


def crop_and_resize(img, coords, size):
    """
    PIL image 'img' is first cropped to a rectangle according to 'coords'. Then the resulting crop is re-scaled according to 'size'
    :param img: PIL image
    :param coords: rectangle coordinates in the form of a list [x_nw, y_nw, x_se, y_se]
    :param size:  tuple or list [h_size, v_size]
    :return:
    """
    img = img.crop(coords)
    img = img.resize(size, PIL.Image.LANCZOS)
    return img


def unify_boxes(boxes):
    """
    This function turns a bunch of bounding boxes into one bounding box that encompasses all of them
    :param boxes: List of lists, each sub-list is a bounding box [x_nw, y_nw, x_se, y_se]
    :return: 1 bounding box
    """
    boxes = np.array(boxes)
    xmin = np.amin(boxes[:, 0])
    ymin = np.amin(boxes[:, 1])
    xmax = np.amax(boxes[:, 2])
    ymax = np.amax(boxes[:, 3])
    return [xmin, ymin, xmax, ymax]


def phrase2seq(phrase, word2idx, vocab):
    """
    This function turns a sequence of words into a sequence of indexed tokens according to a vocabulary and its word2idx mapping dictionary.
    :param phrase:      List of strings
    :param word2idx:    Dictionary
    :param vocab:       Set or list containing entire vocabulary as strings
    :return:            List of integers. Each integer being the token index of the word in the phrase according to the vocabulary.
    """
    # <start> and <end> tokens
    phrase = ['<start>'] + phrase
    phrase = phrase + ['<end>']

    phrase_seq = [0]*len(phrase)
    for i, word in enumerate(phrase):
        if word in vocab:
            phrase_seq[i] = word2idx[word]
        else:
            phrase_seq[i] = word2idx['<unk>']
    return phrase_seq


if __name__ == "__main__":
    build_vocab = True         # If True, the vocabulary will be built again and saved. If False, the last vocabulary will be loaded.
    generate_annotations = True
    generate_visual = False

    data_folder = "C:/Data/GroundeR/data/"

    train_txt = open(data_folder + 'train.txt', 'r')
    val_txt = open(data_folder + 'val.txt', 'r')
    test_txt = open(data_folder + 'test.txt', 'r')

    # Load RESNET
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    resnet = models.resnet101(pretrained=True)
    resnet.eval()
    resnet.cuda()

    ####################################################################
    # BUILD VOCABULARY                                                 #
    ####################################################################
    if build_vocab:
        embeddings_path = "C:/Data/Embeddings/" + "glove.6B/glove.6B.50d.txt"
        print("Loading Glove Model")

        f = open(embeddings_path, 'rb')
        vocabulary = ['<pad>', '<start>', '<end>', '<unk>']

        # Generate random vectors for start, end, pad and unk tokens
        word2vec = {}
        word2vec['<pad>'] = np.random.normal(loc=0.0, scale=1, size=50)
        word2vec['<start>'] = np.random.normal(loc=0.0, scale=1, size=50)
        word2vec['<end>'] = np.random.normal(loc=0.0, scale=1, size=50)
        word2vec['<unk>'] = np.random.normal(loc=0.0, scale=1, size=50)
        t0 = time.time()
        i = 0
        for line in f:
            splitLine = line.decode().split()
            word = splitLine[0]
            vocabulary.append(word)
            embedding = np.array([float(val) for val in splitLine[1:]])
            word2vec[word] = embedding
            if i % 100 == 0:
                print("\r{} - t={:0.10f}s".format(i, time.time() - t0), end="")
                t0 = time.time()
            i += 1

        word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
        idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

        idx2vec = {}
        for idx in idx2word.keys():
            idx2vec[idx] = word2vec[idx2word[idx]]

        weight_matrix = np.zeros((len(vocabulary), 50))
        for idx in range(weight_matrix.shape[0]):
            weight_matrix[idx, :] = idx2vec[idx]

        pickle.dump((vocabulary, word2idx, idx2word, weight_matrix), open("C:/Data/Embeddings/glove.6B/GloVe_6B_50d_vocabulary.pkl", 'wb'))

        print("\nDone.", len(vocabulary), " words loaded!")

    ####################################################################
    # BUILD ANNOTATION INPUT DATA STRUCTURE                            #
    ####################################################################
    if generate_annotations or generate_visual:
        with open("C:/Data/Embeddings/glove.6B/GloVe_6B_50d_vocabulary.pkl", "rb") as f:
            vocab, word2idx, idx2word, _ = pickle.load(f)
            vocab = set(vocab)

        # EXAMPLES LOOP
        phrase_count = 0
        img_ids = [f[:-4] for f in os.listdir(data_folder + 'annotations/Sentences') if f.endswith('.txt')]
        N = len(img_ids)
        t0 = time.time()
        for img_n, img_id in enumerate(img_ids):
            img_id = img_id.replace('\n', '')
            img_path = data_folder + 'flickr30k-images' + img_id + '. jpg'

            # Get Sentence and Annotation info for this image
            corefData = get_sentence_data(data_folder + 'annotations/Sentences/'+img_id+'.txt')
            annotationData = get_annotations(data_folder + 'annotations/Annotations/'+img_id+'.xml')

            ids = []
            bboxes = []
            seqs = []
            num_of_sentences = len(corefData)

            if generate_annotations:
                # DESCRIPTIONS LOOP
                for description in corefData:

                    # PHRASES LOOP
                    for phrase in description['phrases']:
                        # Get object ID
                        obj_id = phrase['phrase_id']

                        # Check if this phrase has a box assigned. If not, then skip phrase.
                        if obj_id not in list(annotationData['boxes'].keys()) or obj_id == '0':
                            continue
                        ids.append(obj_id)

                        # Obtain box coordinates for this phrase
                        boxes = annotationData['boxes'][obj_id]
                        box = unify_boxes(boxes) if len(boxes) > 1 else boxes[0]
                        bboxes.append(box)

                        # Turn phrase from sequence of strings into sequence of indexes
                        phrase = phrase['phrase'].lower().split()
                        phrase_seq = phrase2seq(phrase, word2idx, vocab)
                        seqs.append(phrase_seq)
                        phrase_count += 1

                image_annotations = {'bboxes': bboxes, 'ids': ids, 'seqs': seqs}
                with open('C:/Data/GroundeR/annotation_data/'+img_id+'.pkl', 'wb') as f:
                    pickle.dump(image_annotations, f, protocol=pickle.HIGHEST_PROTOCOL)
                if img_n % 100 == 0:
                    dt = time.time() - t0
                    print(f"\rImage {img_n}/{N} - dt_100 = {dt} - Phrases: {phrase_count}", end="")
                    t0 = time.time()

            if generate_visual:
                #############################################################
                # EXTRACT IMAGE FEATURES  (RESNET101)                       #
                #############################################################

                img = Image.open(data_folder + 'flickr30k-images/' + img_id + '.jpg')

                # Build id to idx dictionary
                id2idx = {id: idx for (idx, id) in enumerate(list(annotationData['boxes'].keys()))}

                # LOOP THROUGH ALL BOXES
                objects = list(annotationData['boxes'].keys())
                vis_matrix = np.zeros((len(objects), 1000), dtype=float)
                for id in objects:
                    # For each Object: extract boxes and unify them
                    boxes = annotationData['boxes'][id]
                    box = unify_boxes(boxes) if len(boxes) > 1 else boxes[0]
                    # For each box: crop original img to box, resize crop to 224z224, normalise image
                    box_img = crop_and_resize(img, box, (224, 224))
                    box_img = transform(box_img)
                    box_img = box_img.unsqueeze(0)
                    box_img = box_img.cuda()
                    # Feed image to ResNet-101 and add to visual feature matrix
                    vis_feature = resnet(box_img)
                    vis_matrix[id2idx[id], :] = vis_feature.cpu().detach().numpy()

                np.save('C:/Data/GroundeR/visualfeatures_data/'+img_id, vis_matrix)
                with open('C:/Data/GroundeR/id2idx_data/'+img_id+'.pkl', 'wb') as f:
                    pickle.dump(id2idx, f)
