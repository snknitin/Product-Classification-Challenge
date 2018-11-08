# Product-Classification-Challenge

**Input Given :**  
* A catalog of 1000 fashion products in product_data.json.
* Each product has a description and a product image url.   


**Objective :**   

Classify the fashion products into 11(currently) categories.

     Dresses, Tops, Jeans, Skirts, Rompers, Shoes, Bags, Jewelry, Swimwear, Intimates, Others

## Thoughts

Here we have a multi-modal dataset with Image and Text information but no labels. This is an unsupervised categorization or classification into those 11 categories. In essense there are 11 possible assignments you can give to a particular sample with an image and description. We can think of the data as having 11 discernable clusters for now.

It is possible to annotate few samples and then learn from those to convert this to a semi-supervised learning approach. However we only have 1000 samples right now. Labelling a partition of it and then training on those to predict the rest won't be effective. Rather it would be faster to label them all and be more accurate. Even though the test size is only 1000, we need to think of a solution that can scale well. 

After exploring the data:
* Number of accessible product images : 967
* Number of product descriptions : 1000(Of which some are empty)

The missing values in each mode of data are not correspondent. The places where we lack text data, we have images and the instances where images cannot be accessed, the text data should suffice.

Essentially,
          
          Product_image + product_description == Product_attributes

We need to extract information from both the image and text data to represent each sample as a vector and try to cluster them, or group them based on a similarity metric. We want to identify pairs or sets of similar products. Without labels, this is either a clustering or a tagging problem and with labels , this is a classification problem.

1) Extract features from the multi-modal data and create a representation of useful attributes
2) Calculate similarity among those feature vectors - Cosine or jaccard similarity/LSH



### Possible Problems 

Product data can be diverse and unbalanced. Not all descriptions may contain the brand name, the size, physical features, price tag, gender info, age groups, product related jargon etc

There can be a hierarchy in the classes. One can be a hypernym of the other or a very close match. Eg: Swimwear and Intimate mgith have similar descriptions

The ethnicity/complexion of the models wearing the clothes might not be stratified and that image data might misclassify few results. We could segment the items in the image from the person wearing the clothes or accessories.



### Text data :

Missing descriptions can be written as <UNK> with an unknown tag as a marker.    
Around 400 descriptions start with "shop the women's", which may be indicative of certain images or products or just noise.     
Some unnecessry tags can be manually removed, the rest of the html tags are removed by BeautifulSoup parser.  
Tokenization and Stemming, along with removing stopwords or punctuations.
Certain descriptions are in a different language like French and German,so it would not be useful to pull pretrained embeddings.


Possible ideas:
1) BOW - can create an n-dimensional vector for unique words but might be sparse.
2) Word2Vec - can use gensim to create it but the data is too small to capture any distributional hypothesis for context words. Doubtful of its efficency
3) TFIDF - might require less processing. Might be overkill for this usecase but likely to yield best results sicne we stick to the lexicon of this dataset and the impact of non-informative words will be lowered by the IDF and the dimension of data can be reduced. 

**Final Text features :** Word2vec embeddings of the words in the sentence averaged by using tfidf scores as weights. Dimension size = 300

### Image Data

Missing data is substituted with a blank(image of zeros in pixel values)   
All images are resized to (200,200)  and I made sure that the channels are 3 and not 4 like in some cases  
I can use a CNN(pretrained/using Image-Net/VGG) to combine low-level features (lines, edges, colors) to more and more abstract features (squares, circles, objects, faces)  and then pass it through a feed forward netwrok to reduce dimensions  
The extracted local features must be:  
* Repeatable and precise so they can be extracted from different images showing the same object.  
* Distinctive to the image, so images with different structure will not have them.  

**Final Text features :** Flattened vectors from the VGG19 imagenet architecture, passed into feed forward layers using dropout and relu activations. Final dimension size = 256  

A 100X100 grid of imagesin the data

![alt text](https://github.com/snknitin/Product-Classification-Challenge/blob/master/static/catalog/products_0.PNG)
![alt text](https://github.com/snknitin/Product-Classification-Challenge/blob/master/static/catalog/products_100.PNG)
![alt text](https://github.com/snknitin/Product-Classification-Challenge/blob/master/static/catalog/products_200.PNG)
![alt text](https://github.com/snknitin/Product-Classification-Challenge/blob/master/static/catalog/products_300.PNG)
![alt text](https://github.com/snknitin/Product-Classification-Challenge/blob/master/static/catalog/products_400.PNG)
![alt text](https://github.com/snknitin/Product-Classification-Challenge/blob/master/static/catalog/products_500.PNG)
![alt text](https://github.com/snknitin/Product-Classification-Challenge/blob/master/static/catalog/products_600.PNG)
![alt text](https://github.com/snknitin/Product-Classification-Challenge/blob/master/static/catalog/products_700.PNG)
![alt text](https://github.com/snknitin/Product-Classification-Challenge/blob/master/static/catalog/products_800.PNG)
![alt text](https://github.com/snknitin/Product-Classification-Challenge/blob/master/static/catalog/products_900.PNG)


### Intuition



* Dresses - Either worn by model or displayed as stand-alone item or a group of them.
* Tops -  Either worn by model or displayed as stand-alone item or a group of them.
* Jeans  - Most of the images are focused on a ppair of legs(waist down) or a model sitting while wearing them(Can be misclassified). 
* Skirts - Either worn by model or displayed as stand-alone item or a group of them.
* Rompers - Either worn by model or displayed as stand-alone item or a group of them.
* Shoes - Almost all images would be stand alone pictures of shoes or a foot wearing one. Might be ambiguous if the whole picture of a model is present. Text description will be the deciding factor.
* Bags - 
* Jewelry -  Mostly empty(white) or similar colored background with a circle/wavy line or a triangular shaped closed object that can be identified easily if the image is turned from RGB to greyscale or we pass in a filter to identify changes in pixels. These should be the easiest to group without many misclassifications or any ambiguity.
* Swimwear -  This and intimates might be a difficult to categorize since the images will mostly be similar
* Intimates - 
* Others - This might be reserved for the outliers or sparse clusters that don't quite fit in with the rest. these include images of random objects like shades, bottles, make up items etc.,


## Conclusions

Answers to the following questions:
1) **Why are you designing the solution in this way?**  
2) **What are the aspects that you considered when designing?**    
3) **What are the cases your solution covers, how are they covered and why are they important?**



4) **What are the cases your solution does not cover and what are the ways you can extend your current solution for them?**  

Confusion Driven Probabilistic Fusion++ (CDPF++), that is cognizant of the disparity in the discriminative power of different types of signals and hence makes use of the confusion matrix of dominant signal (text in our setting) to prudently leverage the weaker signal (image), for an improved performance

Relying exclusively on text, we might suffer from the following problems 
 
* **Overlapping text across fine-grained hierarchical categories :** Some perfectly valid textual descriptions for two completely different products (an Iphone and a Phone cover) might differ in just one word: “Iphone 7 with Case” and “Iphone 7 Case”
* **Short descriptions from merchants** : Some of the text might just be a model number of a particular item
* **Vocabulary mis-match** : Laptop and notebook might be interchageable 


With images, we might have the following problems:

* Few products from the same category (say ‘Tops’) can have completely different images of different types of clothes of different colors. The association
between categories and images can be noisy if the category is broad.
* Instead of using the image signal (the weaker signal compared to text) to learn the entire discriminative surface over a large number of categories, it should be selectively used to learn a decision surface that discriminates a much fewer number of categories. It might be computationally inefficient to learn multiple categories 


## Results

* PCA and T-SNE on 2D 

     ![alt text](https://github.com/snknitin/Product-Classification-Challenge/blob/master/static/plots/PCA2dcomb.PNG)

* PCA and T-SNE on 3D 

     ![alt text](https://github.com/snknitin/Product-Classification-Challenge/blob/master/static/plots/PCA3dcombine.PNG)
     ![alt text](https://github.com/snknitin/Product-Classification-Challenge/blob/master/static/plots/T-SNE3dcombine.PNG)

Classification results for all 1000 fashion products are added in the Data folder
