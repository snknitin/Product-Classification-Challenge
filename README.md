# Product-Classification-Challenge

**Input Given :**  
* A catalog of 1000 fashion products in product_data.json.
* Each product has a description and a product image url.   


**Objective :**   

Classify the fashion products into 11(currently) categories.

     Dresses, Tops, Jeans, Skirts, Rompers, Shoes, Bags, Jewelry, Swimwear, Intimates, Others

## Thoughts

Here we have a multi-modal dataset with Image and Text information but no labels. This is an unsupervised categorization or classification into those 11 categories. In essense there are 11 possible assignmentsyou can give to a particular sample with an image and description.

It is possible to annotate few samples and then learn from those and convert this to a semi supervised learning problem. However we only have 1000 samples right now. Labelling a partition of it and then training on those to predict the rest won't be effective. Rather it would be faster to label them all and be more accurate. Even though the test size is only 1000, we need to think of a solution that can scale well. 

After exploring the data:
* Number of accessible product images : 967
* Number of product descriptions : 1000(Of which some are empty)

The missing values in each mode of data are not correspondent. The places where we lack text data, we have images and the instances where images cannot be accessed, the text data should suffice.

Essentially  Product_image + product_description == Product_attributes

We need to extract information from both the image and text data to represent each sample as a vector and try to cluster them.


### Text data :
Missing descriptions can be written as <UNK> with an unknown tag as a marker




## Conclusions

Answers to the following questions:
1) **Why are you designing the solution in this way?**  
2) **What are the aspects that you considered when designing?**    
3) **What are the cases your solution covers, how are they covered and why are they important?**    
4) **What are the cases your solution does not cover and what are the ways you can extend your current solution for them?**  



## Results

­ Classification results for all 1000 fashion products are added in the results folder
