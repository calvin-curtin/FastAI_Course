# Chapter 2 - Questionnaire

1. Provide an example of where the bear classification model might work poorly in production, due to structural or style differences in the training data.
    - The bear classification model might work poorly in production if we were dealing with video data instead of static image data.

2. Where do text models currently have a major deficiency?
    - Deep learning is currently not good at generating correct responses. i.e. We don't have a reliable way to combine a knowledge base of medical information with a deep learning model for generating medically correct natural language responses.

3. What are possible negative societal implications of text generation models?
    - Text generation models can easily create content that appears to a layman to be compelling, but actually is entirely correct leading to the proliferation of disinformation.

4. In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?
    - It may be beneficial to run the model in parallel to human processes and have humans check all predictions of the model.

5. What kind of tabular data is deep learning particularly good at?
    - Deep learning is useful for tabular data which contains a variety of columns and high-cardinality categorical columns. If you already have a system that is using random forests or gradient boosting machines, then switching to or adding deep learning may not result in any dramatic improvement.

6. What's a key downside of directly using a deep learning model for recommendation systems?
    - Machine learning approachs have the downside that they can only tell you what products a particular user might like, rather than what recommendations would be helpful for a user.

7. What are the steps of the Drivetrain Approach?
    i. Define a clear objective
    ii. Consider what levers you can put (i.e. What actions you can take)
    iii. Consider what new data would be required to be collected
    iv. Build the predictive model

8. How do the steps of the Drivetrain Approach map to a recommendation system?
    - The objective of a recommendation engine is to drive additional sales by surprising and delighting the customer with recommendations of items they would not have purchased without the recommendation.
    - The lever is the ranking of the recommendations.
    - New data must be collected to generate recommenations that will cause new sales.

9. Create an image recognition model using data you curate, and deploy it on the web.

10. What is DataLoaders?
    - DataLoaders is a fastai class that stores multiple DataLoader objects and passes them to a fastai model.

11. What four things do we need to tell fastai to create DataLoaders?
    - What kinds of data we are working with
    - How to get the list of items
    - How to label these items
    - How to create the validation set

12. What does the splitter parameter to DataBlock do?
    - The splitter parameter provides a way for fastai to split up your dataset into subsets (usually train and validation sets)

13. How do we ensure a random split always gives the same validation set?
    - To ensure a random split always gives the same validation set, we provide a random seed value as computers are unable to generate truly random numbers but instead use a process known as psuedo-random generator.

14. What letters are often used to signify the independent and dependent variables?
    - X = independent variable
    - Y = dependent variable

15. What's the difference between the crop, pad, and squish resize approaches? When might you choose one over the others?
    - Crop resizes the image to fit a square shape of the size requested which may end up cropping out a key part of an image.
    - Pad results in padding the matrix of the image's pixels with zeros which can result in a whole lot of empty space. Wasting computer for our model and resulting in a lower effective resolution for the part of the image we actually use.
    - Squish can either squsih or stretch the image, which can cause the image to take on an unrealistic shape.

    Which resizing method to use depends on the underlying problem and the dataset. Another better method is RandomResizedCrop in which we crop on a randomly selected region of the image. So every epoch, the model will see a different part of the image and learn accordingly.

16. What is data augmentation? Why is it needed?
    - Data augmentation refers to creating random variations of our input data, such that they appear different, but do not actually change the meaning of the data. It allows machine learning models to generalise which is especially important when it can be slow and expensive to label data.

17. What is the difference between item_tfms and batch_tfms?
    - item_tfms are transformations that run on each individual item on the CPU.
    - batch_tfms are transformations that are applied to batched data samples on the GPU.

18. What is a confusion matrix?
    - A Class Confusion Matrix is a representation of the predictions made vs the actual labels.
    The rows of the matrix represent the actual labels while the columns represent the predictions.
    The values in the diagonal elements represent the number of correct classifications while the off-diagonal elements represent incorrect classifications.

19. What does export save?
    - export saves both the architecture, as well as the trained parameters of the neural network. It also saves how the DataLoaders are defined.

20. What is it called when we use a model for getting predictions, instead of training?
    - Inference

21. What are IPython widgets?
    - IPython widgets are GUI components that bring together Javascript and Python functionality in a web browser, and can be created and used within a Jupyter notebook.

22. When might you want to use CPU for deployment? When might GPU be better?
    - GPUs are best for doing identical work in parallel. If you will be analyzing single pieces of data at a time (like a single image or single sentence), then CPUs may be more cost effective instead, especially with more market competition for CPU servers versus GPU servers.

23. What are the downsides of deploying your app to a server, instead of to a client (or edge) device such as a phone or PC?
    - When deploying your app to a server, you will require a network connection which can introduce extra network latency time when submitting and returning results. Furthermore, sending private data to a network server can lead to security concerns.

24. What are three examples of problems that could occur when rolling out a bear warning system in practice?
    i. Handling night-time images
    ii. Dealing with low-resolution images (ex: some smartphone images)
    iii. The model returns prediction too slowly to be useful

25. What is "out-of-domain data"?
    - "Out-of-domain" data is data that is fundamentally different to what our model saw during training.

26. What is "domain shift"?
    - "Domain shift" is where the type of data that our model sees changes over time.

27. What are the three steps in the deployment process?
    i. Manual Process - Model is run in parallel and not directly driving any actions, with humans checking all predictions.
    ii. Limited Scope Deployment - The model's scope is limited and carefully supervised.
    iii. Gradual Expansion - Model scope is gradually increased, while good reporting systems are implemented to check for any significant changes to the actions taken compared to manual process.