# Chapter 5 Questionnaire 

1. Why do we first resize to a large size on the CPU, and then to a smaller size on the GPU?
    - Resizing is done per item on the CPU as each image may be differently sized from one another, so it cannot be parallelised. Once all the images are the same size however, data augmentation transformations can then be parallelised on the GPU.

2. If you are not familiar with regular expressions, find a regular expression tutorial, and some problem sets, and complete them. Have a look on the book's website for suggestions.

3. What are the two ways in which data is most commonly provided, for most deep learning datasets?
    - 1. Individual files representing items of data, such as text documents or images, possibly organised into folders or with filenames representing information about those items.
    - 2. A table of data where each row is an item which may include filenames providing a connection between the data in the table and data in other formats, such as text documents and images.

4. Look up the documentation for `L` and try using a few of the new methods that it adds.

5. Look up the documentation for the Python `pathlib` module and try using a few methods of the `Path` class.

6. Give two examples of ways that image transformations can degrade the quality of the data.
    - Rotation can leave empty areas in the final image.
    - Other operations may require interpolation which is based on the original image pixels but are still of lower image quality.

7. What method does fastai provide to view the data in a `DataLoaders`?
    - `.show_batch`

8. What method does fastai provide to help you debug a `DataBlock`?
    - `.summary`
9. Should you hold off on training a model until you have thoroughly cleaned your data?
    - No, you should train a baseline and use the model to assist in cleaning the data.

10. What are the two pieces that are combined into cross-entropy loss in PyTorch?
    - Cross Entropy Loss is a combination of Softmax Function and Negative Log Likelikhood Loss

11. What are the two properties of activations that softmax ensures? Why is this important?
    - It makes the outputs for the classes add up to one. This means the model can only predict one class. 
    - Additionally, it amplifies small changes in the output activations, which is helpful as it means the model will select a label with higher confidence (good for problems with definite labels)

12. When might you want your activations to not have these two properties?
    - When you have multi-label classification problems (more than one label possible).

13. Calculate the `exp` and `softmax` columns of <<bear_softmax>> yourself (i.e., in a spreadsheet, with a calculator, or in a notebook).

14. Why can't we use `torch.where` to create a loss function for datasets where our label can have more than two categories?
    - Because `torch.where` can only select between two possibilities, whereas for multi-class classification, we can have multiple possibilities.

15. What is the value of log(-2)? Why?
    - It is not defined for negative values.

16. What are two good rules of thumb for picking a learning rate from the learning rate finder?
    - 1. Pick one order of magnitude less than where the minimum loss was achieved (i.e. minimum divided by 10)
    - 2. Pick the last point where the loss was clearly descending.

17. What two steps does the `fine_tune` method do?
    - 1. Trains the randomly added layers for one epoch, with all other layers frozen.
    - 2. Unfreezes all of the layers and trains them all for the number of epochs requested.

18. In Jupyter Notebook, how do you get the source code for a method or function?
    - Use `??` after the function. (e.g. `learn.fine_tune??`

19. What are discriminative learning rates?
    - Discriminative learning rates refers to the training trick of using different learning rates for different layers of the model. Lower learning rates are used for the early layers of the neural network whilst higher learning rates are used for the later layers.

20. How is a Python `slice` object interpreted when passed as a learning rate to fastai?
    - The first value passed from a slice will be the learning rate in the earliest layers of the neural network, and the second value will be the learning rate in the final layers.
    - The layers in between will have learning rates that are multiplicatively equidistant throughout that range.

21. Why is early stopping a poor choice when using 1cycle training?
    - If early stopping is used, the training may not have time to reach lower learning rate values in the learning rate schedule, which could easily continue to improve the model. Therefore, it is recommended to retrain the model from scratch and select the number of epochs based on where the previous best results were found.

22. What is the difference between `resnet50` and `resnet101`?
    - The difference between them are the number of layers in the neural network. Resnet50 has 50 layers meanwhile Resnet101 has 101 layers.

23. What does `to_fp16` do?
    - `to_fp16()` enables mixed precision training which uses less precise numbers to speed up training on the GPU.