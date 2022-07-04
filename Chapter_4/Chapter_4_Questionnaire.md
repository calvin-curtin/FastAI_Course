# Chapter 4 Questionnaire 

1. How is a grayscale image represented on a computer? How about a color image?
    - On a computer everything is represented as a number. Images are therefore represented on a computer as a NumPy array or PyTorch tensor, with pixel values representing the content of an image.
    - In grayscale images, white pixels are stored as the number 0, black is the number 255, and shades of gray are between the two.
    - In colour images, there are three colour channels (RGB) which are typically used, with a separate 256-range 2D array used for each channel. A pixel value of 0 again represents white, with 255 representing solid red, green or blue. The three 2D arrays form a final 3D array (rank-3 tensor) representing the colour image.

2. How are the files and folders in the MNIST_SAMPLE dataset structured? Why?
    - The MNIST_SAMPLE dataset is structured into 'train' and 'valid' directories containing the training and validation data. Each directory has subdirectories '3' and '7' which contain the .jpg files for the respective class of images. For the full MNIST dataset, there are 10 subsubfolders, one for the images for each digit.
    - Datasets are split into training and validation sets to ensure the model can be evaluated against a metric and avoid overfitting.
    
3. Explain how the "pixel similarity" approach to classifying digits works.
    - The "Pixel Similarity" approach works by taking an average of all the pixel values of all 3s and 7s across the training set in order to generate a model of an "ideal" 3 and 7. We then compute the distance of an unseen 3 or 7 from their ideal versions to determine their similarity. If the distance of the unseen 3 is lower for the ideal 3 compared to the ideal 7, we can say that it is a 3. The vice versa is also true.

4. What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.
    - List comprehension is a short form method of creating a list using a *for* loop in Python.
        ``` in_list = range(10)
            out_list = [2*i for i in in_list if i%2 == 1]
        ```

5. What is a "rank-3 tensor"?
    - The rank of a tensor is the number of dimensions it has. An easy way to identify the rank is the number of indices you would need to reference a number within a tensor.
    - A rank-0 tensor is a scalar. A rank-1 tensor is a vector. A rank-2 tensor is a matrix. A rank-3 tensor is a stack of matrices.
    - The rank of a tensor is independent of its shape or dimensionality.

6. What is the difference between tensor rank and shape? How do you get the rank from the shape?
    - The rank is the number of dimensions of a tensor (e.g. [x, y, z] is rank 3)
    - The shape is the size of a tensor. It tells us the length of each axis. 
    - The length of a tensor's shape is its rank.

7. What are RMSE and L1 norm?
    - The L1 norm (Mean Absolute Difference) and L2 norm (Root Mean Squared Value) are two common methods used to measure distance. 
    - Simple differences are not effective because some differences can be positive while others are negative, which can cancel each other out. Therefore, a function that focuses on the magnitudes of differencs is needed to properly measure distance.

8. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?
    - Loops are very slow in Python, therefore it is best practice to represent the operations as array operations rather than looping through individual elements.
    - If this can be done, then using NumPy or PyTorch will be thousands of times faster, as they use underlying C code which is much faster than pure Python.

9. Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.
    ``` a_tensor = tensor(list(range(10)).view(3,3))
        b_tensor = 2*a_tensor
        b_tensor[1:,1:]
    ```

10. What is broadcasting?
    - When PyTorch tries to perform a simple operation between two tensors of different ranks, the tensor with the smaller rank is automatically expanded to have the same size as the larger rank.
    - This allows operations to be performed between tensors with a different rank.

11. Are metrics generally calculated using the training set, or the validation set? Why?
    - Metrics tell us how good our model is performing and are typically calculated using the validation set so that we do not inadverdantly overfit. 

12. What is SGD?
    - Stochastic Gradient Descent is an optimisation algorithm that will update the parameters of a model in order to minimise a given loss function that was evaluated on the predictions and target.

13. Why does SGD use mini-batches?
    - In order to take an optimisation step, we are required to calculate the loss over one or more data items.
    - We do not calculate on the entire dataset due to computation and time constraints. Calculating it for a single item would result in a very imprecise and unstable gradient.
    - We take a compromise and calculate the average loss for a few datas at a time, which is known as a mini batch. This is more computationally efficient than single items on a GPU.

14. What are the seven steps in SGD for machine learning?
    - Step 1: Initialise the parameters
    - Step 2: Calculate predictions
    - Step 3: Calculate the loss
    - Step 4: Calculate the gradients
    - Step 5: Step the weights
    - Step 6: Repeat the process
    - Step 7: Stop

15. How do we initialize the weights in a model?
    - We initialise the weights with random values.

16. What is "loss"?
    - The loss function represents how good our model is and will return a value based on given predictions and targets, where lower values correspond to better model predictions.

17. Why can't we always use a high learning rate?
    - We cannot use a high learning rate the the loss may not converge if the optimiser is taking steps that are too large.

18. What is a "gradient"?
    - The gradients tell us how much we have to change each weight to make our model model.
    - It is essentially a measure of how the loss function changes with changes of the weights of the model.

19. Do you need to know how to calculate gradients yourself?
    - No, we do not need to know how to calculate the gradients ourselves. PyTorch can automatically compute the derivative of nearly any function very quickly if we use `.requires_grad_()`.
    The gradients are then calculated when we call `.backward()`, which refers to backpropagation.

20. Why can't we use accuracy as a loss function?
    - A loss function needs to change as the weights are being adjusted.
    - Accuracy only changes if the prediction of a model change. So if there are slight changes to the model but does not change the prediction, the accuracy will not change. As a result, the gradients will be 0 almost everywhere and the model will not be able to learn.

21. Draw the sigmoid function. What is special about its shape?
    - The sigmoid function is a smooth curve that squishes all values between 0 and 1. As most loss functions assumes that the model is outputting some form of probability between 0 and 1, we use the sigmoid function at the end of a model to convert our values.

22. What is the difference between a loss function and a metric?
    - A metric is used to drive humand understanding.
    - A loss function is used to drive automated learning.

23. What is the function to calculate new weights using a learning rate?
    - The Optimiser Step.
    ``` w -= w.grad * lr
    ```
24. What does the DataLoader class do?
    - The DataLoader class can take any Python cllection and turn it into an iterator over mini-batches.

25. Write pseudocode showing the basic steps taken in each epoch for SGD.
    ``` for x,y in dl:
            pred = model(x)
            loss = loss_func(pred, y)
            loss.backward()
            parameters -= parameters.grad * lr
            parameters.grad = None
    ```

26. Create a function that, if passed two arguments [1,2,3,4] and 'abcd', returns [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]. What is special about that output data structure?
    ``` def func(a,b): return list(zip(a,b))
    ```

27. What does view do in PyTorch?
    - `view` is a PyTorch method that changes the shape of a tensor without changing the contests.

28. What are the "bias" parameters in a neural network? Why do we need them?
    - Remembering from linear algebra that the formula for a line is `y=w*x + b`, if the input is zero, the output will always be zero.
    - To add flexibility to our model, we introduce a bias.

29. What does the @ operator do in Python?
    - The @ operator in Python is the matrix multiplication operation.

30. What does the backward method do?
    - The `backward` method returns the gradients.

31. Why do we have to zero the gradients?
    - PyTorch will add the gradients of a variable to any previously stored gradients. If the training loop is called multiple times, without zeroing the gradients, the gradient of the current loss will be added to the previously stored gradient value.

32. What information do we have to pass to Learner?
    - We need to pass in:
        The DataLoaders
        The Model
        The Optimisation Function
        The Loss Function
        Optionally any metrics to print

33. Show Python or pseudocode for the basic steps of a training loop.
    ``` def train_epoch(model):
        for xb,yb in dl:
            calc_grad(xb, yb, model)
            opt.step()
            opt.zero_grad()
    ```
34. What is "ReLU"? Draw a plot of it for values from -2 to +2.
    - ReLU refers to the "rectified linear unit" and simply replaces every negative number with a zero. It is a common activation function.

35. What is an "activation function"?
    - An "activation function" is another function that is part of the neural netwhich which provides a non-linearity to the model. It is required because without an activation function, a series of linear layers is equivalent to a single linear layer with a different set of parapeterms. 
    - By introducing a non-linearity between the linear layers, each layer is somewhat decoupled and the model can fit more complex functions.

36. What's the difference between F.relu and nn.ReLU?
    - F.relu is a Python function for the relu activation function.
    - nn.ReLU is a PyTorch module.

37. The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?
    - With a deeper model, we do not need to use as many parameters as it turns out that we can use smaller matrices with more layers and get better results than we would with larger matrices and fewer layers. This means we can train models more quickly and use less memory.