# Chapter 1 - Questionnaire

1. Do you need these for deep learning?
   - Lots of math F
   - Lots of data F
   - Lots of expensive computers F
   - A PhD F
   
2. Name five areas where deep learning is now the best in the world.
    - Natural Language Processing
    - Computer Vision
    - Recommendation Systems
    - Image Generation
    - Medicine

3. What was the name of the first device that was based on the principle of the artificial neuron?
    - Mark I Perceptron

4. Based on the book of the same name, what are the requirements for parallel distributed processing (PDP)?
    - The book, Parallel Distributed Processing (PDP) by David Rumelhart, James McClellan and the PDP Research Group specified the following requirements:
        i. A set of processing units
        ii. A state of activation
        iii. An output function for each unit
        iv. A pattern of connectivity among units
        v. A propagation rule for propaging patterns of activities through the network of connectivities
        vi. An activation rule for combining the inputs impinging on a unit with the current stat of that unit to produce an output for that unit
        vii. A learning rule whereby patterns of connectivity are modified by experience
        viii. An environment within which the system must operate

5. What were the two theoretical misunderstandings that held back the field of neural networks?
    - The two theoretical misunderstandings that held back the field of neural networks were:
        i. A single layer of artificial neurons cannot learn simple, critical mathematical functions like XOR logic gate. But multiple layers would allow these limitations to be addressed.
        ii. Adding just one extra layer of neurons is enough to allow any mathematical function to be approximated, but in practice such networks were often too book and too slow to be useful.

6. What is a GPU?
    - A Graphical Processing Unit is a special kind of processor in your computer that is designed for displaying 3D environments on a computer for playing games. It can handle thousands of single tasks at the same time which incidentally, are similar to what neural networks do. As such, a few GPUs can run neural networks hundreds of times faster than regular CPUs.

7. Open a notebook and execute a cell containing: `1+1`. What happens?
    - When we create and execute a cell containing `1+1`, the code is run by Python and the output is displayed underneath the code cell giving an answer of `2`.

8. Follow through each cell of the stripped version of the notebook for this chapter. Before executing each cell, guess what will happen.

9. Complete the Jupyter Notebook online appendix.

10. Why is it hard to use a traditional computer program to recognize images in a photo?
    - Humans can recognise images in a photo because our brains have subconciously learned what features define an object in the photo. To program a computer to do the same task would require us to write down for the computer the exact steps necessary to complete the task, which we don't really know as we're not concious of the entire process.

11. What did Samuel mean by "weight assignment"?
    - A "weight assignment" is a particular choice of value for a variable.

12. What term do we normally use in deep learning for what Samuel called "weights"?
    - What Samuel called "weights" are most generally referred to as model parameters. The term weights is reserved for a particular type of model parameter.

13. Draw a picture that summarizes Samuel's view of a machine learning model.
    Inputs
            ---> 
                    Model       --->        Results         --->        Performance
            --->                                                            |
    Weights                                                                 |
            <---------------------------------------------------------------                                                                    
14. Why is it hard to understand why a deep learning model makes a particular prediction?
    - It is hard to understand why a deep learning model makes a particular prediction due to their "deep" nature where a network may have hundreds of layers. It is hard to determine which factors are important in determining the final output due to the interaction of neurons.

15. What is the name of the theorem that shows that a neural network can solve any mathematical problem to any level of accuracy?
    - The Universal Approximation Theorem states that a neural network can theoretically represent any mathematical function.

16. What do you need in order to train a model?
    - To train a model, we need:
        i. An architecture for the given problem
        ii. Data to input to your model
        iii. Labels for your data to compare predictions to
        iv. A loss function that will quantitively measure the performance of your model
        v. A way to update the parameters of your model to improve its performance.
        
17. How could a feedback loop impact the rollout of a predictive policing model?
    - A predictive policing model my have a positive feedback loop where the model may predict arrests in specific areas which leads to increased policing and therefore further arrests.

18. Do we always have to use 224×224-pixel images with the cat recognition model?
    - No, we do not. We can use larger sizes and get better performance, but at the cost of speed and memory consumption.

19. What is the difference between classification and regression?
    - Classification is focused on predicting a discrete classes or categories.
    - Regression is focused on predicting a numeric quantity.

20. What is a validation set? What is a test set? Why do we need them?
    - The validation set is the portion of the dataset that is not used for training the model, but for evaluating the model during trianing, in order to prevent overfitting.
    - The test set is an unseen portion of the dataset that is used for final evaluation of the model.
    - The splitting of the dataset is necessary to ensure that the model generalises to unseen data. A test set is often required because we may adjust hyperparameters and training procedures according to the validation performance which can introduce bias.

21. What will fastai do if you don't provide a validation set?
    - fastaai will automatically create a validation dataset by randomly taknig 20% of the data and assigning it to the validation set.

22. Can we always use a random sample for a validation set? Why or why not?
    - No, we cannot always use a random sample for a validation set. A good validation or test set should be representative of the new data you will dea in the future. For example, for time series data, selecting randomly does not make sense.

23. What is overfitting? Provide an example.
    - Overfitting is when the model fits too closely to a limited set of data but does not generalise to unseen data. An example is where a neural network memorises the dataset it was trained on but will perform abysmally on unseen data.

24. What is a metric? How does it differ from "loss"?
    - A metric is a function that measures the quality of the model's predictions using the validation set.
    - It differs from loss in that it is meant to be human interpretable. Whereas loss is meanted for the optimization algorithm to efficicently update the model parameters.

25. How can pretrained models help?
    - Pretrained models have been trained on other problems that may be quite similar to the current task. They are useful because they have already learned how to handle many features that are similar to our current task, reducing the amount of time required for training.

26. What is the "head" of a model?
    - The "head" of a model are the later layers of a pretrained model which were useful for the task that the model was originalyl trained on. When fine-tuning, these layers are repalced with one or more new layers with randomised weights of an appropriate size for the dataset we are working with.

27. What kinds of features do the early layers of a CNN find? How about the later layers?
    - Earlier layers learn simple features like diagonal, horizontal and even vertical edges.
    - Later layers learn more advanced features like car wheels, flower petals and even outlines of animals.

28. Are image models only useful for photos?
    - No, image models are useful for other types of images. A lot of information can also be represented as images.

29. What is an "architecture"?
    - The architecture is a general template of how that kind of model works internally. It defines the mathematical model we are trying to fit.

30. What is segmentation?
    - Segmentation is a pixel wise classification problem. We attempt to predict a label for every single pixel in the image. This provides a mask for which parts of the image correspond to the given label.

31. What is `y_range` used for? When do we need it?
    - y_range is used to limit the values predicted when our problem is focused on predicting a numeric value in a given range.

32. What are "hyperparameters"?
    - Hyperparemters are parameters about parameters in the model training process.
    This includes learning rate, epochs etc.

33. What's the best way to avoid failures when using AI in an organization?
    - Ensure that a training, validation and test set are defined properly to evaluate the model in an appropriate manner.
    - Try out a simple baseline which future models should hopefully beat.