# Binary-Neural-Networks
Implemented here a Binary Neural Network (BNN) achieving nearly state-of-art results but recorded a significant reduction in memory usage and total time taken during training the network.

![alt text](https://github.com/jaygshah/Binary-Neural-Networks/blob/master/bnn_jay.png)

Through this project, we attempt to train Binary Neural Networks(BNNs) which are essentially Neural Networks with binary weights and activations i.e. +1 and -1. BNNs boost the performance in terms of memory usage and computation complexity during the forward propagation. This is achieved by replacing most arithmetic operations with bitwise boolean operations. We compare the performance of a traditional non- binary network with a network designed using two binariza- tion functions: Deterministic and Stochastic. We also compare the train times, performance, memory usage, and speed for the binary and non-binary versions on the fashion-MNIST dataset. We achieve significant improvements with binarized neural networks.

# Results
We have implemented here a Binary Neural Network (BNN) that makes use of binarized weights and activations during the train time, as well as during the run-time. At the training time, these binarized values are used for computing gradients. This framework helps to train a network on machine with relatively fewer resources. We have used here two methodologies for binarizing the values of the network, deterministic and stochastic and compared both of them to two architectures of traditional non-binary implementation of a neural network on the Fashion-MNIST dataset. With both the binary versions we were not only able to achieve nearly state-of-art results but recorded a significant reduction in memory usage and total time taken during training the network. This was possible because the 32-bit arithmetic operation were replaced by bitwise operations.

In terms of total time taken for training, the deterministic binary version performed 28% and 42% better respectively for both the architectures. For the memory usage, the deter- ministic version performed 14% and 26% better respectively. We also compared the stochastic version of binary implemen- tation which again was able to reach state-of-art performance but also able to slightly improve the deterministic version’s memory usage and time taken for training. These results suggest that with minor trade-offs in performance, with binary neural networks, we can improve the memory and time consumption of training a neural network which is a key challenge in traditional neural networks.

## Contact
Please [reach out to me](https://www.public.asu.edu/~jgshah1/) in case of any concerns with running the codes 
