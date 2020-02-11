# S4
## Strategy
* Total number of parameters is 18,058. 
* final validation/test accuracy 99.44
* To improve the accuracy, I added the Batch normalization and drop out after every convolution layer but not in the last layer.
* The drop out rate is set to 15% and reduced it 10% later in the layers. The dropout is used to regulralize the neural networks and prevents the overfitting.
* I experimented the network in tow ways. First i used 2 Max Pooling layer and then i used only one Max pooling layer.
* When i used two max pooling layers the acuracy 99.44, when i used only layer then the acuracy is 99.42. 
* I uloaded both files. 
* After checking several models, accuracy is increased to 99.44.

## Team
K Bhargava Kiran 
bhargav.kiran@gmail.com
M V Subbarao
subbu.521@gmail.com

## Log
  0%|          | 0/469 [00:00<?, ?it/s]
1
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:42: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
loss=0.12111782282590866 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 40.87it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0882, Accuracy: 9769/10000 (97.69%)

2
loss=0.24784576892852783 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 40.62it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0617, Accuracy: 9808/10000 (98.08%)

3
loss=0.0398121140897274 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 41.74it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0372, Accuracy: 9901/10000 (99.01%)

4
loss=0.03255577012896538 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 40.77it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0313, Accuracy: 9915/10000 (99.15%)

5
loss=0.04599645361304283 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 41.10it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0379, Accuracy: 9892/10000 (98.92%)

6
loss=0.010331481695175171 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 41.36it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0272, Accuracy: 9902/10000 (99.02%)

7
loss=0.04799628257751465 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 40.20it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0280, Accuracy: 9918/10000 (99.18%)

8
loss=0.026990102604031563 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 41.23it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0271, Accuracy: 9922/10000 (99.22%)

9
loss=0.02089104987680912 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 41.90it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0242, Accuracy: 9929/10000 (99.29%)

10
loss=0.008404572494328022 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 32.48it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0233, Accuracy: 9919/10000 (99.19%)

11
loss=0.03724032640457153 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 41.05it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0231, Accuracy: 9926/10000 (99.26%)

12
loss=0.01273116935044527 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 42.17it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0219, Accuracy: 9933/10000 (99.33%)

13
loss=0.027635658159852028 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 40.16it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0218, Accuracy: 9936/10000 (99.36%)

14
loss=0.07462749630212784 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 40.39it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0189, Accuracy: 9938/10000 (99.38%)

15
loss=0.0033007909078150988 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 44.93it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0191, Accuracy: 9940/10000 (99.40%)

16
loss=0.019290516152977943 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 41.17it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0181, Accuracy: 9943/10000 (99.43%)

17
loss=0.018512168899178505 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 44.49it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0222, Accuracy: 9927/10000 (99.27%)

18
loss=0.004880592226982117 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 40.80it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0202, Accuracy: 9941/10000 (99.41%)

19
loss=0.0029897342901676893 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 40.84it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0180, Accuracy: 9944/10000 (99.44%)

20
loss=0.002100229263305664 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 44.49it/s]
Test set: Average loss: 0.0187, Accuracy: 9930/10000 (99.30%)


