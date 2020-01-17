# self-supervised-curriculum

This repository hosts the code for preprint titled **[self-supervised visual feature learning with curriculum](https://arxiv.org/abs/2001.05634)**. Click on [:page_with_curl:](https://arxiv.org/pdf/2001.05634.pdf) to view the paper. Follow the description to run the code. Implementation is done in pytorch:

### Dependancies
* PyTorch (GPU capability will make the code execution faster) with torchvision
* numpy
* cv2
* matplotlib

### Run the code
1. Before running the scripts *jigsaw_experiments.py* and *self_supervised_curriculum.py*, change the *device* variable at the top of the script (depending on the availability of the CPU/GPU).
2. Run **jigsaw_experiments.py**. This is a executable file that executes main function. All the tunables are located in the helper functions.
3. Run **self_supervised_curriculum.py**. This is a executable file that executes main() function. All the tunables are located in the helper functions.
4. The scripts will output the results on the terminal. Pipe it to a file. In order to generate a predictable result, set a random seed or run each script 10 times, calculate the variations in the result.
5. Use **visualize_clusters.py** to generate the clusters.

### Attribution
This work was done as a part of COMPSCI 682 Neural Networks: A Modern Introduction course project.

### Contact
Mailing address: [vkeshav@cs.umass.edu](mailto:vkeshav@cs.umass.edu?subject=[self-supervised-curriculum]%20Github%20Code)
