#   This tool is used for generating data and formatting data 

### For generation tool, there are multiple methods: 
        1. Providing both loading directories for images and labels and saving directories for images and labels
            - it will automatically proccess all the images and labels in the loading directory
                using user provided pipeline and save in the saving directory
                
        2. Providing only the loading diretory: 
            - it will act like a generator, proccess the images in the loading directory then output the numpy array objects that represent the images (h * w * 3) and labels (h * w)

        3. Providing only the saving directory, as well as input images and labels as numpy arrays (h * w):
            - it will automatically save the input in the specified directory
  
        4. Providing only the input images and labels numpy arrays:
            - it will proccess the input and output the corresponding numpy array objects

        For example usage, see data_generation.py 

### For formatting the images and labels
    we assume that each image is with format id_.png
    and each label is with format id_GT.png

    by providing an read directory and save directory, it will reformat the images and save them.

### The CrackDataset class
    it is as other pytorch.Dataset, with example usage



### TO INSTALL: 
    1. create a new conda environment
    2. install your pytorch pytorchvision 
    3. using the command to install the requirements: 
        conda install --yes --file requirements.txt
