This tool is used for generating data and formatting data <br>

For generation tool, there are multiple methods: <br>
    1. Providing both loading directories for images and labels and saving directories for images and labels <br>
        - it will automatically proccess all the images and labels in the loading directory <br>
            using user provided pipeline and save in the saving directory <br>
    2. Providing only the loading diretory: <br>
        - it will act like a generator, proccess the images in the loading directory then output the numpy <br>
        array objects that represent the images (h * w * 3) and labels (h * w)<br>
    3. Providing only the saving directory, as well as input images and labels as numpy arrays (h * w):<br>
        - it will automatically save the input in the specified directory<br>
    4. Providing only the input images and labels numpy arrays:<br>
        - it will proccess the input and output the corresponding numpy array objects<br>

    For example usage, see data_generation.py <br>

For formatting the images and labels, <br>
we assume that each image is with format id_.png<br>
and each label is with format id_GT.png<br>

by providing an read directory and save directory, it will reformat the images and save them.<br>



TO INSTALL: <br>
    1. create a new conda environment<br>
    2. using the command to install the requirements: <br>
        conda install --yes --file requirements.txt <br>
