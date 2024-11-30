[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Project Details

This project is part of Udacity Deep Reinforcement Learning Nanodegree 

![Trained Agent][image1]

The environment is a agent which learn to collect (yellow) banana. The environment is given by Unity's environment

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started
1.  If you want to run this project in Windows-64bit platform, skip and go to step 2.
    
    Else (if your platform is different from Windows-64bit)
        
        Download the environment from one of the links below.  You need only select the environment that matches your operating system:
            - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
            - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
            - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
            - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
            
            (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

            (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

        After download, place the file in the `p1_navigation/` folder, and unzip (or decompress) the file.
2.  
    Create an environment for running the project. 
        E.g: (with conda)

        conda create --name drlnd python=3.6
        conda activate drlnd
    Installing every packages in the "python" folder of this github repo (pip install )

        
        cd /your_path/UdacityDRL/python
        pip install .

3.  (Optional)
    If you want to run with the Jupyter Notebook file, create a kernel using the environment created before

        python -m ipykernel install --user --name drlnd --display-name "drlnd"
        

### Instructions
This part is aim to give instructions to run the project (for training and inferencing)

All of the configs are written in "config.py". Feel free to adjust, however, to produce the same result as mine, keep it the same. After modifying the config file, you are ready to go to next step
#### For Training
You can either train in 2 ways:
- Run the python file "train.py" with appropriate environment
- Run the notebook with appropriate kernel. The notebook has enough instructions

#### For Inferencing
Make sure the checkpoint file is available (if you did retrain the agent)

You can either inference in 2 ways:
- Run the python file "inference.py" with appropriate environment
- Run the notebook with appropriate kernel. The notebook has enough instructions
	