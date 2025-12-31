<h1 align="center">Reinforcement Learning Methods</h1>

<h2>Description</h2>

<p>
To test my understanding of reinforcement learning,
I implemented all the core types of model based and model free reinfocement learning methods on a simple grid game.
The methods implemented include:
<ul>
  <li>Policy Iteration</li>
  <li>Value Iteration</li>
  <li>REINFORCE</li>
  <li>One-Step Actor-Critic</li>
  <li>Deep Q-Learning</li>
</ul>
</p>

<p>
This project also has a UI that was built using PySide6 with Qt Designer.
</p>

<h2>Usage:</h2>
<ol>
  <li>Activate a virtual environment.</li>
  <li>Run <code>pip install -r requirements.txt</code> to install the dependencies.</li>
  <li>Run <code>application.py</code> to train and test a model to play the grid game or play the game yourself.</li>
</ol>

<h2>Hyperparameters:</h2>

<p><strong>Hyperparameters found in <code>main.py</code>:</strong></p>
<ul>
  <li><code>EPISODES</code> (int): The number of episodes to train across.</li>
  <li>
    <code>HIDDEN_NEURONS</code> (tuple[int]): Defines the number of hidden neurons in each hidden layer. 
    The number of hidden layers is <code>len(HIDDEN_NEURONS) - 1</code>. 
    For example, <code>(128, 64, 32)</code> results in two hidden layers: the first with 128 input and 64 output neurons, 
    and the second with 64 input and 32 output neurons.
  </li>
  <li><code>REPLAY_CAPACITY</code> (int): The capacity of the replay buffer.</li>
  <li><code>BATCH_SIZE</code> (int): The number of experiences used in each training step.</li>
  <li><code>ALPHA</code> (float): The learning rate.</li>
  <li><code>GAMMA</code> (float): The discount factor.</li>
  <li><code>TRIAL_NAME</code> (str): The name of the current experiment, used as part of the filename for the TensorBoard logs.</li>
  <li>
    <code>MAIN_UPDATE_COUNT</code> (int): The number of training steps performed on the main network when an update condition is met.
  </li>
  <li><code>MAIN_UPDATE_FREQ</code> (int): The frequency (in episodes) at which the main network is updated.</li>
  <li><code>TARGET_UPDATE_FREQ</code> (int): The frequency (in episodes) at which the target network is updated from the main network.</li>
  <li>
    <code>MODEL_SAVE_NAME</code> (str): The name to save the trained model under. Leave as an empty string if the model should not be saved.
  </li>
</ul>

<p><strong>Hyperparameters found in <code>application.py</code>:</strong></p>
<ul>
  <li><code>MODEL_LOAD_NAME</code> (str): The name of the model to load and use for playing 2048.</li>
  <li>
    <code>MODEL_LOAD_HIDDEN_NEURONS</code> (tuple[int]): The hidden layer structure of the model being loaded. 
    Follows the same format as <code>HIDDEN_NEURONS</code> described above.
  </li>
</ul>

<h2>Controls:</h2>
<ul>
  <li><strong>Game Mode Radio Buttons:</strong>
      <br>
        Lets you select which model you would like to train and/or play the game, 
        when default is selected you can play the game using the arrow keys to move the player.
        Changing model creates a new instance or that model so any model training information will be lost.
      </br>
  </li>
  <li><strong>Parameters</strong>
    <br>Allows you to alter certain aspects of how the environments are generated and how the model trains and plays the game</br>
    <p>
      <br>The parameters do the following:</br>
      <strong>General:</strong>
      <ul>
        <li><strong>Size:</strong> Alters the size of the game environment i.e. a size of 4 gives a 4x4 environment</li>
        <li><strong>Player Position:</strong> x,y coordinates of the player in the grid that is the envionment</li>
        <li><strong>Food:</strong> Amount of food in the environment -1 gives a random amount of food</li>
        <li><strong>Nuke Probility:</strong> Probability of a Nuke tile being generated in valid positions in the environment</li>
        <li><strong>Intended Action Probability:</strong> Probability the action the player intends to take is the action actualy used to make the move</li>
        <li><strong>Move Time</strong> Time taken in seconds between each move the agent makes when playing the game</li>
        <li><strong>Gamma</strong> Discount factor of the expected cumulative reward of the next state</li>
      </ul>
      <strong>Model Based:</strong> (specific to model based methods)
      <ul>
        <li><strong>Epsilon:</strong> Convergence threshold used for automatic stopping. Once the change in values between two iteration falls below epsilon automatic stopping is triggered.</li>
      </ul>
      <ul>
        <li><strong>General:</strong> Toggle ON of you want the model to train on all possable variations of the environment for the parameters give or OFF if you want it to train only for the environment currently displayed </li>
        <li><strong>Alpha:</strong> Learning rate (it is also the learning rate for the actor network for the actor critic model)</li>
        <li><strong>Win Threshold:</strong> Convergence criteria for automatic stopping. The model will stop training once the average amount of wins in the last 100 episodes reaches this value.</li>
      </ul>
    </p>
    <ul>
      <li><strong>S Key:</strong> Starts or stops the agent autoplaying 2048.</li>
    </ul>
  </li>
  <li><strong>Space Bar:</strong> Resets the game (Disabled while the agent is autoplaying).</li>
</ul>


  
<h2>Results</h2>
<h3>Baseline</h3>

![image](https://github.com/user-attachments/assets/849c20c3-d3c3-4754-8a0f-6974410169d9)
<p>
These baseline results are when the agent played using a random policy. This image can originally be found in the Experiment Notebook.
</p>
<h3>Final Results</h3>

![image](https://github.com/user-attachments/assets/9ce39a63-9046-4e6b-b6b4-41f107b993a4)

<p>
After a lot of testing I trained my model for 30,000 episodes which took about 4 days.
These results show an amazing improvement from the baseline with the agent even reaching a score of 2048 occasionally, I am sure that with more training an agent would be able to consistently reach a value of 2048. The original image of the results can be found in the Inference Notebook.
</p>

<p>
Screenshot of the final UI:
</p>

![image](https://github.com/user-attachments/assets/8cf9e56e-9c4b-4616-9c26-cb9ee1429ef6)








