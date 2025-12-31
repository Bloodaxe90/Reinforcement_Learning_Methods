<h1 align="center">Reinforcement Learning Methods</h1>

<h2>Description</h2>

<p>
To test my understanding of reinforcement learning, I implemented core types of model-based and model-free reinforcement learning methods from scratch using PyTorch in a simple grid environment.
The methods implemented include:
</p>

<ul>
  <li>Policy Iteration</li>
  <li>Value Iteration</li>
  <li>REINFORCE</li>
  <li>One-Step Actor-Critic</li>
  <li>Deep Q-Learning</li>
</ul>

<p>
This project also features a UI built using PySide6 and Qt Designer.
</p>

<h2>Usage</h2>
<ol>
  <li>Create and activate a virtual environment.</li>
  <li>Run <code>pip install -r requirements.txt</code> to install the dependencies.</li>
  <li>Run <code>python application.py</code> to train an agent or play the game manually from the UI.</li>
</ol>

<h2>Game</h2>
<p>
The game takes place in a grid world environment. The player moves in the four cardinal directions within fixed boundaries (no wrapping). The objective is to collect all food items while avoiding nukes; touching a nuke results in a loss. In the UI (example below), the Player is the blue tile, Food is the green tile, and Nukes are red tiles. The player tile turns cyan when collecting food and magenta when hitting a nuke.
</p>

<h2>Controls</h2>
<h3><strong>Game Mode Radio Buttons:</strong>  </h3>
  Allows you to select which model you would like to train or use to play the game.
  When "Default" is selected, you can play the game manually using the arrow keys.
  Changing the model creates a new instance of that model, meaning any training progress will be lost.
  
<h3><strong>Parameters:</strong></h3>
Found in the sidebar of the UI (example below) and allows you to alter how environments are generated and how the model trains and plays.

They do the following:
<p><strong>General:</strong></p>
<ul>
  <li><strong>Size:</strong> Alters the size of the game environment (i.e. a size of 4 gives a 4x4 grid).</li>
  <li><strong>Player Position:</strong> The x,y coordinates of the player on the grid, -1,-1 give a random position.</li>
  <li><strong>Food:</strong> Amount of food in the environment, -1 results in a random amount.</li>
  <li><strong>Nuke Probability:</strong> Probability of a Nuke tile being generated in valid positions.</li>
  <li><strong>Intended Action Probability:</strong> Probability that the action the agent intends to take is the action actually executed.</li>
  <li><strong>Move Time:</strong> Time taken in seconds between each move the agent makes when playing (does not require the Update key (U) to be pressed to apply).</li>
  <li><strong>Gamma:</strong> Discount factor for the expected cumulative reward of the next state.</li>
</ul>
<p><strong>Model-Based:</strong> (specific to model-based methods)</p>
<ul>
  <li><strong>Epsilon:</strong> Convergence threshold used for automatic stopping. Stopping is triggered once the change in values between two iterations falls below epsilon.</li>
</ul>
<p><strong>Model-Free:</strong> (specific to model-free methods)</p>
<ul>
  <li><strong>General:</strong> Toggle ON if you want the model to train on all possible variations of the environment for the given parameters or OFF if you want it to train only on the specific environment currently displayed.</li>
  <li><strong>Alpha:</strong> Learning rate (used as the learning rate for the actor network in 1-Step A2C).</li>
  <li><strong>Win Threshold:</strong> Convergence criteria for automatic stopping. The model will stop training once the average win rate over the last 100 episodes reaches this value.</li>
</ul>
<p><strong>Deep Q-Learning:</strong> (specific to DQL)</p>
<ul>
  <li><strong>Batch Size:</strong> Training batch size.</li>
  <li><strong>Buffer Capacity:</strong> Size of the replay buffer.</li>
  <li><strong>Main Update Frequency:</strong> Number of episodes between each update of the main network.</li>
  <li><strong>Target Update Frequency:</strong> Number of episodes between each update of the target network.</li>
  <li><strong>Minimum Epsilon:</strong> Minimum probability of exploration.</li>
  <li><strong>Maximum Epsilon:</strong> Maximum probability of exploration.</li>
</ul>
<p><strong>One-Step Actor-Critic:</strong> (specific to 1-Step A2C)</p>
<ul>
  <li><strong>Critic Alpha:</strong> Learning rate of the critic network.</li>
</ul>

<h3><strong>Keyboard:</strong></h3>
<ul>
  <li><strong>U Key:</strong> Creates a new instance of the current model (meaning any training progress will be lost) and applies any updates made to the parameters.</li>
  <li><strong>T Key:</strong> Trains the currently selected model (no effect if "Default" game mode is selected).</li>
  <li><strong>P Key:</strong> Plays the game using the currently selected model (no effect if "Default" game mode is selected).</li>
  <li><strong>R Key:</strong> Resets the environment state.</li>
  <li><strong>Space Key:</strong> Generates a new environment layout.</li>
</ul>


<h2>Results</h2>
<p>
This project was a success. All model-based methods were able to learn successfully when the state space was small enough and with correctly tuned reward functions, the model-free methods successfully learned to play both single fixed environments and generalise to randomly generated environments.
</p>

<p>
This project also highlighted specific limitations or certaion methods. In more complex environments REINFORCE started falling victim to the credit assignment problem and failing to converge due to high variance. And Actor-Critic and Reinforce required different reward functions to Deep Q-Learning highlighting the main difference between them, ON-Policy and Off-Policy.
</p>

<p>
The GIF below demonstrates the One-Step Actor-Critic model playing the game after training for 30,000 episodes on randomised environments generated with the parameters shown in the sidebar.
</p>

![ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/7a5a7250-1221-48e9-90eb-1c4802b2d3d0)
