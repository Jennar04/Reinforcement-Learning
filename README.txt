FourRooms.py:
	
	Python code given which creates the fourRooms simulation and handles all of the methods associated with it that are needed throughout the scenarios - including graphing.


**all three scenarios are run in the following way (under deterministic actions)**
			python ScenarioX.py
	where 'X' represents the scenario number


Scenario1.py:
	
	Python code for agent to collect 1 package. Both Random and Epsilon-Greedy exploration techniques are in the code but the Random technique is commented out. 
	It is tested under deterministic by default, to make stochastic run:
			python Scenario1.py -stochastic
	

Scenario2.py:
	
	Python code for agent to collect 3 packages.
	It is tested under deterministic by default, to make stochastic run:
			python Scenario2.py -stochastic

Scenario3.py:
	
	Python code for agent to collect 3 packages in the order Red, Green then Blue.
	It is tested under deterministic by default, to make stochastic run:
			python Scenario3.py -stochastic


Images (folder):

	Folder containing the output images for each scenario. Each image is named under the following format:
	[ScenarioNumber]-[Random/Epsilon/Neither(blank)]-[Stochastic/Deterministic].png


venv (folder):

	A folder needed for running in a virtual environment.
	I am not sure if this is needed by the marker but it is required for the code to run on my machine. 


log.txt:

	Git repository commits in a text file.


Requirements.txt:

	Required packages to run the code.


Scenario1-Report.pdf:

	Report explaining the differences between the Random and Epsilon-Greedy Exploration Techniques for Scenario 1.

	