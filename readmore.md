Python, VS Code, and Required Libraries Setup

Follow these steps to set up Python, Visual Studio Code, and the required Python libraries.

1. Install Python

First, download and install Python on your computer.

During installation, make sure to enable “Add Python to PATH” before proceeding with the installation.

After installation, open a terminal and verify that Python is installed:

python --version

2. Install Visual Studio Code

Download and install Visual Studio Code (VS Code).

After installation, open VS Code and install the following extensions:

Python — Microsoft
Pylance — Microsoft

The Python extension provides Python development and execution support in VS Code, while Pylance provides advanced code analysis, autocomplete, type checking, and error detection.

3. Open Your Project Folder

Open your project folder in VS Code.

To verify that the terminal is currently inside the correct project folder, run:

pwd


This command displays the current working directory/path.

4. Install Required Python Libraries

Open a new terminal in VS Code and install the required libraries using:

python -m pip install matplotlib


Then install Pandas and NumPy:

python -m pip install pandas numpy


Alternatively, you can install all three libraries with a single command:

python -m pip install matplotlib pandas numpy

5. Verify the Installation

After the installation is complete, you can verify that the packages were installed successfully:

python -m pip show matplotlib pandas numpy


You are now ready to develop and run Python projects in VS Code with Matplotlib, Pandas, and NumPy installed.
