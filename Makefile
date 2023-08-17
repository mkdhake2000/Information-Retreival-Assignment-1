PYTHON = python3

# Defining an array variable
FILES = input output

# This target is executed whenever we just type `make`
.DEFAULT_GOAL = help

# The @ makes sure that the command itself isn't echoed in the terminal
help:
	@echo "---------------HELP-----------------"
	@echo "To run all the models type make run"
	@echo "------------------------------------"

# The ${} notation is specific to the make syntax and is very similar to bash's $()
run:
	@pip install num2words
	@unzip -o 21111405-ir-systems.zip
	@unzip -o 21111405-qrels.zip
	@rm -rf Output
	@mkdir Output
ifdef filename
	@${PYTHON} 21111405-ir-systems/run.py ${filename}
else
	@echo 'filename not given. Using default filename'
	@${PYTHON} 21111405-ir-systems/run.py  21111405-qrels/Queries
endif

