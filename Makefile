.PHONY: install test train evaluate demo

install:
	python -m pip install -U pip
	python -m pip install -e ".[dev]"

test:
	python -m pytest

train:
	python -m robot_walk.train --timesteps 100000

evaluate:
	python -m robot_walk.evaluate --model models/ppo_biped.zip --episodes 5

demo:
	python -m robot_walk.demo --episodes 3
