all:
	echo "https://packaging.python.org/en/latest/tutorials/packaging-projects/"

upgrade:
	python3 -m pip install --upgrade build
	twine upload dist/*
