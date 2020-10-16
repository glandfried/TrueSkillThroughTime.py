all: data synthetic longtest

synthetic: ./output/smoothing.csv
longtest: ./output/longtest_output.csv

./output/smoothing.csv:
	python3 synthetic.py
./output/longtest_output.csv:
	python3 longtest.py

data:
	make -C input




