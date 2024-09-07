model = model.h5
trainscript = train.py
test: $(model)
	python a.py

$(model): $(trainscript)
	python $(trainscript)
