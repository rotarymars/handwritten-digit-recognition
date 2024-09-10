model = model.h5
trainscript = train.py
data = data/
test: $(model)
	python a.py

$(model): $(trainscript) $(data)
	python $(trainscript)
