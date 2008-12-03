products = mamarama_checkerboard.osg

default: $(products)

clean:
	rm -f $(products)

mamarama_checkerboard.osg: make_mamarama_checkerboard.py
	python make_mamarama_checkerboard.py

.PHONY: clean default