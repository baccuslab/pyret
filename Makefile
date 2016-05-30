all:
	pip install -r requirements.txt
	python setup.py install

develop:
	pip install -r requirements-dev.txt
	python setup.py develop

test:
	py.test -v --cov=pyret --cov-report=html tests

clean:
	rm -rf htmlcov/
	rm -rf pyret.egg-info
	rm -f pyret/*.pyc
	rm -rf pyret/__pycache__

build:
	python setup.py bdist_wininst

upload:
	twine upload dist/*
