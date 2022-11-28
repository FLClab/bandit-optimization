
from distutils.core import setup

setup(
	name="banditopt",
	version="0.1",
	author="Albert",
	install_requires=[
		"pyyaml",
		"statsmodels",
		"deap>=1.3",
		"scikit-learn",
		"numpy",
		"torch"
	],
	packages=["banditopt"],
)
