from setuptools import setup, find_packages


setup(
    name='lam_usfft',
    version='0.1.0',
    author='Viktor Nikitin',
    package_dir={"": "src"},
    packages=find_packages('src'),
    zip_safe=False,
)
