from setuptools import setup, find_packages

setup(name='discountproducts',
    version='0.1',
    packages=find_packages(),
    description='run missing imports on gcloud ml-engine',
    author='author',
    author_email='author@email.de',
    license='license',
    install_requires=[
        'pymongo'
    ],
    zip_safe=False)
