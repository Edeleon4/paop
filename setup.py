from setuptools import setup


setup(
        name='paop',
        version='1.0.0',
        description='A Python library for loading, training, and evaluating the' +
                    ' Pubmed Abstract Outcome Predictor',
        url='https://github.com/edeleon4/paop',
        author='Eduardo de Leon',
        author_email='edeleon4@mit.edu',
        license='MIT',
        packages=['paop'],
        scripts=['bin/paop'],
        test_suite='tests'
)
