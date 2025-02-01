from setuptools import setup, find_packages

setup(
    name='Annaparavai',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'tensorflow>=2.0.0',
        'scikit-learn',
        'joblib',
        'keras',
    ],
    author=['Jubeerathan Thevakumar', 'Luheerathan Thevakumar'],
    author_email=['jubeerathan.20@cse.mrt.ac.lk', 'the.luheerathan@gmail.com'],
    description='A model to detect AI-generated and human-written text.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Jubeerathan/Annaparavai',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
