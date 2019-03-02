from setuptools import find_packages, setup

setup(
    name='crypr',
    version='0.0.1',
    packages=find_packages(),
    description='A prediction API for cryptocurrencies.',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='data_science cryptocurrency bitcoin ethereum deep_learning machine_learning prediction',
    author='Daniel C Stevenson',
    author_email='daniel.cortez.stevenson@gmail.com',
    license='MIT',
    install_requires=[
        'click>=0.7.0',
        'Flask>=1.0.2',
        'Keras>=2.2.2',
        'matplotlib>=2.2.2',
        'numpy<=1.14.5,>=1.13.3',
        'pandas>=0.23.1',
        'python-dotenv>=0.8.2',
        'PyWavelets==1.0.1',
        'requests>=2.19.1',
        'scikit-learn>=0.19.2',
        'scipy>=1.1.0',
        'seaborn>=0.9.0',
        'tensorboard>=1.12.0',
        'tensorflow>=1.8.0',
        'xgboost>=0.7.2',
    ],
    setup_requires=[
        'pytest-runner>=2.0,<3dev',
    ],
    tests_require=[
        'pytest>=4.0.2',
        'coverage>=4.5.2',
    ],
    entry_points = {
        'console_scripts': [
            'crypr-data=crypr.scripts.make_dataset:main',
            'crypr-features=crypr.scripts.make_features:main',
            'crypr-models=crypr.scripts.make_train_models:main',
        ],
    },
    include_package_data=True,
    zip_safe=False
)
