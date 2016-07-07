try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

install_requires = [
    'matplotlib>=1.4.3',
    'scikit-learn>=0.17',
    'scipy>=0.16.1'
]

with open('README') as f:
    readme = f.read()

setup(
    name='ml-proj6',
    version='1.0.0',
    packages=['code'],
    description='CS189 Project 6',
    long_description=readme,
    author='Alex Francis',
    author_email='afrancis@berkeley.com',
    install_requires=install_requires,
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.4',
    ]
)
