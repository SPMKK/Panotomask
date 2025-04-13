from setuptools import setup, find_packages

setup(
    name='Panotomask',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch',
        'Pillow',
        'numpy',
        'scikit-learn',
        'ultralytics',
    ],
    package_data={
        '': ['model/best.pt'],  # nếu cần nhúng model
    },
)
