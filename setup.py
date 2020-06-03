import setuptools

setuptools.setup(
    name="tfhelper",
    version="0.0.8",
    license='MIT',
    author="Jongkuk Lim",
    author_email="lim.jeikei@gmail.com",
    description="This packages contains frequently used methods or classes in use of Tensorflow 2.x",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JeiKeiLim/tfhelper",
    packages=setuptools.find_packages(),
    classifiers=[
        # 패키지에 대한 태그
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    install_requires=[
        'tensorflow>=2.2.0',
        'matplotlib>=3.2.1',
        'seaborn>=0.10.1',
        'tensorflow_model_optimization>=0.3.0',
        'pandas>=1.0.3',
        'opencv-python>=4.2.0'
      ],
)