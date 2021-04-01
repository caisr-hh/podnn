import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="podnn", # Replace with your own username
    version="1.0",
    author="Peyman Mashhadi",
    author_email="peyman.mashhadi@hh.se",
    description="Parallel Orthogonal Deep Neural Networks",
    long_description='PODNN is a method lies in the intersection of deep learning and ensmeble methods. It makes efficient use of deep neural networks in an ensmble setting. It consists of a number of parallel deep neural networks that are made parallel together. Each parallel sub-layer is followed by an orthogonalization sub-layer. ',
    long_description_content_type="text/markdown",
    url="https://gitlab.com/peeymansh/podnn",
    project_urls={
        "Bug Tracker": "https://gitlab.com/peeymansh/podnn",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)