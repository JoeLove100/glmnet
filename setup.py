import os
from setuptools import setup


def get_version() -> str:

    current_file = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_file, "version")
    with open(version_file) as v_file:
        version = v_file.read()
    return version


setup(name="glmnet",
      version=get_version(),
      description="TensorFlow implementation of LocalGLMNet",
      keywords=["TensorFlow", "GLM", "deep learning", "machine learning"],
      author="Joseph Love",
      author_email="joelove100@gmail.com",
      license="BSD",
      data_files=["version", "CHANGELOG.md", "README.md"],
      py_modules=["local_glm_net"],
      install_requires=["numpy",
                        "pandas",
                        "scipy",
                        "matplotlib",
                        "tensorflow>=2.0.0"
                        ])
