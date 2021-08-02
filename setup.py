from setuptools import setup

with open("version.txt") as v_file:
    version = v_file.read()

setup(name="glmnet",
      version=version,
      description="TensorFlow implementation of LocalGLMNet",
      keywords=["TensorFlow", "GLM", "deep learning", "machine learning"],
      author="Joseph Love",
      author_email="joelove100@gmail.com",
      license="BSD",
      py_modules=["local_glm_net"],
      install_requires=["numpy",
                        "pandas",
                        "scipy",
                        "matplotlib",
                        "tensorflow>=2.0.0"
                        ])
