from setuptools import find_packages, setup

setup(
    name='scMINGLE',
    version='1.1.0',
    description='MINGLE: A mutual information-based interpretable framework for automatic cell type annotation in single-cell chromatin accessibility data',
    long_description='MINGLE is an a mutual information-based interpretable framework that leverages similarities and topological structures among cells for accurate cell type annotation. Additionally, we introduce a convex hull-based identification strategy to effectively identify novel cell types.',
    long_description_content_type='text/markdown',
    author='Yifan Huang',
    python_requires=">=3.10.10",
    packages=find_packages(),
    data_files=[
        ('', ['MINGLE.png']),
    ],
    install_requires=[
        'scanpy>=1.9.1',
        'torch>=2.0.1',
        'torch_geometric>=2.5.2',
        'scikit-learn>=1.3.2',
        'scipy>=1.11.4',
    ],
)
