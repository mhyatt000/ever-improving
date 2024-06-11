from setuptools import find_packages, setup

setup(
    name="improve",
    version="0.1.0",
    url="https://github.com/mhyatt000/ever-improving",
    author="Matthew Hyatt",
    author_email="mhyatt000@gmail.com",
    description="A collection of tools for RL self-improvement",
    packages=find_packages(),
    install_requires=[
        "simpler_env",
    ],
    # entry_points={
    # 'console_scripts': [
    # 'your-command=yourpackage:main_function',
    # ],
    # },
)
