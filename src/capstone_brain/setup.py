from glob import glob
from setuptools import find_packages, setup
import os

package_name = "capstone_brain"
package_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(package_dir, "..", ".."))
model_source_dir = os.path.join(repo_root, "Vision", "turbopi_ncnn_model")
# Install the exported NCNN model alongside the Python package so the detector
# can load it from the ROS package share directory inside the container.
model_files = [
    os.path.relpath(path, package_dir)
    for path in glob(os.path.join(model_source_dir, "*"))
]

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            os.path.join("share", "ament_index", "resource_index", "packages"),
            [os.path.join("resource", package_name)],
        ),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "models", "turbopi_ncnn_model"), model_files),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="colin",
    maintainer_email="colin@example.com",
    description="Python ROS 2 nodes for the Capstone TurboPi brain container.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "detector_node = capstone_brain.detector_node:main",
            "tracking_node = capstone_brain.tracking_node:main",
            "fsm_node = capstone_brain.fsm_node:main",
        ],
    },
)
