from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'jetracer'

# Helper for recursive data_files
def get_recursive_data_files(base_dir, share_subdir):
    data_files = []
    for root, dirs, files in os.walk(base_dir):
        if not files:
            continue
        rel_root = os.path.relpath(root, base_dir)
        dest_dir = os.path.join('share', package_name, share_subdir, rel_root)
        source_files = [os.path.join(root, f) for f in files]
        data_files.append((dest_dir, source_files))
    return data_files

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
] + get_recursive_data_files('models', 'models') + get_recursive_data_files('worlds', 'worlds')

setup(
    name=package_name,
    version='0.2.0',
    data_files=data_files,
    packages=find_packages(exclude=['test']),
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='developer',
    maintainer_email='developer@todo.com',
    description='JetRacer autonomous racing platform',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ros_gazebo_ex = jetracer.nodes.nodes.management_move:main',
            'make_gt = jetracer.nodes.nodes.make_gt:main',
        ],
    },
)
