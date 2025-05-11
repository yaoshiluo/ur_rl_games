from setuptools import setup
import os

package_name = 'ur_policy_runner'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        # 安装 config.yaml 和模型
        ('share/' + package_name + '/policy', ['policy/config.yaml']),
        ('share/' + package_name + '/policy/nn', ['policy/nn/industreal_policy_insert_pegs.pth']),
        ('share/' + package_name + '/policy', ['policy/config.yaml','policy/PegInsertion.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yourname',
    maintainer_email='your@email.com',
    description='Run IndustReal policy and publish UR action',
    license='MIT',
    entry_points={
        'console_scripts': [
            'run_policy = ur_policy_runner.policy_node:main',
        ],
    },
)
