from setuptools import setup, find_packages
import sys, os.path

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gym'))
from version import VERSION

# Environment-specific dependencies.
extras = {
  'atari': ['atari_py>=0.1.1', 'Pillow', 'PyOpenGL'],
  'box2d': ['Box2D-kengz'],
  'classic_control': ['PyOpenGL'],
  'dart': ['joblib>=0.11', 'pillow', 'pydart2'],
  'mujoco': ['mujoco_py>=1.50', 'imageio'],
  'robotics': ['mujoco_py>=1.50', 'imageio'],
}

# Meta dependency groups.
extras['all'] = [item for group in extras.values() for item in group]

setup(name='gym',
      version=VERSION,
      description='The OpenAI Gym: A toolkit for developing and comparing your reinforcement learning agents.',
      url='https://github.com/openai/gym',
      author='OpenAI',
      author_email='gym@openai.com',
      license='',
      packages=[package for package in find_packages()
                if package.startswith('gym')],
      zip_safe=False,
      install_requires=[
          'scipy', 'numpy>=1.10.4', 'six', 'pyglet>=1.4.0,<=1.5.0', 'cloudpickle>=1.2.0,<1.4.0',
          'enum34~=1.1.6;python_version<"3.4"',
      ],
      extras_require=extras,
      package_data={'gym': [
        'envs/mujoco/assets/*.xml',
        'envs/classic_control/assets/*.png',
        'envs/robotics/assets/LICENSE.md',
        'envs/robotics/assets/fetch/*.xml',
        'envs/robotics/assets/hand/*.xml',
        'envs/robotics/assets/stls/fetch/*.stl',
        'envs/robotics/assets/stls/hand/*.stl',
        'envs/robotics/assets/textures/*.png']
      },
      tests_require=['pytest', 'mock'],
      python_requires='>=3.5',
      classifiers=[
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
      ],
)
