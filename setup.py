from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
]

setup(
    name='dms_datastore_ui',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Web UI for dms_datastore",
    license="MIT",
    author="Nicky Sandhu",
    author_email='psandhu@water.ca.gov',
    url='https://github.com/dwr-psandhu/dms_datastore_ui',
    packages=['dms_datastore_ui'],
    entry_points={
        'console_scripts': [
            'dms_datastore_ui=dms_datastore_ui.cli:main'
        ]
    },
    install_requires=requirements,
    keywords='dms_datastore_ui',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
