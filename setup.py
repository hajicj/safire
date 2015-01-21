from setuptools import setup
import os

###############################################################################


def walk_tail_dir(path):
    """Enumerate all files in the last directory of the path."""

    headdir, taildir = os.path.split(path)
    from_idx = len(headdir) + 1

    fnames = []

    for rootpath, dirs, files in os.walk(path):
        # Do not descend into repository .svn folders
        if '.svn' in dirs:
            dirs.remove('.svn')

        fnames.extend(dirs)
        fnames.extend([ os.path.join(rootpath[from_idx:], f)
                        for f in files if not f.startswith('.svn')])

    return fnames

###############################################################################

DEPENDENCIES = ['numpy>=1.8.0',
                'scipy>=0.13.1',
                'matplotlib>=1.1.3',
                'gensim>=0.10.0',
                'theano>=0.6.0',
                'pillow']

PACKAGES = ['test',
            'safire',
            'safire.corpora',
            'safire.data',
            'safire.datasets',
            'safire.data.filters',
            'safire.utils',
            'safire.learning',
            'safire.learning.models',
            'safire.learning.learners',
            'safire.learning.updaters',
            'safire.learning.interfaces']

PACKAGE_DATA = { 'test' : walk_tail_dir('test/test-data') }

SCRIPTS = ['scripts/annotation_stats.py',
           'scripts/benchmark_datasets.py',
           'scripts/dataset2corpus.py',
           'scripts/dataset_stats.py',
           'scripts/clean.py',
           'scripts/evaluate.py',
           'scripts/filter_by_t2i.py',
           'scripts/generate_annotation_items.py',
           'scripts/generate_corpora.py',
           'scripts/icorp2index.py',
           'scripts/img_index_explorer.py',
           'scripts/normalize_img_dataset.py',
           'scripts/pretrain.py',
           'scripts/pretrain_multimodal.py',
           'scripts/remove_duplicate_images.py',
           'scripts/rename_iids.py',
           'scripts/run.py',
           'scripts/text_preprocessing_explorer.py']

###############################################################################

if __name__ == '__main__':

    setup(
        name='safire',
        version='0.0.1r3',
        url='http://ufal.mff.cuni.cz/grants/cemi',
        license='LGPL',
        author='Jan Hajic jr.',
        author_email='hajicj@ufal.mff.cuni.cz',
        description='A deep learning and experiment management library.',
        install_requires=DEPENDENCIES,
        packages=PACKAGES,
        scripts=SCRIPTS,
        package_data=PACKAGE_DATA
    )
