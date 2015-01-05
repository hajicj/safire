import logging
import os
from gensim.models import TfidfModel
from safire.data import FrequencyBasedTransformer
from safire.data.filters.positionaltagfilter import PositionalTagTokenFilter
from safire.data.loaders import MultimodalShardedDatasetLoader, ModelLoader
from safire.learning.interfaces import SafireTransformer, \
    MultimodalClampedSamplerModelHandle
from safire.learning.interfaces.model_handle import BackwardModelHandle
from safire.learning.learners import BaseSGDLearner
from safire.learning.models import RestrictedBoltzmannMachine, \
    DenoisingAutoencoder
from safire.utils.transcorp import reset_vtcorp_input, dimension
from test import SafireTestCase

__author__ = 'Lenovo'

import unittest


class TestMultilayerPipeline(SafireTestCase):
    """Tests a multi-layer pipeline.

    Initialization:

    * train text preprocessing steps
    * train a text model (one layer, RBM)
    * train an image model (one layer, DenoisingAutoencoder)
      * has to be sampleable!
    * train a multimodal model (one joint layer, RBM)
      * Again, has to be sampleable

    Application:

    * load text processing pipeline
    * load image processing pipeline
    * load multimodal model

    * Load text data - completely raw, using the VTextCorpus loaded
      in the text processing pipeline, to preserve transformation
      (has to block ``allow_dict_update``)
    * Run text item through transformer, to get the joint inputs representation
    * Run the clamped sampler with the transformed text item
    * Sample down the image processing pipeline

    """

    @classmethod
    def setUpClass(cls):
        super(TestMultilayerPipeline, cls).setUpClass()

        cls.training_root = cls.data_root
        cls.name = 'test-data'

        # Text preprocessing parameters.
        cls.text_preprocessing_label = '.NAV.top110.pf0.2.pff.tfidf'
        cls.vtargs = {
            'token_filter' : PositionalTagTokenFilter(list('NAV'), 0),
            'pfilter' : 0.2,
            'pfilter_full_freqs' : True,
            'label' : cls.text_preprocessing_label
        }
        cls.top_k = 110
        cls.discard_top = 10

        # Image preprocessing parameters
        cls.img_preprocessing_label = '.default'

        # Learner parameters
        cls.batch_size = 1
        cls.learning_rate = 0.13
        cls.n_epochs = 10
        cls.validation_frequency = 4

        # Text layer 1 parameters
        cls.t1_label = cls.text_preprocessing_label + '.RBM-100'
        cls.t1_n_out = 100

        # Image layer 1 parameters
        cls.i1_label = cls.img_preprocessing_label + '.DA-100'
        cls.i1_n_out = 100
        cls.i1_reconstruction = 'mse'

        # Joint layer parameters
        cls.joint_label = cls.t1_label + '-' + cls.i1_label + '.RBM-100'
        cls.joint_n_out = 100

        cls.joint_sampling_label = cls.joint_label + '.csampling'

        # Test corpus parameters
        cls.runtime_vtlist_filename = 'runtime.vtlist'
        cls.test_root = cls.data_root

    def test_processing_pipeline(self):

        loader = MultimodalShardedDatasetLoader(self.training_root, self.name)
        mloader = ModelLoader(self.training_root, self.name)

        # Initialize/train text preprocessing
        vtcorp = loader.get_text_corpus(self.vtargs)
        text_preprocessing_corpus = vtcorp
        tfidf = TfidfModel(text_preprocessing_corpus)
        text_preprocessing_corpus = tfidf[text_preprocessing_corpus]
        freq_transformer = FrequencyBasedTransformer(text_preprocessing_corpus,
                                                     self.top_k,
                                                     self.discard_top)
        text_preprocessing_corpus = freq_transformer[text_preprocessing_corpus]

        # Save corpora, so that the UnsupervisedShardedVTextCorpusDataset
        # for the next level can be initialized.
        loader.save_text_corpus(text_preprocessing_corpus,
                                self.text_preprocessing_label)
        loader.serialize_text_corpus(text_preprocessing_corpus,
                                     self.text_preprocessing_label)

        # Build dataset (this actually creates the new files; it doesn't make
        # sense to create a ShardedDataset without the shard files...)
        loader.build_text(text_preprocessing_corpus,
                          self.text_preprocessing_label,
                          { 'overwrite' : True } )
        text_dataset = loader.load_text(self.text_preprocessing_label)

        self.assertEqual(100, text_dataset.dim)

        # Initialize first text layer
        t1_handle = RestrictedBoltzmannMachine.setup(text_dataset,
                                                     n_out=self.t1_n_out)
        # Train first text layer
        t1_learner = BaseSGDLearner(self.n_epochs, self.batch_size,
                                 learning_rate=self.learning_rate)
        t1_learner.run(t1_handle, text_dataset)
        # Initialize transformer for first text layer
        t1_transformer = SafireTransformer(t1_handle)
        # Transform the original corpus, save the transformed dataset
        text_processing_corpus = t1_transformer[text_preprocessing_corpus]
        loader.build_text(text_processing_corpus, self.t1_label)
        loader.serialize_text_corpus(text_processing_corpus, self.t1_label)
        loader.save_text_corpus(text_processing_corpus, self.t1_label)

        # Initialize image processing & build dataset
        icorp = loader.get_image_corpus({'delimiter' : ';'})
        # Save corpus, so that it is available for the
        # UnsupervisedShardedImagenetCorpus
        loader.save_image_corpus(icorp, self.img_preprocessing_label)
        loader.serialize_image_corpus(icorp, self.img_preprocessing_label)

        loader.build_img(icorp,
                         self.img_preprocessing_label,
                         { 'overwrite' : True })
        img_dataset = loader.load_img(self.img_preprocessing_label)
        self.assertEqual(4096, img_dataset.dim) # Sanity check


        # Build first layer of image processing
        i1_handle = DenoisingAutoencoder.setup(img_dataset, n_out=self.i1_n_out,
                                        reconstruction=self.i1_reconstruction)
        # Train the first image layer
        i1_learner = BaseSGDLearner(n_epochs=self.n_epochs,
                                    b_size=self.batch_size,
                                    learning_rate=self.learning_rate)
        i1_learner.run(i1_handle, img_dataset)
        # Initialize transformer for first image layer, transform corpus
        # and build dataset.
        i1_transformer = SafireTransformer(i1_handle)
        image_processing_corpus = i1_transformer[icorp]

        loader.build_img(image_processing_corpus, self.i1_label)
        loader.save_image_corpus(image_processing_corpus, self.i1_label)
        loader.serialize_image_corpus(image_processing_corpus, self.i1_label)

        # Now the multimodal joint layer:
        joint_dataset = loader.load(text_infix=self.t1_label,
                                    img_infix=self.i1_label)
        self.assertEqual(0, joint_dataset.mode) # Sanity checks
        self.assertEqual(200, joint_dataset.n_in)
        joint_handle = RestrictedBoltzmannMachine.setup(joint_dataset,
                                                        n_out=self.joint_n_out)
        joint_learner = BaseSGDLearner(n_epochs=self.n_epochs,
                                       b_size=self.batch_size,
                                       learning_rate=self.learning_rate)
        joint_learner.run(joint_handle, joint_dataset)

        #self.assertEqual(True, False)
        # We should now save the joint layer... how? Simply as a transformer?
        # --> this saves the transformation to the hidden units, not the
        #     multimodal clamped sampling transformation! That one needs to be
        #     set up separately.
        joint_transformer = SafireTransformer(joint_handle)
        mloader.save_transformer(joint_transformer, self.joint_label)

        ##########################################################
        # The training part is done, now comes the runtime part. #
        ##########################################################

        # We should try loading everything.
        text_processing_corpus = loader.load_text_corpus(self.t1_label)
        image_processing_corpus = loader.load_image_corpus(self.i1_label)
        joint_transformer = mloader.load_transformer(self.joint_label)

        dim_text = dimension(text_processing_corpus)
        dim_img = dimension(image_processing_corpus)

        # The joint dataset needs to be loaded, so that we know the dimensions
        # for individual modalities. (This is slightly unwieldy... could be
        # derived from top-level handles in text/image pipelines?)
        joint_dataset = loader.load(text_infix=self.t1_label,
                                    img_infix=self.i1_label)
        self.assertEqual(image_processing_corpus.obj.n_out, joint_dataset.dim_img)
        self.assertEqual(text_processing_corpus.obj.n_out, joint_dataset.dim_text)

        # Initialize the multimodal sampling handle
        joint_sampling_handle = MultimodalClampedSamplerModelHandle.clone(joint_transformer.model_handle,
                                                                          dim_text,
                                                                          dim_img)
        t2i_transformer = SafireTransformer(joint_sampling_handle)
        mloader.save_transformer(t2i_transformer, self.joint_sampling_label)

        # Now: which corpus should be transformed through the t2i transformer?
        # --> a text corpus transformed through the processing pipeline!
        joint_sampling_corpus = t2i_transformer[text_processing_corpus]

        # For convenience, we use the same vtlist again in testing.
        # The "correct" way of doing this is to initialize a loader in the
        # runtime data dir with the appropriate name and get the vtlist from
        # there.
        runtime_vtlist = os.path.join(loader.root, self.runtime_vtlist_filename)
        reset_vtcorp_input(text_processing_corpus, runtime_vtlist)

        sampled_imgs = [ img_features for img_features in joint_sampling_corpus ]
        print sampled_imgs[0]

        self.assertEqual(len(sampled_imgs[0]), joint_sampling_handle.n_out)

        # Propagate the sampled image features down to the first image layer.
        backward_handle = BackwardModelHandle.clone(i1_handle)
        i1_backward_transformer = SafireTransformer(backward_handle)

        i1_backward_corpus = i1_backward_transformer[joint_sampling_corpus]

        output_imgs = [ img for img in i1_backward_corpus ]
        print output_imgs[0][:10]
        print len(output_imgs[0])

        self.assertEqual(i1_transformer.n_in, i1_backward_transformer.n_out)


##############################################################################

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestMultilayerPipeline)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
