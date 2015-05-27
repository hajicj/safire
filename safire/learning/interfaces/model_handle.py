import cPickle
import gensim
import logging
import theano
import safire
from safire.learning.interfaces.clamped_sampler import MultimodalClampedSampler


class ModelHandle(object):
    """Provides the train/validate/test/... functionality for a learner class
    through its ``run`` method. During model setup, a training handle is created
    that ``run``s the theano function that performs one training step,
    a validation handle gets the theano function for a validation batch, etc.

    A handle works like the "runnable" interface to a model. It is what makes
    the model "do something". If the model itself had to provide all the actions
    that one can do with it, the interface you would have to implement for each
    new model would be very, very clumsy. This way, you can implement various
    functionalities for a single model through various handles, depending on
    how you define the handle's ``run()`` method.

    The handle keeps the reference to the underlying model and its class.

    Also, a ModelHandle provides save/load functionality for model persistence.
    """
    def __init__(self, model_instance, run):
        """Initializes the handle.

        :type model_instance: safire.learning.models.BaseModel
        :param model_instance: The model to which the handle provieds
                               access. Think about it as a read-only
                               thing.

        :type run: theano.function
        :param run: Theano function used to run the model, i.e. transform
            inputs into outputs.
        """

        self.model_instance = model_instance
        self.model_class = type(model_instance)

        self.n_in = self.model_instance.n_in
        self.n_out = self.model_instance.n_out

        self.run = run

    def save(self, filename, protocol=-1):
        """Saves the handle. Uses the ``_init_arg_snapshot()`` method of the
        model to save the model, attempts to save the theano functions
        directly."""

        save_dict = self._export_pickleable_obj()

        with open(filename, 'wb') as pickle_handle:
            cPickle.dump(save_dict, pickle_handle, protocol)

    def _export_pickleable_obj(self):
        """
        Exports a dicitonary that can directly be pickled to sufficiently
        describe the handle.
        """
        if self.model_instance is not None:
            model_pickleable_obj = self.model_instance._export_pickleable_object()
        else:
            model_pickleable_obj = None

        init_args = {'train': self.train,
                     'validate': self.validate,
                     'test': self.test,
                     'run': self.run}

        save_dict = {'model': model_pickleable_obj,
                     'init_args': init_args}

        return save_dict

    @classmethod
    def load(cls, filename):
        """Loads handle from previously saved handle.
        """
        with open(filename, 'rb') as unpickle_handle:
            save_dict = cPickle.load(unpickle_handle)

        model_handle = cls._load_from_save_dict(save_dict)
        return model_handle

    @classmethod
    def _load_from_save_dict(cls, save_dict):

        if save_dict['model'] is not None:
            model_init_args = save_dict['model']['init_args']
            model_class = save_dict['model']['class']

            logging.debug('Handle: loading model class: %s' % str(model_class))

            model = model_class(**model_init_args)

        else:
            model = save_dict['model']

        handle_init_args = save_dict['init_args']
        model_handle = ModelHandle(model_instance=model, **handle_init_args)

        return model_handle


class AlternatingGibbsHandle(ModelHandle):
    """Runs an alternating Gibbs chain from a sampleable handle. The last step
    is a mean, all others are samples."""
    def __init__(self):
        pass


class BackwardModelHandle(ModelHandle):
    """Performs a backward pass on the model, instead of forward.
    Requires a sampleable model.

    The ``run`` member of this handle will expect a *hidden* activation
    and will compute the *visible* activation for the given model.

    """
    def __init__(self, model_instance, run,
                 heavy_debug=False):
        """Initializes the handle. Assumes the model is sample-able.

        :type model_instance: safire.learning.models.BaseModel
        :param model_instance: The model to which the handle provides
            access. Think about it as a read-only thing.

        :type run: theano.function
        :param run: Theano function used to run the model, i.e. transform
            inputs into outputs. However, this function will not be used
            as the ``run`` member of the handle; it will remain as the more
            obscure ``get_hidden`` member.

        :type heavy_debug: bool
        :param heavy_debug: If set, will compile the sampling functions in
            theano's ``MonitorMode`` with a detailed input/output/node printing
            function.
        """
        self.model_instance = model_instance
        self.model_class = model_instance.__class__

        # self.train = train
        # self.validate = validate
        # self.test = test
        #
        self.get_hidden = run

        self.heavy_debug = heavy_debug

        self.backward_mean = self.init_backward_run(sample_visible=False)
        self.backward_sample = self.init_backward_run(sample_visible=True)

        self.n_in = self.model_instance.n_out
        self.n_out = self.model_instance.n_in

    def init_backward_run(self, sample_visible=False):
        """Creates the Theano function  to compute the backward mean."""
        h = theano.tensor.matrix('backward_h', dtype=theano.config.floatX)

        if sample_visible:
            v = self.model_instance.sample_v_given_h(h)
        else:
            v = self.model_instance.mean_v_given_h(h)

        vhv_kwargs = {}
        if self.heavy_debug:
            vhv_kwargs['mode'] = theano.compile.MonitorMode(
                post_func=safire.utils.merciless_print).excluding(
                    'local_elemwise_fusion', 'inplace')

        backward_run = theano.function(inputs=[h],
                                       outputs=v,
                                       allow_input_downcast=True,
                                       **vhv_kwargs)
        return backward_run

    def run(self, hidden, sample=False):

        if sample:
            result = self.backward_sample(hidden)
        else:
            result = self.backward_mean(hidden)
        return result

    @classmethod
    def clone(cls, handle):
        """Clones a BackwardModelHandle from an existing ModelHandle.
        We needed the dataset anyway to call model ``setup`` that created
        the first handle.

        :type handle: ModelHandle
        :param handle: The model handle from which to take the model instance,
            train, validate and test functions. The ``run`` function is
            taken as well, but tucked away under a ``handle.get_hidden()``
            member, since the "product" of this handle are not the hidden
            values.
        """
        if hasattr(handle, 'get_hidden'):
            run_fn = handle.get_hidden
        else:
            run_fn = handle.run

        cl_handle = BackwardModelHandle(handle.model_instance,
                                        run_fn)
        return cl_handle

    def _export_pickleable_obj(self):
        """
        Exports a dicitonary that can directly be pickled to sufficiently
        describe the handle.
        """
        model_pickleable_obj = self.model_instance._export_pickleable_object()

        init_args = {'run': self.get_hidden,
                     'dim_text': self.n_in,
                     'dim_img': self.n_out}

        save_dict = {'model': model_pickleable_obj,
                     'init_args': init_args}

        return save_dict


class MultimodalClampedSamplerModelHandle(ModelHandle):
    """This handle provides special ``run()`` functionality. Instead of using
    the ``run`` function given by a model's ``setup``, it will use
    a MultimodalClampedSampler to sample image one part of input values
    based on clamping the values of the other part.

    To set up a clamped sampling handle:

    >>> mdloader = MultimodalShardedDatasetLoader(root_orig, name_orig)
    >>> multimodal_dataset = mdloader.load(text_infix, img_infix)
    >>> handle = RestrictedBoltzmannMachine.setup(multimodal_dataset, n_out=1000)
    >>>
    >>> clamped_handle = MultimodalClampedSamplerModelHandle.clone(handle, multimodal_dataset)
    >>>
    >>> transformer = SafireTransformer(clamped_handle)

    To run the handle: the text data can come from a different dataset,
    but bear in mind they need to share the same feature space.

    >>> run_mdloader = MultimodalShardedDatasetLoader(root_run, name_run)
    >>> text_dataset = run_mdloader.load_text(text_infix)
    >>> text = text_dataset.test_X_batch(0, 1)
    >>> img = clamped_handle.run(text)



    """

    def __init__(self, model_instance, run,
                 dim_text, dim_img, k=10,
                 sample_hidden=True, sample_visible=True):
        """Initializes the handle. Assumes the model is sample-able.

        :type model_instance: safire.learning.models.BaseModel
        :param model_instance: The model to which the handle provides
            access. Think about it as a read-only thing.

        :type run: theano.function
        :param run: Theano function used to run the model, i.e. transform
            inputs into outputs. However, this function will not be used
            as the ``run`` member of the handle; it will remain as the more
            obscure ``get_hidden`` member.

        :type dim_text: int
        :param dim_text: The size of the text feature section from the beginning
            of the multimodal dataset features.

        :type dim_img: int
        :param dim_img: The size of the image feature section from the end
            of the multimodal dataset features.

        :type k: int
        :param k: The number of sampling steps to produce an image sample.
            Note that setting ``k=1`` will not sample anything at all and use
            the hidden/visible mean in the one vhv step.

        :type sample_visible: bool
        :param sample_visible: If set, will sample the visible layer during the
            first ``k - 1`` steps. Otherwise, uses visible mean. Note that
            sampling only works when the visible layer activations are in the
            range ``(0, 1)``. True by default.

        :type sample_hidden: bool
        :param sample_hidden: If set, will sample the hidden layer during the
            first ``k - 1`` steps. Otherwise, uses hidden mean. Note that
            sampling only works when the hidden layer activations are in the
            range ``(0, 1)``. True by default.

        """
        self.model_instance = model_instance
        self.model_class = type(model_instance)

        self.sampler = MultimodalClampedSampler(self.model_instance,
                                                dim_text=dim_text,
                                                dim_img=dim_img)
        self.k = k

        # A trick for transparent swapping of normal handles/clamped sampling
        # handles in SafireTransformer
        self.n_in = self.sampler.n_in
        self.n_out = self.sampler.n_out

        self.sample_hidden = sample_hidden
        self.sample_visible = sample_visible

        # The 'run' function is implemented differently.
        self.get_hidden = run

    @classmethod
    def clone(cls, handle, dim_text, dim_img, **clamped_handle_kwargs):
        """Clones a clamped sampling handle from the existing ModelHandle.
        We needed the dataset anyway to call model ``setup`` that created
        the first handle.

        :type handle: ModelHandle
        :param handle: The model handle from which to take the model instance,
            train, validate and test functions. The ``run`` function is
            taken as well, but tucked away under a ``handle.get_hidden()``
            member, since the "product" of this handle are not the hidden
            values.
        """
        # Different types of handles redefine their ``run`` function.
        if hasattr(handle, 'get_hidden'):
            run_fn = handle.get_hidden
        else:
            run_fn = handle.run

        cl_handle = MultimodalClampedSamplerModelHandle(handle.model_instance,
                                                        run_fn,
                                                        dim_text,
                                                        dim_img,
                                                        **clamped_handle_kwargs)
        return cl_handle

    def run(self, text_features):
        """Runs the sampler on text features input. Returns image
        representation."""
        img = self.sampler.t2i_run_chain_mean_last(
            text_features=text_features,
            k=self.k,
            sample_hidden=self.sample_hidden,
            sample_visible=self.sample_visible)
        return img

    def _export_pickleable_obj(self):
        """
        Exports a dicitonary that can directly be pickled to sufficiently
        describe the handle.
        """
        model_pickleable_obj = self.model_instance._export_pickleable_object()

        init_args = {'train': self.train,
                     'validate': self.validate,
                     'test': self.test,
                     'run': self.get_hidden,
                     'dim_text': self.n_in,
                     'dim_img': self.n_out,
                     'k': self.k}

        save_dict = {'model': model_pickleable_obj,
                     'init_args': init_args}

        return save_dict


class LasagneModelHandle(ModelHandle):
    """An implementation of ModelHandle that deals with Lasagne models."""
    def __init__(self, model_instance, run):
        """Initializes the handle.

        :type model_instance: lasagne.layers.base.Layer
        :param model_instance: The model to which the handle provieds
                               access. Think about it as a read-only
                               thing.

        :type run: theano.function
        :param run: Theano function used to run the model, i.e. transform
            inputs into outputs.
        """
        try:
            import lasagne
            if not isinstance(model_instance, lasagne.layers.base.Layer):
                raise TypeError('Initializing LasagneHandle with non-Lasagne'
                                ' model of type {0}.'
                                ''.format(type(model_instance)))
        except ImportError:
            raise ImportError('Cannot use LasagneHandle without being able to'
                              ' import Lasagne!')

        self.model_instance = model_instance
        self.model_class = type(model_instance)

        # Lasagne initializes input/output shapes not only as dimension of one
        # vector but also with minibatch size, because it always wants to use
        # the ``givens`` mechanism of theano.function.
        self.n_in = self.model_instance.input_shape[1]
        self.n_out = self.model_instance.output_shape[1]

        self.run = run

    def _export_pickleable_obj(self):
        """
        Exports a dicitonary that can directly be pickled to sufficiently
        describe the handle.
        """
        return self

    @classmethod
    def _load_from_save_dict(cls, save_dict):

        return save_dict