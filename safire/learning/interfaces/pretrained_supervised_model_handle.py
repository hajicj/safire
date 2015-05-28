
from safire.learning.interfaces.model_handle import ModelHandle

class PretrainedSupervisedModelHandle(ModelHandle):
    """Provides the train/validate/test handle for a learner class.
    This is, so far, just a wrapper for the model plus output of
    the ModelClass.setup() function.

    A handle is the "runnable" interface to a model.
    """
    def __init__(self, model_instance, train, validate, test, run_expr, pretrain):
        """Initializes the handle.

        :type model_instance: safire.learning.models.BaseModel
        :param model_instance: The model to which the handle provieds
                               access. Think about it as a read-only
                               thing.

        :type train: theano.function
        :param train: Theano function used to train the given model.

        :type validate: theano.function
        :param validate: Theano function used to test the given model.

        :type test: theano.function
        :param test: Theano function used to test the given model.

        :type run_expr: theano.function
        :param run_expr: Theano function used to run the model, i.e. transform
            inputs into outputs.

        :type pretrain: list(theano.function)
        :param pretrain: A list of pretraining functions for the model.
        """
        super(PretrainedSupervisedModelHandle, self).__init__(model_instance,
                                                              train, 
                                                              validate,
                                                              test,
                                                              run_expr)
        
        self.pretrain = pretrain
