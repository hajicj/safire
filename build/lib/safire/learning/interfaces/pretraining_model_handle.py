
class PretrainingModelHandle(object):
    """Provides the train/validate/test handle for a learner class.
    This is, so far, just a wrapper for the model plus output of
    the ModelClass.setup() function.

    A handle is the "runnable" interface to a model.
    """
    def __init__(self, model_instance, pretrain):
        """Initializes the handle.

        :type model_instance: safire.learning.models.BaseModel
        :param model_instance: The model to which the handle provieds
                               access. Think about it as a read-only
                               thing.

        :type pretrain: theano.function
        :param pretrain: Theano function used to pretrain the given model.
        Note that the model instance is linked to some other model's layer;
        that's kind of the point for pre-training. However, the passed
        ``model_instance`` can't recover the model it's pre-training; the two
        are perfectly oblivious to each other and the only one who knows they
        are linked is the person who wrote the code. This may change in the
        future.
        """

#
# This doesn't work because of circular imports:
#
#        if not isinstance(model_instance, BaseModel):
#            raise TypeError('Provided model instance is not an instance of BaseModel. (rather: %s)' % str(type(model_instance)))

        self.model_instance = model_instance
        self.model_class = type(model_instance)

        self.pretrain = pretrain
