import inspect

from retrying import retry


class Retry(object):
    def __init__(self, func, *dargs, **dkw):
        self.func = func
        self.wrapped = retry(*dargs, **dkw)(func)
        self.param_validator = self._make_signature_validator()

    def _make_signature_validator(self):
        argspec = inspect.getfullargspec(self.func)

        # For bound functions
        if hasattr(self.func, "im_self") and getattr(self.func, "im_self"):
            argspec.args.pop(0)

        formatted_args = inspect.formatargvalues(
            argspec.args,
            argspec.varargs,
            argspec.varkw,
            dict([(k, None) for k in argspec.args]),
        )

        fndef = "lambda %s: True" % (formatted_args.lstrip("(").rstrip(")"))

        # Generate a function that will validate the params passed to func(). If signatures mismatch it will raise ex.
        param_validator = eval(fndef, {})

        def _validate_func(*args, **kwargs):
            try:
                # We must validate the params before calling the actual function.
                # Failing to do so will raise a TypeError internally which will be treated as a 'valid' exception and will
                # be retried.
                param_validator(*args, **kwargs)
            except TypeError as e:
                raise TypeError(e.message.replace("<lambda>", self.func.__name__))

        return _validate_func

    def __call__(self, *args, **kwargs):
        self.param_validator(*args, **kwargs)

        return self.wrapped(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
