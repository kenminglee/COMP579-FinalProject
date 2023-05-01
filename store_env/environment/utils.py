import argparse

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value

def add_env_kwargs_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--env-kwargs', nargs='*', action=ParseKwargs, default={})
    return parser