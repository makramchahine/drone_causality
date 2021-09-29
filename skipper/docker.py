"""The deepdrone manager for experiments."""

import argparse
import sys
import subprocess

DOCKER_ENV_TAG = "mitdrl/deepdrone:env"
DOCKER_ENV_FILE = "./DockerfileEnv"
DOCKER_TAG = "mitdrl/deepdrone:minicity"
DOCKER_FILE = "./Dockerfile"


class CommandParser(object):
    """A high-level command parser for different use cases."""

    def __init__(self):
        """Initialize the command parser mainly with argument parser."""
        parser = argparse.ArgumentParser(description="A deep minicity")

        parser.add_argument(
            "command",
            nargs=1,
            default=None,
            choices=[
                "build_all",
                "build",
            ],
            help="specify one of the options described above",
        )

        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])

        # use dispatch pattern to invoke method with same name
        getattr(self, args.command[0])()

    def build_all(self):
        """Build docker environment image and push."""
        self._docker_build_push(DOCKER_ENV_TAG, DOCKER_ENV_FILE)
        self.build()

    def build(self):
        """Build and push deployment docker image."""
        self._docker_build_push(DOCKER_TAG, DOCKER_FILE)

    def _docker_build_push(self, tag, file):
        subprocess.run(
            [
                "docker",
                "build",
                "-t",
                tag,
                "-f",
                file,
                ".",
            ]
        )

        subprocess.run(["docker", "push", tag])


if __name__ == "__main__":
    CommandParser()
