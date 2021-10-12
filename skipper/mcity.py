#!/usr/bin/env python3
"""The deepdrone manager for experiments."""

import argparse
import sys
import subprocess
import os
import re
import yaml

DOCKER_ENV_TAG = "mitdrl/deepdrone:env"
DOCKER_ENV_FILE = "./DockerfileEnv"
DOCKER_TAG = "mitdrl/deepdrone:minicity"
DOCKER_FILE = "./Dockerfile"

TRAIN_CONFIG_DEFAULT = "default/train_config.yaml"
SKIPPER_CONFIG_DEFAULT = "default/pod_config.yaml"

ARG_PARSE_DESCRIPTION = "A deep minicity."
ARG_PARSE_USAGE = """

To run the 'r-deepdrone-bidirect.yaml' with 5 repetitions, 1 GPU per job,
sequence length 128, and lowest skipper priority, run:

./skipper/mcity.py deploy \\
    --config-file config/r-deepdrone-bidirect.yaml \\
    --seq-len 128 \\
    --skipper-args '-g1 -j5 -p lowest'


Run the following command for MORE HELP:

./skipper/mcity.py --help

 
"""


class CommandParser(object):
    """A high-level command parser for different use cases."""

    def __init__(self):
        """Initialize the command parser mainly with argument parser."""
        parser = argparse.ArgumentParser(
            description=ARG_PARSE_DESCRIPTION, usage=ARG_PARSE_USAGE
        )

        # add a subparser for each option
        subparsers = parser.add_subparsers(
            help="commands", dest="command", required=True
        )

        # docker build all subparser
        args_docker_build_all = subparsers.add_parser("docker_build_all")
        args_docker_build = subparsers.add_parser("docker_build")
        args_docker_run = subparsers.add_parser("docker_run")

        # skipper deploy subparser
        args_deploy = subparsers.add_parser("deploy")

        args_deploy.add_argument(
            "--skipper-args",
            "-a",
            default="",
            type=str,
            help="Specify any skipper apply args as one string! (except -f).",
            dest="skipper_args",
        )

        args_deploy.add_argument(
            "--config-file",
            "-f",
            help="specify training configuration yaml file.",
            dest="config_file",
            required=True,
        )

        args_deploy.add_argument(
            "--seq-len",
            "-l",
            default=64,
            type=int,
            help="Modify the sequence length for the run.",
            dest="seq_len",
        )

        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:])

        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)(args)

    def docker_build_all(self, args):
        """Build docker environment image and push."""
        self._docker_build_push(DOCKER_ENV_TAG, DOCKER_ENV_FILE)
        self.build()

    def docker_build(self, args):
        """Build and push deployment docker image."""
        self._docker_build_push(DOCKER_TAG, DOCKER_FILE)

    def docker_run(self, args):
        """Run the container interactively locally with one GPU."""
        subprocess.run(
            ["docker", "run", "-it", "--gpus", "device=0", DOCKER_TAG, "bash"]
        )

    def _docker_build_push(self, tag, file):
        self._docker_build(tag, file)
        self._docker_push(tag)

    def _docker_build(self, tag, file):
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

    def _docker_push(self, tag):
        subprocess.run(["docker", "push", tag])

    def deploy(self, args):
        """Now deploy to skipper."""

        # parse train args.
        train_args = self._parse_train_config(args.config_file, args.seq_len)

        # get path to skipper config file
        skipper_file = os.path.join(
            self._get_script_path(), SKIPPER_CONFIG_DEFAULT
        )

        # get default skipper config
        skipper_config = {}
        with open(skipper_file, "r") as ymlfile:
            skipper_config.update(yaml.full_load(ymlfile))

        # customize skipper config file now
        skipper_config["spec"]["containers"][0]["args"][0] = train_args
        pod_name = os.path.splitext(args.config_file)[0]
        pod_name += f"_seq{args.seq_len}"
        pod_name = "-".join(re.split("/|-|_", pod_name))
        skipper_config["metadata"]["name"] = pod_name

        # save as temp file for skipper to find.
        tmp_filename = f"/tmp/skipper-pod-{pod_name}.yaml"
        with open(tmp_filename, "w") as config_file:
            config_file.write(yaml.dump(skipper_config))

        # now deploy to skipper
        deploy_cmd = [
            "skipper",
            "apply",
            "-f",
            tmp_filename,
            *args.skipper_args.split(),
        ]
        subprocess.run(deploy_cmd)

    def _parse_train_config(self, config_file, seq_len):
        """Build up training configuration from standard yaml + custom."""
        # get script path and assume everything is relative to script path
        script_path = self._get_script_path()

        # get full file paths
        config_file = os.path.join(script_path, config_file)
        config_file_default = os.path.join(script_path, TRAIN_CONFIG_DEFAULT)

        # start filling configurations
        config_train = {}

        def _fill_config(file_name):
            with open(file_name, "r") as ymlfile:
                config_train.update(yaml.full_load(ymlfile))

        _fill_config(config_file_default)
        _fill_config(config_file)

        # also make sure seq_len corresponds to provided arg
        config_train["seq_len"] = seq_len

        # now convert from dictionary format (key: value) to args format, which
        # is a list of ["--key1", "value1", "--key2", "value2", ...]
        train_args = sum(
            [[f"--{k}", str(v)] for k, v in config_train.items()],
            start=["python", "tf_data_training.py"],
        )

        return " ".join(train_args)

    def _get_script_path(self):
        script_path = os.path.realpath(__file__)
        return os.path.split(script_path)[0]


if __name__ == "__main__":
    print(os.path.realpath(__file__))
    CommandParser()
