# -*- coding: utf-8 -*-
"""Parse a Dockerfile and build a udocker container/image from it"""

import os
import re
import glob
import json
import shlex
import tarfile
import time

from udocker.msg import Msg
from udocker.helper.hostinfo import HostInfo
from udocker.helper.unique import Unique
from udocker.utils.fileutil import FileUtil
from udocker.utils.uenv import Uenv
from udocker.utils.chksum import ChkSUM
from udocker.engine.execmode import ExecutionMode


class DockerfileParser(object):
    """Parses a Dockerfile into a list of (INSTRUCTION, argument) tuples.
    Handles comments, blank lines and line continuations (trailing '\\').
    """

    def parse(self, filename):
        """Read and parse a Dockerfile"""
        with open(filename, 'r') as filep:
            raw_lines = filep.readlines()
        return self._parse_lines(raw_lines)

    def _join_continuations(self, raw_lines):
        """Join lines ending in '\\' into a single logical line.
        A '#' comment line is dropped even when it appears in the
        middle of a continued instruction (as docker itself allows).
        """
        logical_lines = []
        buf = ""
        for raw_line in raw_lines:
            line = raw_line.rstrip('\n').rstrip('\r')
            stripped = line.strip()
            if stripped.startswith('#'):
                if not buf:
                    logical_lines.append(line)
                continue
            if not buf and not stripped:
                logical_lines.append(line)
                continue
            rline = line.rstrip()
            if rline.endswith('\\'):
                buf += rline[:-1]
                continue
            buf += line
            logical_lines.append(buf)
            buf = ""
        if buf:
            logical_lines.append(buf)
        return logical_lines

    def _parse_lines(self, raw_lines):
        """Split each logical line into (INSTRUCTION, argument)"""
        instructions = []
        for line in self._join_continuations(raw_lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(None, 1)
            instruction = parts[0].upper()
            argument = parts[1].strip() if len(parts) > 1 else ""
            instructions.append((instruction, argument))
        return instructions

    @staticmethod
    def split_stages(instructions):
        """Split a flat instruction list into build stages at each FROM.
        Returns a list of {"image": str, "name": str|None,
        "instructions": [...]} dicts, in Dockerfile order. A stage may
        be referred to later by COPY --from=<name> or --from=<index>,
        the latter using the stage position as a string, e.g. "0".
        """
        stages = []
        current = None
        for (instruction, arg) in instructions:
            if instruction == "FROM":
                parts = arg.split()
                image = parts[0] if parts else ""
                name = None
                if len(parts) >= 3 and parts[1].upper() == "AS":
                    name = parts[2]
                current = {"image": image, "name": name, "instructions": []}
                stages.append(current)
            else:
                if current is None:
                    raise ValueError(
                        "Dockerfile must start with a FROM instruction")
                current["instructions"].append((instruction, arg))
        return stages


class ContainerBuilder(object):
    """Applies the instructions of a parsed Dockerfile to a udocker
    container previously created from the FROM image, and optionally
    commits the result as a new image in the local repository.

    Limitations: ADD does not fetch remote URLs, and RUN executes using
    the execution mode currently configured for the container.
    """

    SHELL_CMD = ["/bin/sh", "-c"]

    _OPT_DEFAULTS = {
        "nometa": False, "nosysdirs": False, "dri": False,
        "bindhome": False, "hostenv": False, "hostauth": False,
        "containerauth": False, "novol": [], "envfile": [], "vol": [],
        "cpuset": "", "entryp": [], "hostname": "", "domain": "",
        "volfrom": [], "portsmap": [], "portsexp": [], "devices": [],
        "nobanner": True, "netcoop": False, "kernel": "", "platform": "",
        "dns": [], "dnssearch": [],
    }

    def __init__(self, localrepo, container_id, context_dir="."):
        self.localrepo = localrepo
        self.container_id = container_id
        self.container_dir = localrepo.cd_container(container_id)
        self.container_root = self.container_dir + "/ROOT"
        self.context_dir = context_dir
        self.container_json = self.localrepo.load_json(
            self.container_dir + "/container.json") or {}
        self.config = self.container_json.get("config") or {}
        self.container_json["config"] = self.config
        self.envs = Uenv()
        self.envs.extend(self.config.get("Env") or [])
        self.args = {}
        self.build_args = {}
        self.stage_roots = {}

    def set_build_args(self, build_arg_list):
        """Register --build-arg VAR=VALUE overrides"""
        for item in (build_arg_list or []):
            if "=" in item:
                (key, val) = item.split("=", 1)
            else:
                key, val = item, os.environ.get(item, "")
            key = key.strip()
            if key:
                self.build_args[key] = val.strip()

    def set_stage_roots(self, stage_roots):
        """Register the ROOT directories of previously built stages,
        keyed by stage name and by stage index (as a string), so that
        COPY --from=<name-or-index> can resolve its source"""
        self.stage_roots = stage_roots or {}

    def _variables(self):
        """Variables available for ${VAR} / $VAR substitution"""
        variables = dict(self.args)
        for (key, val) in self.envs:
            variables[key] = val
        return variables

    def _expand(self, text):
        """Perform simple ARG/ENV variable substitution"""
        if not text:
            return text
        variables = self._variables()
        pattern = re.compile(
            r'\$(\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))')

        def _sub(match):
            name = match.group(2) or match.group(3)
            if name in variables:
                return variables[name]
            return match.group(0)

        return pattern.sub(_sub, text)

    @staticmethod
    def _ensure_dir(path):
        if path and not os.path.isdir(path):
            try:
                os.makedirs(path)
            except (IOError, OSError):
                pass

    @staticmethod
    def _parse_cmd_form(arg):
        """Parse exec form (JSON array) or shell form (plain string)"""
        arg = arg.strip()
        if arg.startswith('['):
            try:
                cmd = json.loads(arg)
                if isinstance(cmd, list):
                    return (True, [str(item) for item in cmd])
            except ValueError:
                pass
        return (False, arg)

    @staticmethod
    def _is_tar(path):
        return path.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2",
                              ".tbz2", ".tar.xz", ".txz"))

    def run_instructions(self, instructions):
        """Apply all parsed Dockerfile instructions of a single stage,
        in order. FROM is not expected here: the caller splits a
        Dockerfile into per-stage instruction lists beforehand"""
        for (index, (instruction, raw_arg)) in enumerate(instructions):
            handler = getattr(self, "_do_" + instruction.lower(), None)
            if handler is None:
                Msg().out("Warning: ignoring unsupported instruction:",
                          instruction, raw_arg, l=Msg.WAR)
                continue
            arg = self._expand(raw_arg)
            Msg().out("Step %d: %s %s" % (index + 1, instruction, arg),
                      l=Msg.INF)
            if not handler(arg):
                return False
        return True

    def _do_arg(self, arg):
        if "=" in arg:
            (name, default) = arg.split("=", 1)
            name = name.strip()
            default = default.strip().strip('"').strip("'")
        else:
            name = arg.strip()
            default = ""
        if not name:
            Msg().err("Error: ARG requires a name")
            return False
        self.args[name] = self.build_args.get(name, default)
        return True

    def _do_env(self, arg):
        arg = arg.strip()
        if not arg:
            Msg().err("Error: ENV requires arguments")
            return False
        try:
            if "=" in arg.split(None, 1)[0]:
                tokens = shlex.split(arg)
                pairs = []
                for token in tokens:
                    if "=" not in token:
                        Msg().err("Error: invalid ENV assignment:", token)
                        return False
                    pairs.append(tuple(token.split("=", 1)))
            else:
                parts = arg.split(None, 1)
                if len(parts) != 2:
                    Msg().err("Error: invalid ENV instruction:", arg)
                    return False
                pairs = [(parts[0], parts[1])]
        except ValueError as error:
            Msg().err("Error: parsing ENV:", error)
            return False
        for (key, val) in pairs:
            self.envs.append("%s=%s" % (key, val))
        self.config["Env"] = ["%s=%s" % (key, val) for (key, val) in self.envs]
        return True

    def _do_workdir(self, arg):
        workdir = arg.strip()
        if not workdir:
            Msg().err("Error: WORKDIR requires a path")
            return False
        if not workdir.startswith("/"):
            workdir = (self.config.get("WorkingDir") or "/") + "/" + workdir
        workdir = os.path.normpath(workdir)
        self.config["WorkingDir"] = workdir
        self._ensure_dir(self.container_root + workdir)
        return True

    def _do_user(self, arg):
        user = arg.strip()
        if not user:
            Msg().err("Error: USER requires a value")
            return False
        self.config["User"] = user
        return True

    def _do_label(self, arg):
        try:
            tokens = shlex.split(arg)
        except ValueError as error:
            Msg().err("Error: parsing LABEL:", error)
            return False
        labels = dict(self.config.get("Labels") or {})
        for token in tokens:
            if "=" not in token:
                Msg().err("Error: invalid LABEL:", token)
                return False
            (key, val) = token.split("=", 1)
            labels[key] = val
        self.config["Labels"] = labels
        return True

    def _do_expose(self, arg):
        exposed = dict(self.config.get("ExposedPorts") or {})
        for port in arg.split():
            if "/" not in port:
                port += "/tcp"
            exposed[port] = {}
        self.config["ExposedPorts"] = exposed
        return True

    def _do_volume(self, arg):
        try:
            tokens = json.loads(arg) if arg.strip().startswith("[") \
                else shlex.split(arg)
        except ValueError as error:
            Msg().err("Error: parsing VOLUME:", error)
            return False
        volumes = dict(self.config.get("Volumes") or {})
        for vol in tokens:
            volumes[vol] = {}
            self._ensure_dir(self.container_root + vol)
        self.config["Volumes"] = volumes
        return True

    def _do_cmd(self, arg):
        (is_exec, cmd) = self._parse_cmd_form(arg)
        self.config["Cmd"] = cmd if is_exec else self.SHELL_CMD + [cmd]
        return True

    def _do_entrypoint(self, arg):
        (is_exec, cmd) = self._parse_cmd_form(arg)
        self.config["Entrypoint"] = cmd if is_exec else self.SHELL_CMD + [cmd]
        return True

    def _copy_one(self, src, dest_path, dest_is_dir, allow_extract):
        """Copy or extract a single resolved source path into dest_path"""
        if os.path.isdir(src):
            self._ensure_dir(dest_path)
            return FileUtil(src).copydir(dest_path)
        if allow_extract and self._is_tar(src):
            target_dir = dest_path if dest_is_dir else os.path.dirname(dest_path)
            self._ensure_dir(target_dir)
            try:
                with tarfile.open(src) as tfile:
                    tfile.extractall(target_dir)
            except (tarfile.TarError, IOError, OSError) as error:
                Msg().err("Error: extracting:", src, error)
                return False
            return True
        target = os.path.join(dest_path, os.path.basename(src)) \
            if dest_is_dir else dest_path
        self._ensure_dir(os.path.dirname(target))
        return FileUtil(src).copyto(target)

    def _copy_or_add(self, arg, allow_extract):
        try:
            tokens = shlex.split(arg)
        except ValueError as error:
            Msg().err("Error: parsing COPY/ADD arguments:", error)
            return False
        from_stage = None
        plain_tokens = []
        for tok in tokens:
            if tok.startswith("--from="):
                from_stage = tok.split("=", 1)[1]
            elif tok.startswith("--"):
                continue
            else:
                plain_tokens.append(tok)
        tokens = plain_tokens
        if len(tokens) < 2:
            Msg().err("Error: COPY/ADD requires source(s) and a destination:",
                      arg)
            return False
        dest_arg = tokens[-1]
        sources = tokens[:-1]
        dest = dest_arg if dest_arg.startswith("/") else \
            (self.config.get("WorkingDir") or "/").rstrip("/") + "/" + dest_arg
        dest_path = os.path.normpath(self.container_root + dest)
        dest_is_dir = dest_arg.endswith("/") or len(sources) > 1 or \
            os.path.isdir(dest_path)
        if dest_is_dir:
            self._ensure_dir(dest_path)

        if from_stage is not None:
            src_root = self.stage_roots.get(from_stage)
            if src_root is None:
                Msg().err("Error: COPY --from stage not found:", from_stage)
                return False
        else:
            src_root = self.context_dir

        for src in sources:
            if from_stage is None and re.match(r'^https?://', src):
                Msg().err("Error: fetching remote sources is not supported:",
                          src)
                return False
            matches = glob.glob(os.path.join(src_root, src.lstrip("/")))
            if not matches:
                Msg().err("Error: COPY/ADD source not found:", src)
                return False
            for match in matches:
                if not self._copy_one(match, dest_path, dest_is_dir,
                                      allow_extract):
                    return False
        return True

    def _do_copy(self, arg):
        return self._copy_or_add(arg, allow_extract=False)

    def _do_add(self, arg):
        return self._copy_or_add(arg, allow_extract=True)

    def _reset_exec_opt(self, exec_engine):
        """Reset the shared execution engine options before a RUN"""
        for (key, val) in self._OPT_DEFAULTS.items():
            exec_engine.opt[key] = list(val) if isinstance(val, list) else val
        exec_engine.opt["user"] = self.config.get("User") or ""
        exec_engine.opt["cwd"] = self.config.get("WorkingDir") or ""
        env = Uenv()
        for (key, val) in self.envs:
            env.append("%s=%s" % (key, val))
        exec_engine.opt["env"] = env
        exec_engine.opt["cmd"] = []

    def _do_run(self, arg):
        if not arg.strip():
            Msg().err("Error: RUN requires a command")
            return False
        (is_exec, cmd) = self._parse_cmd_form(arg)
        cmd_list = cmd if is_exec else self.SHELL_CMD + [cmd]
        exec_mode = ExecutionMode(self.localrepo, self.container_id)
        exec_engine = exec_mode.get_engine()
        if not exec_engine:
            Msg().err("Error: no execution engine available for RUN")
            return False
        self._reset_exec_opt(exec_engine)
        exec_engine.opt["cmd"] = cmd_list
        status = exec_engine.run(self.container_id)
        if status:
            Msg().err("Error: RUN instruction failed with exit status",
                      status, ":", arg)
            return False
        return True

    def commit(self, imagerepo, tag):
        """Create a new local image from the current container filesystem
        and accumulated metadata"""
        self.localrepo.setup_imagerepo(imagerepo)
        if not self.localrepo.setup_tag(tag):
            Msg().err("Error: setting up image repository and tag")
            return False
        if not self.localrepo.set_version("v1"):
            Msg().err("Error: setting repository version")
            return False
        layer_id = Unique().layer_v1()
        layer_file = self.localrepo.layersdir + '/' + layer_id + ".layer"
        json_file = self.localrepo.layersdir + '/' + layer_id + ".json"
        if not FileUtil(self.container_root).tar(layer_file):
            Msg().err("Error: creating image layer from container filesystem")
            return False
        self.localrepo.add_image_layer(layer_file)
        self.localrepo.save_json("ancestry", [layer_id])
        self.container_json["id"] = layer_id
        self.container_json["comment"] = "created by udocker build"
        self.container_json["created"] = \
            time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        if not self.container_json.get("architecture"):
            self.container_json["architecture"] = HostInfo().arch("docker")
        if not self.container_json.get("os"):
            self.container_json["os"] = HostInfo().osversion()
        self.container_json["size"] = FileUtil(layer_file).size()
        layer_chksum = ChkSUM().hash(layer_file, "sha256")
        if layer_chksum:
            self.container_json["rootfs"] = {
                "type": "layers", "diff_ids": ["sha256:" + layer_chksum]}
        self.container_json["config"] = self.config
        self.container_json["container_config"] = self.config
        if not self.localrepo.save_json(json_file, self.container_json):
            Msg().err("Error: saving image metadata")
            return False
        self.localrepo.add_image_layer(json_file)
        return True
