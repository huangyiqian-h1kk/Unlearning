import ast
import contextlib
import hashlib
import importlib
import inspect
import json
import pathlib
import re
import subprocess
import sys
import types
import unittest
import warnings

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs" / "llm2vec_migration_contract.json"
COMPLETION_COMMIT = "2ca1a47ff87fa0376725618a39d092a868b0bfa5"
COMPLETION_TREE = "2ba2f20310c8ae5028f704e58620cf8af95baac9"
MISSING = object()


def git(*args):
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout


def path_hash(paths):
    return hashlib.sha256("".join(path + "\n" for path in sorted(paths)).encode("utf-8")).hexdigest()


def index_entries():
    entries = {}
    for record in git("ls-files", "-s", "-z").split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, blob_sha, stage = metadata.split()
        entries[raw_path.decode("utf-8")] = {
            "mode": mode.decode("ascii"),
            "blob_sha": blob_sha.decode("ascii"),
            "stage": stage.decode("ascii"),
        }
    return entries


def method_ast_hash(path, class_name, method_name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )
    return hashlib.sha256(ast.dump(method, include_attributes=False).encode("utf-8")).hexdigest()


def absolute_imports(path, text=None):
    if text is None:
        text = (ROOT / path).read_text(encoding="utf-8")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text, filename=path)
    except SyntaxError:
        imports = []
        pattern = re.compile(r"^\s*from\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import\s+([^#\n]+)", re.MULTILINE)
        for match in pattern.finditer(text):
            module = match.group(1)
            for raw_symbol in match.group(2).split(","):
                symbol = raw_symbol.strip().split(" as ", 1)[0]
                if symbol:
                    imports.append((module, symbol))
        return imports
    return [
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0
        for alias in node.names
    ]


class ArrayTensor:
    """Small NumPy-backed tensor surface for dependency-free pooling contracts."""

    def __init__(self, value):
        self.array = np.asarray(value)

    @property
    def shape(self):
        return self.array.shape

    @property
    def device(self):
        return "cpu"

    def sum(self, dim=None):
        return ArrayTensor(self.array.sum(axis=dim))

    def mean(self, dim=None):
        return ArrayTensor(self.array.mean(axis=dim))

    def unsqueeze(self, dim):
        return ArrayTensor(np.expand_dims(self.array, axis=dim))

    def __iter__(self):
        for value in self.array:
            yield value.item() if np.ndim(value) == 0 else ArrayTensor(value)

    @staticmethod
    def _unwrap(value):
        return value.array if isinstance(value, ArrayTensor) else value

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = tuple(self._unwrap(item) for item in key)
        else:
            key = self._unwrap(key)
        return ArrayTensor(self.array[key])

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            key = tuple(self._unwrap(item) for item in key)
        else:
            key = self._unwrap(key)
        self.array[key] = self._unwrap(value)

    def __add__(self, other):
        return ArrayTensor(self.array + self._unwrap(other))

    def __mul__(self, other):
        return ArrayTensor(self.array * self._unwrap(other))

    def __truediv__(self, other):
        return ArrayTensor(self.array / self._unwrap(other))

    def __itruediv__(self, other):
        self.array = self.array / self._unwrap(other)
        return self

    def __eq__(self, other):
        return ArrayTensor(self.array == self._unwrap(other))


def make_dependency_stubs():
    class Module:
        def __init__(self, *args, **kwargs):
            super().__init__()

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    class Device:
        pass

    class LoadedModel:
        def __init__(self, loader_name):
            self.loader_name = loader_name
            self.config = types.SimpleNamespace(_name_or_path="fixture-model")

    class Loader:
        calls = []

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.calls.append((args, kwargs))
            return LoadedModel(cls.__name__)

    class AutoModel(Loader):
        calls = []

    class AutoModelForCausalLM(Loader):
        calls = []

    class MistralBiModel(Loader):
        calls = []

    class LlamaBiModel(Loader):
        calls = []

    class GemmaBiModel(Loader):
        calls = []

    class Qwen2BiModel(Loader):
        calls = []

    class BaseConfig:
        def __init__(self, name="fixture-model"):
            self._name_or_path = name

    class MistralConfig(BaseConfig):
        pass

    class LlamaConfig(BaseConfig):
        pass

    class GemmaConfig(BaseConfig):
        pass

    class Qwen2Config(BaseConfig):
        pass

    class AutoConfig:
        calls = []

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.calls.append((args, kwargs))
            return MistralConfig()

    class PretrainedConfig:
        @classmethod
        def from_dict(cls, values):
            return BaseConfig(values.get("_name_or_path", "fixture-model"))

    class StubTokenizer:
        def __init__(self):
            self.eos_token = "<eos>"
            self.pad_token = None
            self.padding_side = "right"
            self.bos_token_id = 101

    class AutoTokenizer:
        calls = []

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.calls.append((args, kwargs))
            return StubTokenizer()

    class PeftModel:
        @classmethod
        def from_pretrained(cls, model, path):
            return model

    class CausalLMOutputWithPast:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    torch_module = types.ModuleType("torch")
    torch_module.__path__ = []
    nn_module = types.ModuleType("torch.nn")
    nn_module.Module = Module
    nn_module.Embedding = type("Embedding", (), {})
    multiprocessing_module = types.ModuleType("torch.multiprocessing")
    multiprocessing_module.get_context = lambda name: None
    torch_module.Tensor = ArrayTensor
    torch_module.device = Device
    torch_module.nn = nn_module
    torch_module.multiprocessing = multiprocessing_module
    torch_module.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    torch_module.zeros_like = lambda value: ArrayTensor(np.zeros_like(value.array))
    torch_module.ones = lambda length: ArrayTensor(np.ones(length))
    torch_module.cat = lambda values, dim=0: ArrayTensor(np.concatenate([value.array for value in values], axis=dim))
    torch_module.stack = lambda values, dim=0: ArrayTensor(np.stack([value.array for value in values], axis=dim))
    torch_module.zeros = lambda *shape, device=None: ArrayTensor(np.zeros(shape))
    torch_module.arange = lambda length: ArrayTensor(np.arange(length))
    torch_module.clamp = lambda value, min=None: ArrayTensor(np.maximum(ArrayTensor._unwrap(value), min))
    torch_module.sum = lambda value, dim=None: value.sum(dim=dim)
    torch_module.float32 = np.float32

    peft_module = types.ModuleType("peft")
    peft_module.PeftModel = PeftModel

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.__path__ = []
    autonotebook_module = types.ModuleType("tqdm.autonotebook")
    autonotebook_module.tqdm = lambda iterable=None, *args, **kwargs: iterable
    autonotebook_module.trange = lambda *args, **kwargs: range(*args)
    tqdm_module.autonotebook = autonotebook_module

    transformers_module = types.ModuleType("transformers")
    transformers_module.__path__ = []
    for cls in (
        AutoModel,
        AutoModelForCausalLM,
        AutoConfig,
        PretrainedConfig,
        AutoTokenizer,
        LlamaConfig,
        MistralConfig,
        GemmaConfig,
        Qwen2Config,
    ):
        setattr(transformers_module, cls.__name__, cls)
    outputs_module = types.ModuleType("transformers.modeling_outputs")
    outputs_module.CausalLMOutputWithPast = CausalLMOutputWithPast
    transformers_module.modeling_outputs = outputs_module

    model_exports = {
        "MistralBiModel": MistralBiModel,
        "LlamaBiModel": LlamaBiModel,
        "GemmaBiModel": GemmaBiModel,
        "Qwen2BiModel": Qwen2BiModel,
    }
    canonical_models = types.ModuleType("llm2vec.models")
    derivative_models = types.ModuleType("llm22vec.models")
    for name, value in model_exports.items():
        setattr(canonical_models, name, value)
        setattr(derivative_models, name, value)

    modules = {
        "torch": torch_module,
        "torch.nn": nn_module,
        "torch.multiprocessing": multiprocessing_module,
        "peft": peft_module,
        "tqdm": tqdm_module,
        "tqdm.autonotebook": autonotebook_module,
        "transformers": transformers_module,
        "transformers.modeling_outputs": outputs_module,
        "llm2vec.models": canonical_models,
        "llm22vec.models": derivative_models,
    }
    classes = types.SimpleNamespace(
        AutoModel=AutoModel,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoConfig=AutoConfig,
        AutoTokenizer=AutoTokenizer,
        MistralBiModel=MistralBiModel,
        LlamaBiModel=LlamaBiModel,
        GemmaBiModel=GemmaBiModel,
        Qwen2BiModel=Qwen2BiModel,
        MistralConfig=MistralConfig,
        LlamaConfig=LlamaConfig,
        GemmaConfig=GemmaConfig,
        Qwen2Config=Qwen2Config,
        CausalLMOutputWithPast=CausalLMOutputWithPast,
    )
    return modules, classes


@contextlib.contextmanager
def loaded_legacy_packages():
    dependency_modules, classes = make_dependency_stubs()
    package_names = {
        "llm2vec",
        "llm2vec.llm2vec",
        "llm2vec.models",
        "llm22vec",
        "llm22vec.llm22vec",
        "llm22vec.models",
        "llm22vec.openunlearn_wrapper",
    }
    managed_names = set(dependency_modules) | package_names
    previous = {name: sys.modules.get(name, MISSING) for name in managed_names}
    previous_dont_write = sys.dont_write_bytecode
    package_root = str(ROOT / "llm2vec")
    sys.path.insert(0, package_root)
    sys.dont_write_bytecode = True
    try:
        for name in package_names:
            sys.modules.pop(name, None)
        sys.modules.update(dependency_modules)
        canonical = importlib.import_module("llm2vec")
        derivative = importlib.import_module("llm22vec")
        wrapper = importlib.import_module("llm22vec.openunlearn_wrapper")
        yield types.SimpleNamespace(
            canonical=canonical,
            derivative=derivative,
            wrapper=wrapper,
            classes=classes,
        )
    finally:
        sys.dont_write_bytecode = previous_dont_write
        if sys.path and sys.path[0] == package_root:
            sys.path.pop(0)
        else:
            sys.path.remove(package_root)
        for name, value in previous.items():
            if value is MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        importlib.invalidate_caches()


class Phase3D1LLM2VecMigrationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        cls.phase = cls.contract["phase_scope"]

    def test_phase_scope_is_exact_and_additive(self):
        self.assertEqual(self.contract["schema_version"], "1.0")
        self.assertEqual(self.contract["record_kind"], "phase3d1_llm2vec_migration_contract")
        self.assertEqual(git("rev-parse", self.phase["base_commit"] + "^{tree}").decode().strip(), self.phase["base_tree"])
        expected_added = {
            "docs/llm2vec_migration_contract.json",
            "docs/llm2vec_migration_contract.md",
            "tests/test_llm2vec_migration_contract.py",
        }
        expected_modified = {"tests/test_source_inventory.py"}
        self.assertEqual(set(self.phase["allowed_additive_paths"]), expected_added)
        self.assertEqual(set(self.phase["allowed_modified_paths"]), expected_modified)
        self.assertEqual(
            git("rev-parse", COMPLETION_COMMIT + "^{tree}").decode().strip(),
            COMPLETION_TREE,
        )
        changed = {}
        for line in git(
            "diff", "--name-status", self.phase["base_commit"], COMPLETION_COMMIT, "--"
        ).decode().splitlines():
            status, path = line.split("\t", 1)
            changed[path] = status
        expected = {path: "A" for path in expected_added}
        expected.update({path: "M" for path in expected_modified})
        self.assertEqual(changed, expected)

        base = {
            path: (entry["mode"], entry["blob_sha"])
            for path, entry in self._tree_entries(self.phase["base_commit"]).items()
        }
        completed = self._tree_entries(COMPLETION_COMMIT)
        self.assertEqual(set(completed), set(base) | expected_added)
        self.assertEqual(len(completed), 641)
        for path, identity in base.items():
            if path not in expected_modified:
                self.assertEqual((completed[path]["mode"], completed[path]["blob_sha"]), identity, path)

    @staticmethod
    def _tree_entries(commit):
        result = {}
        for record in git("ls-tree", "-rlz", commit).split(b"\0"):
            if not record:
                continue
            metadata, raw_path = record.split(b"\t", 1)
            mode, kind, blob_sha, size = metadata.split()
            if kind == b"blob":
                result[raw_path.decode("utf-8")] = {
                    "mode": mode.decode("ascii"),
                    "blob_sha": blob_sha.decode("ascii"),
                    "size": int(size),
                }
        return result

    def test_source_anchors_are_unchanged(self):
        completed = self._tree_entries(COMPLETION_COMMIT)
        for anchor in self.contract["source_anchors"]:
            path = anchor["path"]
            self.assertEqual(completed[path]["blob_sha"], anchor["git_blob_sha"], path)
            self.assertEqual(completed[path]["size"], anchor["ordinary_blob_bytes"], path)
            self.assertEqual(completed[path]["mode"], anchor["mode"], path)

    def test_direct_import_consumers_are_exact(self):
        actual = {key: set() for key in self.contract["import_consumers"]}
        mapping = {
            (item["module"], item["symbol"]): key
            for key, item in self.contract["import_consumers"].items()
        }
        paths = [
            path
            for path in self._tree_entries(COMPLETION_COMMIT)
            if path.startswith("llm2vec/") and path.endswith(".py")
        ]
        for path in paths:
            text = git("show", f"{COMPLETION_COMMIT}:{path}").decode("utf-8")
            for module, symbol in absolute_imports(path, text=text):
                key = mapping.get((module, symbol))
                if key is not None:
                    actual[key].add(path)
        for key, expected in self.contract["import_consumers"].items():
            self.assertEqual(actual[key], set(expected["paths"]), key)
            self.assertEqual(len(actual[key]), expected["path_count"], key)
            self.assertEqual(path_hash(actual[key]), expected["sorted_path_list_sha256"], key)

    def test_exact_source_parse_finding_is_preserved(self):
        actual = []
        for path in self._tree_entries(COMPLETION_COMMIT):
            if not path.startswith("llm2vec/") or not path.endswith(".py"):
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    text = git("show", f"{COMPLETION_COMMIT}:{path}").decode("utf-8")
                    ast.parse(text, filename=path)
            except SyntaxError as exc:
                actual.append({"path": path, "line": exc.lineno})
        expected = [
            {"path": item["path"], "line": item["line"]}
            for item in self.contract["source_parse_findings"]
        ]
        self.assertEqual(actual, expected)
        self.assertTrue(all(not item["repair_performed"] for item in self.contract["source_parse_findings"]))

    def test_method_ast_anchors_and_shared_pooling(self):
        files = {
            "canonical": ("llm2vec/llm2vec/llm2vec.py", "LLM2Vec"),
            "causal_derivative": ("llm2vec/llm22vec/llm22vec.py", "LLM2Vec"),
            "openunlearning_adapter": ("llm2vec/llm22vec/openunlearn_wrapper.py", "LLM2Vec2CausalLM"),
        }
        method_names = {
            "init": "__init__",
            "model_class_selection": "_get_model_class",
            "from_pretrained": "from_pretrained",
            "forward": "forward",
            "generate": "generate",
        }
        for component, expected_methods in self.contract["method_ast_anchors"].items():
            path, class_name = files[component]
            for key, expected_hash in expected_methods.items():
                self.assertEqual(method_ast_hash(path, class_name, method_names[key]), expected_hash, component + ":" + key)
        pooling = self.contract["behavior_contracts"]["pooling"]
        canonical = method_ast_hash(files["canonical"][0], "LLM2Vec", "get_pooling")
        derivative = method_ast_hash(files["causal_derivative"][0], "LLM2Vec", "get_pooling")
        self.assertEqual(canonical, pooling["canonical_ast_sha256"])
        self.assertEqual(derivative, pooling["causal_derivative_ast_sha256"])
        self.assertEqual(canonical, derivative)
        self.assertTrue(pooling["implementations_ast_identical"])

    def test_public_imports_resolve_distinct_implementations(self):
        with loaded_legacy_packages() as loaded:
            self.assertEqual(loaded.canonical.LLM2Vec.__module__, "llm2vec.llm2vec")
            self.assertEqual(loaded.derivative.LLM2Vec.__module__, "llm22vec.llm22vec")
            self.assertIsNot(loaded.canonical.LLM2Vec, loaded.derivative.LLM2Vec)
            self.assertEqual(loaded.wrapper.LLM2Vec2CausalLM.__module__, "llm22vec.openunlearn_wrapper")

    def test_constructor_defaults_and_model_selection_are_distinct(self):
        with loaded_legacy_packages() as loaded:
            canonical = loaded.canonical.LLM2Vec
            derivative = loaded.derivative.LLM2Vec
            classes = loaded.classes
            shared = self.contract["behavior_contracts"]["shared_constructor_defaults"]
            for implementation in (canonical, derivative):
                signature = inspect.signature(implementation.__init__)
                for name, expected in shared.items():
                    self.assertEqual(signature.parameters[name].default, expected, name)
            self.assertTrue(inspect.signature(canonical.from_pretrained).parameters["enable_bidirectional"].default)
            self.assertFalse(inspect.signature(derivative.from_pretrained).parameters["enable_bidirectional"].default)
            self.assertIs(canonical._get_model_class("MistralConfig", False), classes.AutoModel)
            self.assertIs(derivative._get_model_class("MistralConfig", False), classes.AutoModelForCausalLM)
            for config_name, model_class in (
                ("MistralConfig", classes.MistralBiModel),
                ("LlamaConfig", classes.LlamaBiModel),
                ("GemmaConfig", classes.GemmaBiModel),
                ("Qwen2Config", classes.Qwen2BiModel),
            ):
                self.assertIs(canonical._get_model_class(config_name, True), model_class)
                self.assertIs(derivative._get_model_class(config_name, True), model_class)
            with self.assertRaises(ValueError):
                canonical._get_model_class("UnsupportedConfig", True)

    def test_from_pretrained_routes_without_loading_real_models(self):
        with loaded_legacy_packages() as loaded:
            classes = loaded.classes
            canonical = loaded.canonical.LLM2Vec.from_pretrained(
                "fixture-model", pooling_mode="last_token", max_length=64, fixture_flag="canonical"
            )
            derivative = loaded.derivative.LLM2Vec.from_pretrained(
                "fixture-model", pooling_mode="weighted_mean", max_length=32, fixture_flag="causal"
            )
            self.assertEqual(canonical.model.loader_name, "MistralBiModel")
            self.assertEqual(derivative.model.loader_name, "AutoModelForCausalLM")
            self.assertEqual((canonical.pooling_mode, canonical.max_length), ("last_token", 64))
            self.assertEqual((derivative.pooling_mode, derivative.max_length), ("weighted_mean", 32))
            self.assertEqual(classes.AutoTokenizer.calls[0], (("fixture-model",), {}))
            self.assertEqual(classes.AutoTokenizer.calls[1], (("fixture-model",), {"token": ""}))
            self.assertEqual(classes.AutoConfig.calls[0], (("fixture-model",), {}))
            self.assertEqual(classes.AutoConfig.calls[1], (("fixture-model",), {"token": ""}))
            self.assertEqual(classes.MistralBiModel.calls, [(('fixture-model',), {"fixture_flag": "canonical"})])
            self.assertEqual(
                classes.AutoModelForCausalLM.calls,
                [(('fixture-model',), {"token": "", "fixture_flag": "causal"})],
            )

    def test_forward_contracts_select_different_hidden_state_surfaces(self):
        with loaded_legacy_packages() as loaded:
            canonical_hidden = object()
            causal_hidden = object()

            class RecordingModel:
                def __init__(self):
                    self.config = types.SimpleNamespace(_name_or_path="fixture")
                    self.calls = []

                def __call__(self, **kwargs):
                    self.calls.append(kwargs)
                    return types.SimpleNamespace(
                        last_hidden_state=canonical_hidden,
                        hidden_states=[object(), causal_hidden],
                    )

            for implementation, expected_hidden, expected_extra in (
                (loaded.canonical.LLM2Vec, canonical_hidden, {}),
                (loaded.derivative.LLM2Vec, causal_hidden, {"output_hidden_states": True}),
            ):
                model = RecordingModel()
                instance = implementation(model, types.SimpleNamespace(padding_side="left"))
                pooled = object()
                capture = {}

                def record_pooling(features, hidden):
                    capture["features"] = features
                    capture["hidden"] = hidden
                    return pooled

                instance.get_pooling = record_pooling
                embed_mask = object()
                features = {"input_ids": "ids", "attention_mask": "mask", "embed_mask": embed_mask}
                self.assertIs(instance.forward(features), pooled)
                self.assertEqual(model.calls, [{"input_ids": "ids", "attention_mask": "mask", **expected_extra}])
                self.assertIs(features["embed_mask"], embed_mask)
                self.assertIs(capture["features"], features)
                self.assertIs(capture["hidden"], expected_hidden)

    def test_pooling_modes_execute_identically_under_array_fixture(self):
        hidden = ArrayTensor(
            [
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
                [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
            ]
        )
        attention = ArrayTensor([[0, 1, 1], [1, 1, 1]])
        input_ids = ArrayTensor([[0, 101, 2], [101, 4, 5]])
        expected = {
            "mean": np.asarray([[2.5, 25.0], [5.0, 50.0]]),
            "weighted_mean": np.asarray([[8.0 / 3.0, 80.0 / 3.0], [32.0 / 6.0, 320.0 / 6.0]]),
            "eos_token": np.asarray([[3.0, 30.0], [6.0, 60.0]]),
            "last_token": np.asarray([[3.0, 30.0], [6.0, 60.0]]),
            "bos_token": np.asarray([[2.0, 20.0], [4.0, 40.0]]),
        }
        with loaded_legacy_packages() as loaded:
            for implementation in (loaded.canonical.LLM2Vec, loaded.derivative.LLM2Vec):
                outputs = {}
                for mode in expected:
                    model = types.SimpleNamespace(config=types.SimpleNamespace(_name_or_path="fixture"))
                    tokenizer = types.SimpleNamespace(padding_side="left", bos_token_id=101)
                    instance = implementation(model, tokenizer, pooling_mode=mode, skip_instruction=False)
                    features = {"attention_mask": attention, "input_ids": input_ids}
                    outputs[mode] = instance.get_pooling(features, hidden)
                    np.testing.assert_allclose(outputs[mode].array, expected[mode], rtol=0, atol=1e-12)
                with self.assertRaises(ValueError):
                    instance.pooling_mode = "unsupported"
                    instance.get_pooling({"attention_mask": attention, "input_ids": input_ids}, hidden)
                for mode in expected:
                    other = loaded.derivative.LLM2Vec if implementation is loaded.canonical.LLM2Vec else loaded.canonical.LLM2Vec
                    peer = other(
                        types.SimpleNamespace(config=types.SimpleNamespace(_name_or_path="fixture")),
                        types.SimpleNamespace(padding_side="left", bos_token_id=101),
                        pooling_mode=mode,
                        skip_instruction=False,
                    )
                    np.testing.assert_allclose(
                        outputs[mode].array,
                        peer.get_pooling({"attention_mask": attention, "input_ids": input_ids}, hidden).array,
                        rtol=0,
                        atol=1e-12,
                    )

    def test_skip_instruction_replaces_attention_mask_before_pooling(self):
        with loaded_legacy_packages() as loaded:
            for implementation in (loaded.canonical.LLM2Vec, loaded.derivative.LLM2Vec):
                instance = implementation(
                    types.SimpleNamespace(config=types.SimpleNamespace(_name_or_path="fixture")),
                    types.SimpleNamespace(padding_side="left", bos_token_id=101),
                    pooling_mode="mean",
                    skip_instruction=True,
                )
                attention = ArrayTensor([[1, 1, 1]])
                embed = ArrayTensor([[0, 0, 1]])
                features = {"attention_mask": attention, "embed_mask": embed, "input_ids": ArrayTensor([[1, 2, 3]])}
                result = instance.get_pooling(features, ArrayTensor([[[1.0], [2.0], [9.0]]]))
                self.assertIs(features["attention_mask"], embed)
                np.testing.assert_allclose(result.array, [[9.0]], rtol=0, atol=0)

    def test_openunlearning_wrapper_delegation_contract(self):
        with loaded_legacy_packages() as loaded:
            class InnerModel:
                def __init__(self):
                    self.config = object()
                    self.generate_calls = []
                    self.forward_calls = []

                def generate(self, *args, **kwargs):
                    self.generate_calls.append((args, kwargs))
                    return "generated"

                def __call__(self, **kwargs):
                    self.forward_calls.append(kwargs)
                    return types.SimpleNamespace(loss="loss", logits="logits", past_key_values="not-copied")

            inner = InnerModel()
            tokenizer = object()
            wrapper = loaded.wrapper.LLM2Vec2CausalLM(types.SimpleNamespace(model=inner, tokenizer=tokenizer))
            self.assertIs(wrapper.inner_model, inner)
            self.assertIs(wrapper.tokenizer, tokenizer)
            self.assertIs(wrapper.config, inner.config)
            self.assertEqual(wrapper.generate("ids", temperature=0.2), "generated")
            self.assertEqual(inner.generate_calls, [(('ids',), {"temperature": 0.2})])
            result = wrapper.forward(
                input_ids="ids",
                attention_mask="mask",
                labels="labels",
                use_cache=False,
            )
            self.assertEqual(
                inner.forward_calls,
                [{"input_ids": "ids", "attention_mask": "mask", "labels": "labels", "return_dict": True}],
            )
            self.assertIsInstance(result, loaded.classes.CausalLMOutputWithPast)
            self.assertEqual((result.loss, result.logits), ("loss", "logits"))
            self.assertFalse(hasattr(result, "past_key_values"))

    def test_historical_entrypoint_keeps_explicit_causal_contract(self):
        entry = self.contract["historical_entrypoint_contract"]
        text = git("show", f"{COMPLETION_COMMIT}:{entry['path']}").decode("utf-8")
        tree = ast.parse(text, filename=entry["path"])
        imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module == entry["imports_module"]
            and any(alias.name == entry["imports_symbol"] for alias in node.names)
        ]
        self.assertEqual(len(imports), 1)
        model_arguments = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelArguments"
        )
        bidirectional = next(
            node
            for node in model_arguments.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "bidirectional"
        )
        default_keyword = next(keyword for keyword in bidirectional.value.keywords if keyword.arg == "default")
        self.assertIs(ast.literal_eval(default_keyword.value), entry["model_arguments_bidirectional_default"])
        loader_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "LLM2Vec"
        ]
        self.assertEqual(len(loader_calls), 1)
        self.assertIn("enable_bidirectional", {keyword.arg for keyword in loader_calls[0].keywords})

    def test_real_model_equivalence_and_source_movement_remain_blocked(self):
        tiers = self.contract["validation_tiers"]
        gates = self.contract["migration_gates"]
        self.assertEqual(tiers["real_dependency_import"], "not_run")
        self.assertEqual(tiers["real_model_initialization"], "not_run")
        self.assertEqual(tiers["real_model_forward_or_numerical_equivalence"], "not_run")
        self.assertEqual(gates["real_model_numerical_equivalence"], "not_validated")
        self.assertFalse(gates["llm22vec_removal_allowed"])
        self.assertFalse(gates["source_movement_allowed"])
        for key in (
            "source_movement_performed",
            "source_files_modified",
            "duplicate_package_removed",
            "dependency_consolidation_performed",
            "real_model_execution_performed",
            "dependency_download_performed",
            "scientific_or_provenance_content_changed",
        ):
            self.assertFalse(self.phase[key], key)


if __name__ == "__main__":
    unittest.main()
