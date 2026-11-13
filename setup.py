"""Package the maintained training, visualization and physics libraries."""
from setuptools import find_namespace_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parent
SCOPE = runpy.run_path(str(ROOT / "tools/packaging/public_source_scope.py"))
BUILD_ROOT = ROOT / "artifacts" / "build"
METADATA_ROOT = ROOT / "artifacts" / "build_metadata"
METADATA_ROOT.mkdir(parents=True, exist_ok=True)


class PublicBuildPy(build_py):
    def find_package_modules(self, package, package_dir):
        return [entry for entry in super().find_package_modules(package, package_dir)
                if SCOPE["allows_wheel"](package.replace(".", "/") + "/" + entry[1] + ".py")]

    def find_data_files(self, package, src_dir):
        return [name for name in super().find_data_files(package, src_dir)
                if SCOPE["allows_wheel"](package.replace(".", "/") + "/"
                                         + Path(name).relative_to(src_dir).as_posix())]

    def run(self):
        # Old broad builds must not leave excluded files in a later wheel.
        # Only clean the generated build directory, never source directories.
        build_root = Path(self.build_lib).resolve()
        if not build_root.is_relative_to(BUILD_ROOT):
            raise RuntimeError("Public build output must be under artifacts/build/")
        if build_root.exists():
            for path in build_root.rglob("*"):
                if path.is_file():
                    if not path.resolve().is_relative_to(build_root):
                        raise RuntimeError(f"Build output contains an external symlink: {path}")
                    if not SCOPE["allows_wheel"](path.relative_to(build_root).as_posix()):
                        path.unlink()
        super().run()


class PublicSdist(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, SCOPE["source_files"]())


def packages_under(root: str, *excluded: str) -> list[str]:
    # Search source roots only; never walk datasets or artifacts at repo root.
    return [root] + [
        f"{root}.{name}" for name in find_namespace_packages(
            where=root, exclude=["__pycache__", "*.__pycache__", *excluded])
    ]


setup(
    packages=(packages_under("core", "resources*")
              + packages_under("training", "conf", "conf.*", "archive", "archive.*")
              + packages_under("scripts")),
    cmdclass={"build_py": PublicBuildPy, "sdist": PublicSdist},
    options={"build": {"build_base": str(BUILD_ROOT.relative_to(ROOT))},
             "egg_info": {"egg_base": str(METADATA_ROOT.relative_to(ROOT))}},
)
