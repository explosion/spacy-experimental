from setuptools import setup
from setuptools_rust import RustExtension

setup(
    name="biaffine-parser",
    rust_extensions=[
        RustExtension("biaffine_parser.mst", "rust/mst/Cargo.toml")
    ],
    #packages=["biaffine_parser"],
    zip_safe=False,
)
