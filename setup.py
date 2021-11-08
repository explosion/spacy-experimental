import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

include_dirs = [np.get_include()]

COMPILER_DIRECTIVES = {
    "language_level": -3,
    "embedsignature": True,
    "annotation_typing": False,
}

MOD_NAMES = [
    "spacy_biaffine_parser.arc_predicter",
    "spacy_biaffine_parser.arc_labeler",
    "spacy_biaffine_parser.mst",
]

ext_modules = []
for name in MOD_NAMES:
    mod_path = name.replace(".", "/") + ".pyx"
    ext = Extension(
        name,
        [mod_path],
        language="c++",
        include_dirs=include_dirs,
        extra_compile_args=["-std=c++11"],
    )
    ext_modules.append(ext)

ext_modules = cythonize(ext_modules, compiler_directives=COMPILER_DIRECTIVES)

setup(
    name="biaffine-parser",
    ext_modules=ext_modules,
    zip_safe=False,
)
