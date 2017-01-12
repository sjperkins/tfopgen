Tensorflow Custom Operator Code Outline Generator
=================================================

.. image:: https://travis-ci.org/sjperkins/tfopgen.svg?branch=master
    :target: https://travis-ci.org/sjperkins/tfopgen

Writing a tensorflow operator requires writing fair amounts of
boilerplate C++ and CUDA code. This script generates code for the CPU
and GPU version of a tensorflow operator. More specifically, given
tensorflow ``inputs``, ``outputs`` and ``attribute``\ s, it generates:

-  C++ Header file that defines the operator class, templated on Device.
-  C++ Header file that defines the CPU implementation of the operator.
-  C++ Source file with Shape Function, REGISTER\_OP and
   REGISTER\_KERNEL\_BUILDER constructs.
-  Cuda Header that defines the GPU implementation of the operator,
   including a CUDA kernel.
-  Cuda Source file with GPU REGISTER\_KERNEL\_BUILDER's for the
   operator.
-  python unit test case, which constructs random input data, and calls
   the operator.
-  Makefile for compiling the operator into a shared library, using g++
   and nvcc.

Requirements
------------

A tensorflow installation, required for building the operator.

.. code:: bash

    pip install tensorflow


Installation
------------

.. code:: bash

    pip install tfopgen

Usage
-----

The user should provide a YAML configuration file defining the operator:

-  inputs and optionally, their shapes.
-  outputs and optionally, their outputs.
-  polymorphic type attributes.
-  other attributes.
-  documentation.

For example, we can define the outline for a ``ComplexPhase`` operator in the ``complex_phase.yml`` file.

.. code:: yaml

    ---
    project: astronomy
    library: fourier
    name: ComplexPhase
    type_attrs:
      - "FT: {float, double} = DT_FLOAT"
      - "CT: {complex64, complex128} = DT_COMPLEX64"
    inputs:
      - ["uvw: FT", [null, null, 3]]   # (ntime, nbl, 3)
      - ["frequency: FT", [null]]      # (nchan, )
      - ["lm: FT", [null, 2]]          # (nsrc, 2)
    outputs:
      - ["complex_phase: CT", [null, null, null, null]]
    doc: >
      Given tensors
        (1) of (U, V, W) baseline coordinates with shape (ntime, nbl, 3)
        (2) of (L, M) sky coordinates with shape (nsrc, 2)
        (3) of frequencies,
      compute the complex phase with shape (nsrc, ntime, nbl, nchan)

We can then run:

.. code:: bash

    $ tfopgen complex_phase.yml

to create the following directory structure and files:

.. code:: bash

    $ tree fourier/
    fourier/
    ├── complex_phase_op_cpu.cpp
    ├── complex_phase_op_cpu.h
    ├── complex_phase_op_gpu.cu
    ├── complex_phase_op_gpu.cuh
    ├── complex_phase_op.h
    ├── Makefile
    └── test_complex_phase.py

The ``project`` and ``library`` options specify C++ namespaces within
which the operator is created. Additionally, the Makefile will create a
``fourier.so`` shared library that can be loaded with ``tf.load_op_library('fourier.so')``.

Any polymorphic type attributes should be supplied. The generator will
template the operators on type attributes. It will also generate
concrete permutations of REGISTER\_KERNEL\_BUILDER for both the CPU and
GPU op using the actual types supplied in the type attributes (float,
double, complex64 and complex128) below:

.. code:: yaml

    type_attrs:
      - "FT: {float, double} = DT_FLOAT"
      - "CT: {complex64, complex128} = DT_COMPLEX64"


The operator inputs and their optional shapes should be specified as a
list containing a string defining the ``.Input`` directive, and a list
describing the shape of the input tensor. A ``null`` value in the shape
will be translated into a python ``None``. If concrete dimensions are specified,
corresponding checks will be generated in the Shape Function associated with the
operator.

.. code:: yaml

    inputs:
      - ["uvw: FT", [null, null, 3]]   # (ntime, nbl, 3)
      - ["frequency: FT", [null]]      # (nchan, )
      - ["lm: FT", [null, 2]]          # (nsrc, 2)

The operator outputs should similarly defined.

.. code:: yaml

    outputs:
      - ["complex_phase: CT", [null, null, null, null]]

Given these inputs and outputs, CPU and GPU operators are created with
named variables corresponding to the inputs and outputs. Additionally, a
CUDA kernel with the given inputs and outputs is created, as well as a
shape function checking the rank and dimensions of the supplied inputs.


Other attributes may be specified (and will be output in the
REGISTER\_OP) directive, but are not catered for automatically by the
generator code as the range of attribute behaviour is complex.

.. code:: yaml

    op_other_attrs:
        - "iterations: int32 >= 2",

Finally operator documentation may also be supplied.

.. code:: yaml

    doc: >
      Given tensors
        (1) of (U, V, W) baseline coordinates with shape (ntime, nbl, 3)
        (2) of (L, M) sky coordinates with shape (nsrc, 2)
        (3) of frequencies,
      compute the complex phase with shape (nsrc, ntime, nbl, nchan)
