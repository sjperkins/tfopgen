import itertools
import os

from tfopgen.util import (parse_args, load_config, parse_inout,
                          parse_attr_type, camel_to_snake_case,
                          header_guard, jinja_env_factory)


def make_template_kwargs(op_name, py_op_name,
                         project, library, op_inputs, op_outputs,
                         op_type_attrs, op_other_attrs, op_doc):
    """
    Creates a dictionary suitable for rendering the jinja2 templates in this package
    """

    NB = '_namespace_begin'
    NE = '_namespace_stop'

    type_constraints = [tuple(t for t in a.np_types) for a in op_type_attrs]

    # Permute the type constraints
    op_tf_type_perms = itertools.product(*(a.tf_types for a in op_type_attrs))
    op_tf_type_perms = [list(p) for p in op_tf_type_perms]

    op_np_type_perms = itertools.product(*(a.np_types for a in op_type_attrs))
    op_np_type_perms = [list(p) for p in op_np_type_perms]

    # Create dictionary with variables required for creating the templates
    template_kwargs = {
        # Names
        'op_name': op_name,
        'py_op_name': py_op_name,
        'project': project,
        'library': library,
        'kernel_name': ''.join([library, '_', py_op_name]),

        # Operator inputs, outputs, attributes and documentation
        'op_inputs': op_inputs,
        'op_outputs': op_outputs,
        'op_type_attrs': op_type_attrs,
        'op_other_attrs': op_other_attrs,
        'op_tf_type_perms': op_tf_type_perms,
        'op_np_type_perms': op_np_type_perms,
        'type_constraints': type_constraints,
        'op_doc': op_doc,

        # Filenames
        'main_header_file': ''.join([py_op_name, '_op.h']),
        'cpp_header_file': ''.join([py_op_name, '_op_cpu.h']),
        'cpp_source_file': ''.join([py_op_name, '_op_cpu.cpp']),
        'cuda_header_file': ''.join([py_op_name, '_op_gpu.cuh']),
        'cuda_source_file': ''.join([py_op_name, '_op_gpu.cu']),
        'python_test_file': ''.join(['test_', py_op_name, '.py']),
        'makefile': 'Makefile',
        'shared_library': ''.join([library, '.so']),

        # C++ namespace
        'project_namespace_start': ''.join([project, NB]).upper(),
        'project_namespace_stop': ''.join([project, NE]).upper(),
        'op_namespace_start': ''.join([project, '_', py_op_name, NB]).upper(),
        'op_namespace_stop': ''.join([project, '_', py_op_name, NE]).upper(),
    }

    template_kwargs.update({
        # C++ header guards
        'main_header_guard': header_guard(library, template_kwargs['main_header_file']),
        'cpp_header_guard': header_guard(library, template_kwargs['cpp_header_file']),
        'cuda_header_guard': header_guard(library, template_kwargs['cuda_header_file']),
    })

    return template_kwargs


def run(args):
    """
    Runs the operator generator

    Arguments:
        args: list
            List of command line arguments stripped of the program name.
            sys.argv[1:] is appropriate in most cases.
    """
    args = parse_args(args)
    cfg = load_config(args.config)

    try:
        op_name = cfg['name']
        library = cfg['library']
        project = cfg['project']
    except KeyError as e:
        raise ValueError("Key '{}' was not present in '{}'"
            .format(e.message, args.config))

    op_inputs = cfg.get('inputs', [])
    op_outputs = cfg.get('outputs', [])
    op_type_attrs = cfg.get('type_attrs', [])
    op_other_attrs = cfg.get('other_attrs', [])
    op_doc = cfg.get('doc', "Documentation")

    # Parse input ops
    op_inputs = [parse_inout(i, s) for i, s in op_inputs]

    # Parse output ops
    op_outputs = [parse_inout(o, s) for o, s in op_outputs]

    # Parse type constrained attrs
    op_type_attrs = [parse_attr_type(a) for a in op_type_attrs]

    # Snake case python version of the operator
    py_op_name = camel_to_snake_case(op_name)

    template_path = os.path.join(os.path.dirname(__file__), 'templates')
    jinja_env = jinja_env_factory(template_path)

    # Create library directory if it does not exist
    if not os.path.exists(library):
        os.makedirs(library)

    # Create dictionary for rendering jinja2 templates
    kwargs = make_template_kwargs(op_name, py_op_name,
                                  project, library, op_inputs, op_outputs,
                                  op_type_attrs, op_other_attrs, op_doc)

    def render(template, output):
        """ Hook to render template file to output """
        with open(os.path.join(library, kwargs[output]), 'w') as f:
            header_template = jinja_env.get_template(template)
            f.write(header_template.render(**kwargs))

    render('main_header.j2', 'main_header_file')
    render('cpp_header.j2', 'cpp_header_file')
    render('cpp_source.j2', 'cpp_source_file')
    render('cuda_header.j2', 'cuda_header_file')
    render('cuda_source.j2', 'cuda_source_file')
    render('test_source.j2', 'python_test_file')
    render('Makefile.j2', 'makefile')
