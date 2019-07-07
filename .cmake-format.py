# How wide to allow formatted cmake files
line_width = 80

# How many spaces to tab for indent
tab_size = 2

# If arglists are longer than this, break them always
max_subargs_per_line = 5

# If true, separate flow control names from their parentheses with a space
separate_ctrl_name_with_space = False

# If true, separate function names from parentheses with a space
separate_fn_name_with_space = False

# If a statement is wrapped to more than one line, than dangle the closing
# parenthesis on it's own line
dangle_parens = False

# What character to use for bulleted lists
bullet_char = '*'

# What character to use as punctuation after numerals in an enumerated list
enum_char = '.'

# What style line endings to use in the output.
line_ending = 'unix'

# Format command names consistently as 'lower' or 'upper' case
command_case = 'canonical'

# Format keywords consistently as 'lower' or 'upper' case
keyword_case = 'upper'

# Specify structure for custom cmake functions
# * = ZERO_OR_MORE
# + = ONE_OR_MORE
additional_commands = {
    "o2_add_executable": {
        "flags": ["IS_TEST", "IS_BENCHMARK", "NO_INSTALL"],
        "kwargs": {
            "SOURCES": '+',
            "PUBLIC_LINK_LIBRARIES": '*',
            "COMPONENT_NAME": '*',
            "EXEVARNAME": '*'
        }
    },
    "o2_add_header_only_library": {
        "kwargs": {
            "INCLUDE_DIRECTORIES": '*',
            "INTERFACE_LINK_LIBRARIES": '*',
        }
    },
    "o2_add_library": {
        "kwargs": {
            "SOURCES": '+',
            "PUBLIC_INCLUDE_DIRECTORIES": '*',
            "PUBLIC_LINK_LIBRARIES": '*',
            "PRIVATE_INCLUDE_DIRECTORIES": '*',
            "TARGETVARNAME": '*',
        }
    },
    "o2_target_root_dictionary": {
        "kwargs": {
            "LINKDEF": '+',
            "HEADERS": '*',
        }
    },
    "o2_target_man_page": {
        "kwargs": {
            "NAME": '+',
            "SECTION": '*',
        }
    },
    "add_root_dictionary": {
        "kwargs": {
            "LINKDEF": '+',
            "HEADERS": '*',
            "BASENAME": '*',
        }
    },
    "o2_data_file": {
        "kwargs": {
            "COPY": '+',
            "DESTINATION": '*',
        }
    },
    "o2_add_test_wrapper": {
        "flags": ["DONT_FAIL_ON_TIMEOUT", "NON_FATAL"],
        "kwargs": {
            "COMMAND": '*',
            "NO_BOOST_TEST": '*',
            "MAX_ATTEMPTS": '*',
            "TIMEOUT": '*',
            "NAME": '*',
            "WORKING_DIRECTORY": '*',
            "CONFIGURATIONS": '*',
            "COMMAND_LINE_ARGS": '*',
            "LABELS": '*',
            "ENVIRONMENT": '*',
        }
    },
    "o2_add_test": {
        "kwargs": {
            "INSTALL": '*',
            "NO_BOOST_TEST": '*',
            "NON_FATAL": '*',
            "COMPONENT_NAME": '*',
            "MAX_ATTEMPTS": '*',
            "TIMEOUT": '*',
            "WORKING_DIRECTORY": '*',
            "SOURCES": '*',
            "PUBLIC_LINK_LIBRARIES": '*',
            "COMMAND_LINE_ARGS": '*',
            "LABELS": '*',
            "ENVIRONMENT": '*',
        }
    },
    "o2_add_test_root_macro": {
        "flags": ["NON_FATAL", "LOAD_ONLY"],
        "kwargs": {
            "ENVIRONMENT": '*',
            "PUBLIC_LINK_LIBRARIES": '*',
            "LABELS": '*',
        }
    },
    "o2_name_target": {
        "kwargs": {
            "INCLUDE_DIRECTORIES": '*',
            "INTERFACE_LINK_LIBRARIES": '*',
        }
    },
    "find_package_handle_standard_args": {
        "flags": ["CONFIG_MODE"],
        "kwargs": {
            "DEFAULT_MSG": '*',
            "REQUIRED_VARS": '*',
            "VERSION_VAR": '*',
            "HANDLE_COMPONENTS": '*',
            "FAIL_MESSAGE": '*'
        }
    },
    "set_package_properties": {
        "kwargs": {
            "PROPERTIES": '*',
            "URL": '*',
            "TYPE": '*',
            "PURPOSE": '*'
        }
    }
}

# A list of command names which should always be wrapped
always_wrap = []

# Specify the order of wrapping algorithms during successive reflow attempts
algorithm_order = [0, 1, 2, 3, 4]

# If true, the argument lists which are known to be sortable will be sorted
# lexicographicall
autosort = False

# enable comment markup parsing and reflow
enable_markup = True

# If comment markup is enabled, don't reflow the first comment block in
# eachlistfile. Use this to preserve formatting of your
# copyright/licensestatements.
first_comment_is_literal = False

# If comment markup is enabled, don't reflow any comment block which matchesthis
# (regex) pattern. Default is `None` (disabled).
literal_comment_pattern = None

# Regular expression to match preformat fences in comments
# default=r'^\s*([`~]{3}[`~]*)(.*)$'
fence_pattern = '^\\s*([`~]{3}[`~]*)(.*)$'

# Regular expression to match rulers in comments
# default=r'^\s*[^\w\s]{3}.*[^\w\s]{3}$'
ruler_pattern = '^\\s*[^\\w\\s]{3}.*[^\\w\\s]{3}$'

# If true, emit the unicode byte-order mark (BOM) at the start of the file
emit_byteorder_mark = False

# If a comment line starts with at least this many consecutive hash characters,
# then don't lstrip() them off. This allows for lazy hash rulers where the first
# hash char is not separated by space
hashruler_min_length = 10

# If true, then insert a space between the first hash char and remaining hash
# chars in a hash ruler, and normalize it's length to fill the column
canonicalize_hashrulers = True

# Specify the encoding of the input file. Defaults to utf-8.
input_encoding = 'utf-8'

# Specify the encoding of the output file. Defaults to utf-8. Note that cmake
# only claims to support utf-8 so be careful when using anything else
output_encoding = 'utf-8'

# A dictionary containing any per-command configuration overrides. Currently
# only `command_case` is supported.
per_command = {}
