# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

include_guard()

# message(FATAL_ERROR "there is a CMake module to do that!")

function(o2_dump_target_properties)
  set(target ${ARGV0})
  get_property(targetType TARGET ${target} PROPERTY TYPE)
  message(STATUS "--------------------------------------------------------")
  message(STATUS "Properties of target ${target} of type ${targetType}")
  message(STATUS)
  set(properties
      INTERFACE_COMPILE_DEFINITIONS
      INTERFACE_COMPILE_FEATURES
      INTERFACE_COMPILE_OPTIONS
      INTERFACE_INCLUDE_DIRECTORIES
      INTERFACE_LINK_DEPENDS
      INTERFACE_LINK_DIRECTORIES
      INTERFACE_LINK_LIBRARIES
      INTERFACE_LINK_OPTIONS
      INTERFACE_POSITION_INDEPENDENT_CODE
      INTERFACE_SOURCES
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
  if(NOT ${targetType} STREQUAL "INTERFACE_LIBRARY")
    list(APPEND properties
                ALIASED_TARGET
                ARCHIVE_OUTPUT_DIRECTORY_Debug
                ARCHIVE_OUTPUT_DIRECTORY
                ARCHIVE_OUTPUT_NAME_Debug
                ARCHIVE_OUTPUT_NAME
                AUTOGEN_BUILD_DIR
                AUTOGEN_ORIGIN_DEPENDS
                AUTOGEN_PARALLEL
                AUTOGEN_TARGET_DEPENDS
                AUTOMOC_COMPILER_PREDEFINES
                AUTOMOC_DEPEND_FILTERS
                AUTOMOC_EXECUTABLE
                AUTOMOC_MACRO_NAMES
                AUTOMOC_MOC_OPTIONS
                AUTOMOC
                AUTOUIC
                AUTOUIC_EXECUTABLE
                AUTOUIC_OPTIONS
                AUTOUIC_SEARCH_PATHS
                AUTORCC
                AUTORCC_EXECUTABLE
                AUTORCC_OPTIONS
                BINARY_DIR
                BUILD_RPATH
                BUILD_RPATH_USE_ORIGIN
                BUILD_WITH_INSTALL_NAME_DIR
                BUILD_WITH_INSTALL_RPATH
                BUNDLE_EXTENSION
                BUNDLE
                C_EXTENSIONS
                C_STANDARD
                C_STANDARD_REQUIRED
                COMMON_LANGUAGE_RUNTIME
                COMPATIBLE_INTERFACE_BOOL
                COMPATIBLE_INTERFACE_NUMBER_MAX
                COMPATIBLE_INTERFACE_NUMBER_MIN
                COMPATIBLE_INTERFACE_STRING
                COMPILE_DEFINITIONS
                COMPILE_FEATURES
                COMPILE_FLAGS
                COMPILE_OPTIONS
                COMPILE_PDB_NAME
                COMPILE_PDB_NAME_Debug
                COMPILE_PDB_OUTPUT_DIRECTORY
                COMPILE_PDB_OUTPUT_DIRECTORY_Debug
                Debug_OUTPUT_NAME
                Debug_POSTFIX
                CROSSCOMPILING_EMULATOR
                CUDA_PTX_COMPILATION
                CUDA_SEPARABLE_COMPILATION
                CUDA_RESOLVE_DEVICE_SYMBOLS
                CUDA_EXTENSIONS
                CUDA_STANDARD
                CUDA_STANDARD_REQUIRED
                CXX_EXTENSIONS
                CXX_STANDARD
                CXX_STANDARD_REQUIRED
                DEBUG_POSTFIX
                DEFINE_SYMBOL
                DEPLOYMENT_REMOTE_DIRECTORY
                DEPLOYMENT_ADDITIONAL_FILES
                DOTNET_TARGET_FRAMEWORK_VERSION
                EchoString
                ENABLE_EXPORTS
                EXCLUDE_FROM_ALL
                EXCLUDE_FROM_DEFAULT_BUILD_Debug
                EXCLUDE_FROM_DEFAULT_BUILD
                EXPORT_NAME
                EXPORT_PROPERTIES
                FOLDER
                Fortran_FORMAT
                Fortran_MODULE_DIRECTORY
                FRAMEWORK
                FRAMEWORK_VERSION
                GENERATOR_FILE_NAME
                GHS_INTEGRITY_APP
                GHS_NO_SOURCE_GROUP_FILE
                GNUtoMS
                HAS_CXX
                IMPLICIT_DEPENDS_INCLUDE_TRANSFORM
                IMPORTED_COMMON_LANGUAGE_RUNTIME
                IMPORTED_CONFIGURATIONS
                IMPORTED_GLOBAL
                IMPORTED_IMPLIB_Debug
                IMPORTED_IMPLIB
                IMPORTED_LIBNAME_Debug
                IMPORTED_LIBNAME
                IMPORTED_LINK_DEPENDENT_LIBRARIES_Debug
                IMPORTED_LINK_DEPENDENT_LIBRARIES
                IMPORTED_LINK_INTERFACE_LANGUAGES_Debug
                IMPORTED_LINK_INTERFACE_LANGUAGES
                IMPORTED_LINK_INTERFACE_LIBRARIES_Debug
                IMPORTED_LINK_INTERFACE_LIBRARIES
                IMPORTED_LINK_INTERFACE_MULTIPLICITY_Debug
                IMPORTED_LINK_INTERFACE_MULTIPLICITY
                IMPORTED_LOCATION_Debug
                IMPORTED_LOCATION
                IMPORTED_NO_SONAME_Debug
                IMPORTED_NO_SONAME
                IMPORTED_OBJECTS_Debug
                IMPORTED_OBJECTS
                IMPORTED
                IMPORTED_SONAME_Debug
                IMPORTED_SONAME
                IMPORT_PREFIX
                IMPORT_SUFFIX
                INCLUDE_DIRECTORIES
                INSTALL_NAME_DIR
                INSTALL_RPATH
                INSTALL_RPATH_USE_LINK_PATH
                INTERFACE_AUTOUIC_OPTIONS
                INTERFACE_COMPILE_DEFINITIONS
                INTERFACE_COMPILE_FEATURES
                INTERFACE_COMPILE_OPTIONS
                INTERFACE_INCLUDE_DIRECTORIES
                INTERFACE_LINK_DEPENDS
                INTERFACE_LINK_DIRECTORIES
                INTERFACE_LINK_LIBRARIES
                INTERFACE_LINK_OPTIONS
                INTERFACE_POSITION_INDEPENDENT_CODE
                INTERFACE_SOURCES
                INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                INTERPROCEDURAL_OPTIMIZATION_Debug
                INTERPROCEDURAL_OPTIMIZATION
                IOS_INSTALL_COMBINED
                JOB_POOL_COMPILE
                JOB_POOL_LINK
                LABELS
                <LANG>_CLANG_TIDY
                <LANG>_COMPILER_LAUNCHER
                <LANG>_CPPCHECK
                <LANG>_CPPLINT
                <LANG>_INCLUDE_WHAT_YOU_USE
                <LANG>_VISIBILITY_PRESET
                LIBRARY_OUTPUT_DIRECTORY_Debug
                LIBRARY_OUTPUT_DIRECTORY
                LIBRARY_OUTPUT_NAME_Debug
                LIBRARY_OUTPUT_NAME
                LINK_DEPENDS_NO_SHARED
                LINK_DEPENDS
                LINKER_LANGUAGE
                LINK_DIRECTORIES
                LINK_FLAGS_Debug
                LINK_FLAGS
                LINK_INTERFACE_LIBRARIES_Debug
                LINK_INTERFACE_LIBRARIES
                LINK_INTERFACE_MULTIPLICITY_Debug
                LINK_INTERFACE_MULTIPLICITY
                LINK_LIBRARIES
                LINK_OPTIONS
                LINK_SEARCH_END_STATIC
                LINK_SEARCH_START_STATIC
                LINK_WHAT_YOU_USE
                MACOSX_BUNDLE_INFO_PLIST
                MACOSX_BUNDLE
                MACOSX_FRAMEWORK_INFO_PLIST
                MACOSX_RPATH
                MANUALLY_ADDED_DEPENDENCIES
                MAP_IMPORTED_CONFIG_Debug
                NAME
                NO_SONAME
                NO_SYSTEM_FROM_IMPORTED
                OSX_ARCHITECTURES_Debug
                OSX_ARCHITECTURES
                OUTPUT_NAME_Debug
                OUTPUT_NAME
                PDB_NAME_Debug
                PDB_NAME
                PDB_OUTPUT_DIRECTORY_Debug
                PDB_OUTPUT_DIRECTORY
                POSITION_INDEPENDENT_CODE
                PREFIX
                PRIVATE_HEADER
                PROJECT_LABEL
                PUBLIC_HEADER
                RESOURCE
                RULE_LAUNCH_COMPILE
                RULE_LAUNCH_CUSTOM
                RULE_LAUNCH_LINK
                RUNTIME_OUTPUT_DIRECTORY_Debug
                RUNTIME_OUTPUT_DIRECTORY
                RUNTIME_OUTPUT_NAME_Debug
                RUNTIME_OUTPUT_NAME
                SKIP_BUILD_RPATH
                SOURCE_DIR
                SOURCES
                SOVERSION
                STATIC_LIBRARY_FLAGS_Debug
                STATIC_LIBRARY_FLAGS
                STATIC_LIBRARY_OPTIONS
                SUFFIX
                TYPE
                VERSION
                VISIBILITY_INLINES_HIDDEN
                WIN32_EXECUTABLE
                WINDOWS_EXPORT_ALL_SYMBOLS
                XCODE_EXPLICIT_FILE_TYPE
                XCODE_PRODUCT_TYPE
                XCODE_SCHEME_ADDRESS_SANITIZER
                XCODE_SCHEME_ADDRESS_SANITIZER_USE_AFTER_RETURN
                XCODE_SCHEME_ARGUMENTS
                XCODE_SCHEME_DISABLE_MAIN_THREAD_CHECKER
                XCODE_SCHEME_DYNAMIC_LIBRARY_LOADS
                XCODE_SCHEME_DYNAMIC_LINKER_API_USAGE
                XCODE_SCHEME_ENVIRONMENT
                XCODE_SCHEME_EXECUTABLE
                XCODE_SCHEME_GUARD_MALLOC
                XCODE_SCHEME_MAIN_THREAD_CHECKER_STOP
                XCODE_SCHEME_MALLOC_GUARD_EDGES
                XCODE_SCHEME_MALLOC_SCRIBBLE
                XCODE_SCHEME_MALLOC_STACK
                XCODE_SCHEME_THREAD_SANITIZER
                XCODE_SCHEME_THREAD_SANITIZER_STOP
                XCODE_SCHEME_UNDEFINED_BEHAVIOUR_SANITIZER
                XCODE_SCHEME_UNDEFINED_BEHAVIOUR_SANITIZER_STOP
                XCODE_SCHEME_ZOMBIE_OBJECTS
                XCTEST)

    get_property(imported TARGET ${target} PROPERTY IMPORTED)
    if(${imported})
      list(APPEND properties LOCATION_Debug LOCATION)

    endif()

  endif()
  foreach(prop IN LISTS properties)
    get_property(is_set TARGET ${target} PROPERTY ${prop} SET)
    if(is_set)
      get_property(value TARGET ${target} PROPERTY ${prop})
      message(STATUS "${prop} = ${value}")
      message(STATUS)
    endif()
  endforeach()
endfunction()
