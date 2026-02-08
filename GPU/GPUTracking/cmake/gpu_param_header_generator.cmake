# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

# file gpu_param_header_generator.cmake
# author Gabriele Cimador

function(generate_gpu_param_header GPU_PARAM_JSON_FILES ARCH_LIST OUT_HEADER OUT_HEADER_DEVICE)
    list(FIND ARCH_LIST "ALL" do_all_architectures)
    list(FIND ARCH_LIST "AUTO" do_auto_architectures)
    if(do_all_architectures GREATER -1 OR do_auto_architectures GREATER -1)
        if(do_auto_architectures GREATER -1)
            detect_gpu_arch("AUTO")
            list(REMOVE_ITEM ARCH_LIST "AUTO")
        else()
            detect_gpu_arch("ALL")
        endif()
        list(APPEND ARCH_LIST ${TARGET_ARCH})
    endif()

    # Types
    set(TYPES CORE LB PAR)
    set(ARCH_LIST_EXT "${ARCH_LIST};default;default_cpu")
    # Per architecture definitions
    set(JSON_ARCHITECTURES)

    set(GPU_PARAM_JSON_N_FILES 0)
    foreach(GPU_PARAM_JSON_FILE IN LISTS GPU_PARAM_JSON_FILES)
        file(READ "${GPU_PARAM_JSON_FILE}" JSON_CONTENT)
        foreach(TYPE IN LISTS TYPES)
            string(JSON n_params LENGTH "${JSON_CONTENT}" "${TYPE}")
            math(EXPR last "${n_params} - 1")
            foreach(i RANGE 0 ${last})
                string(JSON param_name MEMBER "${JSON_CONTENT}" "${TYPE}" "${i}")
                string(JSON n_archs LENGTH "${JSON_CONTENT}" "${TYPE}" "${param_name}")
                if(n_archs GREATER 0)
                math(EXPR last_arch "${n_archs} - 1")
                    foreach(iArch RANGE 0 ${last_arch})
                        string(JSON arch MEMBER "${JSON_CONTENT}" "${TYPE}" "${param_name}" "${iArch}")
                        if(arch STREQUAL "default_cpu" AND NOT TYPE STREQUAL "PAR")
                            message(FATAL_ERROR "Bogus entry ${param_name} for ${arch}")
                        endif()
                        if(arch MATCHES ^default AND GPU_PARAM_JSON_N_FILES GREATER 0)
                            message(FATAL_ERROR "Defaults must be provided in first parameter file")
                        endif()
                        if(do_all_architectures GREATER -1)
                            if(NOT arch MATCHES ^default)
                                list(APPEND JSON_ARCHITECTURES "${arch}")
                            endif()
                            set(list_idx 0)
                        else()
                            list(FIND ARCH_LIST_EXT "${arch}" list_idx)
                        endif()
                        if(list_idx GREATER -1)
                            string(JSON param_values GET "${JSON_CONTENT}" "${TYPE}" "${param_name}" "${arch}")
                            if(TYPE STREQUAL "LB")
                                set(MACRO_NAME "GPUCA_LB_${param_name}")
                            elseif(TYPE STREQUAL "PAR")
                                set(MACRO_NAME "GPUCA_PAR_${param_name}")
                            else()
                                set(MACRO_NAME "GPUCA_${param_name}")
                            endif()
                            set(vals "${param_values}")
                            string(REGEX REPLACE "^\\[ *" "" vals "${vals}")
                            string(REGEX REPLACE " *\\]$" "" vals "${vals}")
                            string(REGEX REPLACE "\"" "" vals "${vals}")
                            set(MACRO_DEFINITION "#define ${MACRO_NAME} ${vals}")
                            if(arch MATCHES ^default)
                                # fallback defaults are wrapped in #ifndef
                                string(APPEND generate_gpu_param_header_OUTPUT_TMP_${arch} "#ifndef ${MACRO_NAME}\n  ${MACRO_DEFINITION}\n#endif\n\n")
                            else()
                                string(APPEND generate_gpu_param_header_OUTPUT_TMP_${arch} "${MACRO_DEFINITION}\n")
                            endif()
                        endif()
                    endforeach()
                endif()
            endforeach()
        endforeach()
        math(EXPR GPU_PARAM_JSON_N_FILES "${GPU_PARAM_JSON_N_FILES} + 1")
    endforeach()

    list(REMOVE_DUPLICATES JSON_ARCHITECTURES)
    list(SORT JSON_ARCHITECTURES)
    if(ARGC GREATER 4)
        set(${ARGV4} "${JSON_ARCHITECTURES}" PARENT_SCOPE)
    endif()
    if(do_all_architectures GREATER -1)
        list(REMOVE_ITEM ARCH_LIST "ALL")
        list(APPEND ARCH_LIST ${JSON_ARCHITECTURES})
    endif()
    list(REMOVE_DUPLICATES ARCH_LIST)
    list(SORT ARCH_LIST)

    get_filename_component(DEVICE_HEADER_FILE "${OUT_HEADER_DEVICE}" NAME)

    set(TMP_HEADER "#ifndef GPUDEFPARAMETERSDEFAULTS_H\n#define GPUDEFPARAMETERSDEFAULTS_H\n\n")
    set(TMP_HEADER_DEVICE "#ifndef GPUDEFPARAMETERSDEFAULTSDEVICE_H\n#define GPUDEFPARAMETERSDEFAULTSDEVICE_H\n\n")
    string(APPEND TMP_HEADER "// This file is auto-generated from gpu_params.json. Do not edit directly.\n")
    string(APPEND TMP_HEADER_DEVICE "// This file is auto-generated from gpu_params.json. Do not edit directly.\n")
    string(APPEND TMP_HEADER_DEVICE "// Architectures: ${TARGET_ARCH}\n\n")
    string(APPEND TMP_HEADER "#if defined(GPUCA_GPUCODE) && !defined(GPUCA_GPUCODE_GENRTC) && !defined(GPUCA_GPUCODE_NO_LAUNCH_BOUNDS) // Avoid including for RTC generation besides normal include protection.\n\n")
    string(APPEND TMP_HEADER "#include \"${DEVICE_HEADER_FILE}\"\n")

    string(APPEND TMP_HEADER_DEVICE "#if 0\n")
    foreach(ARCH IN LISTS ARCH_LIST)
        string(APPEND TMP_HEADER_DEVICE "\n#elif defined(GPUCA_GPUTYPE_${ARCH})\n")
        string(APPEND TMP_HEADER_DEVICE ${generate_gpu_param_header_OUTPUT_TMP_${ARCH}})
    endforeach()
    string(APPEND TMP_HEADER_DEVICE "#else\n#error GPU TYPE NOT SET\n#endif\n")

    # Default parameters
    string(APPEND TMP_HEADER "\n// Default parameters if not defined for the target architecture\n\n")
    string(APPEND TMP_HEADER ${generate_gpu_param_header_OUTPUT_TMP_default})
    string(APPEND TMP_HEADER "#endif // defined(GPUCA_GPUCODE) && !defined(GPUCA_GPUCODE_GENRTC) && !defined(GPUCA_GPUCODE_NO_LAUNCH_BOUNDS)\n\n")

    # CPU fallback
    string(APPEND TMP_HEADER "#ifndef GPUCA_GPUCODE_GENRTC // Defaults for non-LB parameters also for CPU fallback\n\n")
    string(APPEND TMP_HEADER ${generate_gpu_param_header_OUTPUT_TMP_default_cpu})
    string(APPEND TMP_HEADER "\n#endif // GPUCA_GPUCODE_GENRTC\n")

    string(APPEND TMP_HEADER "\n#endif // GPUDEFPARAMETERSDEFAULTS_H\n")
    string(APPEND TMP_HEADER_DEVICE "\n#endif // GPUDEFPARAMETERSDEFAULTSDEVICE_H\n")
    file(GENERATE OUTPUT "${OUT_HEADER}" CONTENT "${TMP_HEADER}")
    file(GENERATE OUTPUT "${OUT_HEADER_DEVICE}" CONTENT "${TMP_HEADER_DEVICE}")
    message(STATUS "Generated ${OUT_HEADER} and ${OUT_HEADER_DEVICE}")
endfunction()
