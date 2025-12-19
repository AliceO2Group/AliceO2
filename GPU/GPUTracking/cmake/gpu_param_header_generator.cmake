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

file(READ "${GPU_PARAM_JSON}" JSON_CONTENT)
set(TMP_HEADER "${OUT_HEADER}.tmp")
file(WRITE "${TMP_HEADER}" "#ifndef GPUDEFPARAMETERSDEFAULTS_H\n#define GPUDEFPARAMETERSDEFAULTS_H\n\n")
file(APPEND "${TMP_HEADER}" "// This file is auto-generated from gpu_params.json. Do not edit directly.\n")
string(REPLACE "," ";" ARCH_LIST "${TARGET_ARCH_SHORT}")
file(APPEND "${TMP_HEADER}" "// Architectures: ${TARGET_ARCH_SHORT}\n\n")
file(APPEND "${TMP_HEADER}" "#if defined(GPUCA_GPUCODE) && !defined(GPUCA_GPUCODE_GENRTC) && !defined(GPUCA_GPUCODE_NO_LAUNCH_BOUNDS) // Avoid including for RTC generation besides normal include protection.\n\n")

# Types
set(TYPES CORE LB PAR)
foreach(ARCH IN LISTS ARCH_LIST)
    file(APPEND "${TMP_HEADER}" "#if defined(GPUCA_GPUTYPE_${ARCH})\n\n")
    foreach(TYPE IN LISTS TYPES)
        # Get all keys of this TYPE as a semicolon-separated list
        string(JSON n_params LENGTH "${JSON_CONTENT}" "${TYPE}")
        math(EXPR last "${n_params} - 1")
        foreach(i RANGE 0 ${last})
            string(JSON param_name MEMBER "${JSON_CONTENT}" "${TYPE}" "${i}")
            string(JSON n_archs LENGTH "${JSON_CONTENT}" "${TYPE}" "${param_name}")
            math(EXPR last_arch "${n_archs} - 1")

            foreach(iArch RANGE 0 ${last_arch})
                string(JSON arch MEMBER "${JSON_CONTENT}" "${TYPE}" "${param_name}" "${iArch}")
                if(arch STREQUAL "${ARCH}")
                    string(JSON param_values GET "${JSON_CONTENT}" "${TYPE}" "${param_name}" "${ARCH}")
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
                    file(APPEND "${TMP_HEADER}" "${MACRO_DEFINITION}\n")
                endif()
            endforeach()
        endforeach()
    endforeach()
    file(APPEND "${TMP_HEADER}" "\n#endif // GPUCA_GPUTYPE_${ARCH}\n\n")
endforeach()

file(APPEND "${TMP_HEADER}" "\n// Default parameters if not defined for the target architecture\n\n")
#Default parameters
foreach(TYPE IN LISTS TYPES)
    # Get all keys of this TYPE as a semicolon-separated list
    string(JSON n_params LENGTH "${JSON_CONTENT}" "${TYPE}")
    math(EXPR last "${n_params} - 1")
    foreach(i RANGE 0 ${last})
        string(JSON param_name MEMBER "${JSON_CONTENT}" "${TYPE}" "${i}")
        string(JSON param_values GET "${JSON_CONTENT}" "${TYPE}" "${param_name}" "default")
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
        file(APPEND "${TMP_HEADER}" "#ifndef ${MACRO_NAME}\n  ${MACRO_DEFINITION}\n#endif\n\n")
    endforeach()
endforeach()
file(APPEND "${TMP_HEADER}" "#endif // defined(GPUCA_GPUCODE) && !defined(GPUCA_GPUCODE_GENRTC) && !defined(GPUCA_GPUCODE_NO_LAUNCH_BOUNDS)\n\n")

#Defaults for non-LB parameters also for CPU fallback
file(APPEND "${TMP_HEADER}" "#ifndef GPUCA_GPUCODE_GENRTC //Defaults for non-LB parameters also for CPU fallback\n\n")    # Get all keys of this TYPE as a semicolon-separated list
string(JSON n_params LENGTH "${JSON_CONTENT}" "PAR")
math(EXPR last "${n_params} - 1")
foreach(i RANGE 0 ${last})
    string(JSON param_name MEMBER "${JSON_CONTENT}" "PAR" "${i}")
    string(JSON param_values GET "${JSON_CONTENT}" "PAR" "${param_name}" "default_cpu")
    set(MACRO_NAME "GPUCA_PAR_${param_name}")
    set(vals "${param_values}")
    string(REGEX REPLACE "^\\[ *" "" vals "${vals}")
    string(REGEX REPLACE " *\\]$" "" vals "${vals}")
    string(REGEX REPLACE "\"" "" vals "${vals}")
    set(MACRO_DEFINITION "#define ${MACRO_NAME} ${vals}")
    file(APPEND "${TMP_HEADER}" "#ifndef ${MACRO_NAME}\n  ${MACRO_DEFINITION}\n#endif\n\n")
endforeach()
file(APPEND "${TMP_HEADER}" "\n#endif // GPUCA_GPUCODE_GENRTC\n")

file(APPEND "${TMP_HEADER}" "\n#endif // GPUDEFPARAMETERSDEFAULTS_H\n")
file(RENAME "${TMP_HEADER}" "${OUT_HEADER}")
message(STATUS "Generated ${OUT_HEADER}")
