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

set(O2_TARGETPCMMAP_TARGET
    ""
    CACHE INTERNAL
          "target/PCM map (target)")
set(O2_TARGETPCMMAP_PCM
    ""
    CACHE INTERNAL
          "target/PCM map (pcm)")

function(set_root_pcm_dependencies)
  foreach(
    target
    pcm
    IN
    ZIP_LISTS
    O2_TARGETPCMMAP_TARGET
    O2_TARGETPCMMAP_PCM)
    if(NOT
       pcm
       STREQUAL
       "")
      # message(STATUS "target ${target} has pcm ${pcm}")
      list(
        APPEND
        target_pcms_${target}
        ${pcm})
    endif()
  endforeach()

  foreach(
    target
    pcm
    IN
    ZIP_LISTS
    O2_TARGETPCMMAP_TARGET
    O2_TARGETPCMMAP_PCM)
    if(NOT
       pcm
       STREQUAL
       "")
      unset(pcm_dep_list)
      get_target_property(
        dep_targets
        ${target}
        LINK_LIBRARIES)
      foreach(
        dep_target
        IN
        LISTS dep_targets)
        if(dep_target
           MATCHES
           "^O2::")
          string(
            REPLACE "O2::"
                    ""
                    dep_target
                    ${dep_target})
          o2_name_target(${dep_target} NAME dep_target)
          # message(STATUS "target ${target} depends on ${dep_target}")
          foreach(
            dep_pcm
            IN
            LISTS target_pcms_${dep_target})
            # message(STATUS "${pcm} depends on ${dep_pcm}")
            list(
              APPEND
              pcm_dep_list
              ${dep_pcm})
          endforeach()
        endif()
      endforeach()
      set(list_pcm_deps_${target}
          "${pcm_dep_list}"
          CACHE INTERNAL
                "List of pcm dependencies for ${target}")
    endif()
  endforeach()
endfunction()
