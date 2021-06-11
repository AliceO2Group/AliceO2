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
#
# o2_find_dependencies_from_alibuild(basedir ...)  will locate, all our external
# dependencies from a typical alibuild installation.
#
# * basedir : the top directory of the alibuild installation,
#   $HOME/alice/sw/[osname] typically
# * LABEL (`latest` by default) : the label of the installation
#
# In the example below, basedir would be ~/alice/sw/osx_x86-64 and LABEL could
# be either `label` or `label-cmake-reorg-o2`
#
# cmake-format: off
#
# ~/alice/sw
# ├── BUILD
# ├── INSTALLROOT
# ├── MIRROR
# ├── MODULES
# ├── SOURCES
# ├── SPECS
# ├── TARS
# ├── osx_x86-64
# │   ├── FairMQ
# │   │   ├── latest -> v1.4.2-1
# │   │   ├── latest-cmake-reorg-o2 -> v1.4.2-1
# │   │   └── v1.4.2-1
# │   ├── GEANT4
# │   │   ├── latest -> v10.4.2-1
# │   │   ├── latest-cmake-reorg-o2 -> v10.4.2-1
# │   │   └── v10.4.2-1
# │   ├── ROOT
# │   │   ├── latest -> v6-16-00-1
# │   │   ├── latest-cmake-reorg-o2 -> v6-16-00-1
# │   │   └── v6-16-00-1
# │   ├── RapidJSON
# │   │   ├── 091de040edb3355dcf2f4a18c425aec51b906f08-1
# │   │   ├── latest -> 091de040edb3355dcf2f4a18c425aec51b906f08-1
# │   │   └── latest-cmake-reorg-o2 -> 091de040edb3355dcf2f4a18c425aec51b906f08-1
# 
# cmake-format: on

function(o2_find_dependencies_from_alibuild)

  if(DEPENDENCIES_FROM_ALIBUILD_DONE)
    return()
  endif()

  cmake_parse_arguments(PARSE_ARGV
                        1
                        A
                        ""
                        "LABEL;QUIET"
                        "")

  set(basedir ${ARGV0})
  if(A_LABEL AND NOT "${A_LABEL}" STREQUAL "")
    set(label ${A_LABEL})
  else()
    set(label latest)
  endif()

  macro(protected_set_root var)
    if(DEFINED ${var}_ROOT)
      if(NOT A_QUIET)
        message(STATUS "${var}_ROOT already defined. Not autodetecting it.")
      endif()
    else()
      if(${ARGC} LESS 2)
        # if we have only one argument, use value=var
        set(value ${var})
      else()
        set(value ${ARGV1})
      endif()
      set(dir ${basedir}/${value}/${label})
      if(IS_DIRECTORY ${dir})
        set(${var}_ROOT ${dir} CACHE STRING "top dir for ${pkg}")
        if(NOT A_QUIET)
          message(STATUS "Detected ${var}_ROOT=${dir}")
        endif()
      else()
        if(NOT A_QUIET)
          message(
            STATUS "Could not detect ${var}_ROOT from alibuild installation")
        endif()
      endif()
      unset(dir)
      unset(value)
    endif()
  endmacro()

  macro(protected_set_dir var)
    if(DEFINED ${var}_DIR)
      if(NOT A_QUIET)
        message(STATUS "${var}_DIR already defined. Not autodetecting it.")
      endif()
    else()
      if(${ARGC} LESS 2)
        # if we have only one argument, use value=var
        set(value ${var})
        set(pkg ${var})
      elseif(${ARGC} LESS 3)
        set(value ${ARGV1})
        set(pkg ${var})
      else()
        set(value ${ARGV1})
        set(pkg ${ARGV2})
      endif()
      set(dir ${basedir}/${pkg}/${label}/lib/cmake/${value})
      if(IS_DIRECTORY ${dir})
        set(${var}_DIR
            ${dir}
            CACHE STRING "location of cmake config for ${pkg}")
        if(NOT A_QUIET)
          message(STATUS "Detected ${var}_DIR=${dir}")
        endif()
      else()
        if(NOT A_QUIET)
          message(
            STATUS "Could not detect ${var}_DIR from alibuild installation")
        endif()
      endif()
      unset(dir)
    endif()
  endmacro()

  protected_set_root(DDS)
  protected_set_root(protobuf)
  protected_set_root(Common Common-O2)
  protected_set_root(Configuration)
  protected_set_root(Monitoring)
  protected_set_root(FairMQ)
  protected_set_root(FairLogger)
  protected_set_root(FairRoot)
  protected_set_root(InfoLogger libInfoLogger)
  protected_set_root(BOOST boost)
  protected_set_root(ROOT)
  protected_set_root(RapidJSON)
  protected_set_root(ms_gsl)
  protected_set_root(ZeroMQ)

  protected_set_root(pythia)
  protected_set_root(pythia6)
  protected_set_root(Geant3 GEANT3)
  protected_set_root(Geant4 GEANT4)
  protected_set_root(Geant4VMC GEANT4_VMC)
  protected_set_root(VGM vgm)
  protected_set_root(HepMC HepMC3)

  protected_set_dir(arrow)
  protected_set_dir(benchmark benchmark googlebenchmark)
  protected_set_dir(Vc)

  protected_set_root(cub)

  find_program(brew_CMD brew)
  if(brew_CMD)
    execute_process(COMMAND ${brew_CMD} --PREFIX glfw
                    OUTPUT_VARIABLE result
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    get_filename_component(glfw ${result}/lib/cmake ABSOLUTE)
    if(EXISTS ${glfw})
      if(NOT A_QUIET)
        message(STATUS "Detected GLFW_DIR=${glfw}")
      endif()
      set(GLFW_DIR ${glfw} PARENT_SCOPE)
    endif()
  endif()

  set(
    DEPENDENCIES_FROM_ALIBUILD_DONE
    TRUE
    CACHE BOOL
          "whether the dependencies where found from an alibuild installation")
endfunction()
