# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/Framework
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

o2_add_library(FrameworkFoundation
               SOURCES src/RuntimeError.cxx
               TARGETVARNAME targetName
               PUBLIC_LINK_LIBRARIES O2::FrameworkFoundation3rdparty
              )
set(DPL_ENABLE_BACKTRACE ON CACHE BOOL "Enable backtrace on o2::framework::runtime_error")

if (DPL_ENABLE_BACKTRACE)
target_compile_definitions(${targetName} PUBLIC -DDPL_ENABLE_BACKTRACE)
endif()


o2_add_test(test_FunctionalHelpers NAME test_FrameworkFoundation_test_FunctionalHelpers
            COMPONENT_NAME FrameworkFoundation
            SOURCES test/test_FunctionalHelpers.cxx
            PUBLIC_LINK_LIBRARIES O2::FrameworkFoundation)

o2_add_test(test_Traits NAME test_FrameworkFoundation_test_Traits
            COMPONENT_NAME FrameworkFoundation
            SOURCES test/test_Traits.cxx
            PUBLIC_LINK_LIBRARIES O2::FrameworkFoundation)

o2_add_test(test_StructToTuple NAME test_FrameworkFoundation_StructToTuple
            COMPONENT_NAME FrameworkFoundation
            SOURCES test/test_StructToTuple.cxx
            PUBLIC_LINK_LIBRARIES O2::FrameworkFoundation)

o2_add_test(test_CompilerBuiltins NAME test_FrameworkFoundation_CompilerBuiltins
            COMPONENT_NAME FrameworkFoundation
            SOURCES test/test_CompilerBuiltins.cxx
            PUBLIC_LINK_LIBRARIES O2::FrameworkFoundation)

o2_add_test(test_Signpost NAME test_FrameworkFoundation_Signpost
            COMPONENT_NAME FrameworkFoundation
            SOURCES test/test_Signpost.cxx
            PUBLIC_LINK_LIBRARIES O2::FrameworkFoundation)

o2_add_test(test_RuntimeError NAME test_FrameworkFoundation_RuntimeError
            COMPONENT_NAME FrameworkFoundation
            SOURCES test/test_RuntimeError.cxx
            PUBLIC_LINK_LIBRARIES O2::FrameworkFoundation)

add_subdirectory(3rdparty)
