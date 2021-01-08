# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# Given GCC 7.3 does not provide std::filesystem we use Boost instead
# Drop this once we move to GCC 8.2+
o2_add_library(FrameworkGUISupport
               SOURCES src/Plugin.cxx
                       src/FrameworkGUIDebugger.cxx
                       src/FrameworkGUIDevicesGraph.cxx
                       src/FrameworkGUIDeviceInspector.cxx
                       src/FrameworkGUIDataRelayerUsage.cxx
                       src/PaletteHelpers.cxx
               PRIVATE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/src
               PUBLIC_LINK_LIBRARIES O2::Framework AliceO2::DebugGUI)

if (NOT APPLE)
set (DEBUG_GUI_TESTS_WORKFLOW
     CustomGUIGL
     CustomGUISokol
     SimpleTracksED
     )
endif()

foreach(w
        ${DEBUG_GUI_TESTS_WORKFLOW}
        )
  o2_add_test(${w} NAME test_Framework_test_${w}
              SOURCES test/test_${w}.cxx
              COMPONENT_NAME Framework
              LABELS framework workflow
              PUBLIC_LINK_LIBRARIES O2::Framework AliceO2::DebugGUI
              TIMEOUT 30
              NO_BOOST_TEST
              COMMAND_LINE_ARGS ${DPL_WORKFLOW_TESTS_EXTRA_OPTIONS} --run --shm-segment-size 20000000)
endforeach()

# TODO: investigate the problem with the two unit tests, maybe setup of the CI
# environment assertion fired X11: The DISPLAY environment variable is missing
# glfw-3.2.1/src/window.c:579: glfwGetFramebufferSize: Assertion `window !=
# ((void *)0)' failed.
if(NOT APPLE)
set_property(TEST test_Framework_test_SimpleTracksED PROPERTY DISABLED TRUE)
set_property(TEST test_Framework_test_CustomGUIGL PROPERTY DISABLED TRUE)
set_property(TEST test_Framework_test_CustomGUISokol PROPERTY DISABLED TRUE)
endif()
