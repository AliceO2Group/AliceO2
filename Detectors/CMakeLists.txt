# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

add_subdirectory(Base)

add_subdirectory(Passive) # must be first as some detector's macros use it

add_subdirectory(PHOS)
add_subdirectory(CPV)
add_subdirectory(EMCAL)
add_subdirectory(FIT)
add_subdirectory(HMPID)
add_subdirectory(TOF)
add_subdirectory(ZDC)

add_subdirectory(ITSMFT)
add_subdirectory(TRD)

add_subdirectory(MUON)

add_subdirectory(TPC)

add_subdirectory(GlobalTracking)
add_subdirectory(GlobalTrackingWorkflow)

if(BUILD_SIMULATION)
  add_subdirectory(gconfig)
endif()

o2_data_file(COPY Geometry gconfig DESTINATION Detectors)
