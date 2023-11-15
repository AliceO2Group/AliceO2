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

o2_add_test(
  Filtering
  SOURCES testFilter.cxx
  LABELS mid
  COMPONENT_NAME mid
  PUBLIC_LINK_LIBRARIES O2::MIDFiltering O2::MIDBase)

# if(benchmark_FOUND) o2_add_executable( tracker COMPONENT_NAME mid SOURCES
# bench_Tracker.cxx IS_BENCHMARK PUBLIC_LINK_LIBRARIES O2::MIDTracking
# benchmark::benchmark) endif()
