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

add_subdirectory(base)
add_subdirectory(simulation)
add_subdirectory(macros)
add_subdirectory(tracking)
add_subdirectory(workflow)
add_subdirectory(calibration)
add_subdirectory(condition)
add_subdirectory(assessment)
add_subdirectory(alignment)

o2_data_file(COPY data DESTINATION Detectors/Geometry/MFT/)
