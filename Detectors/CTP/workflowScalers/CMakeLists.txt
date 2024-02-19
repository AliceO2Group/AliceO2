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
o2_add_executable(
                  proxy
                  COMPONENT_NAME ctp
                  SOURCES src/ctp-proxy.cxx
                  PUBLIC_LINK_LIBRARIES O2::DCStestWorkflow
                  O2::DataFormatsCTP)
o2_add_executable(
                  qc-proxy
                  COMPONENT_NAME ctp
                  SOURCES src/ctp-qc-proxy.cxx
                  PUBLIC_LINK_LIBRARIES O2::DCStestWorkflow
                  O2::DataFormatsCTP)
o2_add_executable(
                  ccdb-orbit
                  COMPONENT_NAME ctp
                  SOURCES src/ctp-ccdb-orbit.cxx
                  PUBLIC_LINK_LIBRARIES O2::DataFormatsCTP
                  Boost::program_options)
