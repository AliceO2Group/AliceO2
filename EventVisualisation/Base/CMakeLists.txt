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

o2_add_library(EventVisualisationBase
               SOURCES src/ConfigurationManager.cxx
                       src/DataSourceOnline.cxx
                       src/DataSourceOffline.cxx
                       src/GeometryManager.cxx
                       src/FileWatcher.cxx
                       src/DirectoryLoader.cxx

        PUBLIC_LINK_LIBRARIES ROOT::Eve
                                     O2::CCDB
                                     O2::EventVisualisationDataConverter
                                     O2::DetectorsBase
)
