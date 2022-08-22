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

o2_add_library(EventVisualisationView
               SOURCES src/MultiView.cxx
                       src/Initializer.cxx
                       src/EventManager.cxx
                       src/Options.cxx
                       src/EventManagerFrame.cxx
                       src/Screenshot.cxx
               PUBLIC_LINK_LIBRARIES ROOT::ASImage
                                     O2::EventVisualisationBase
                                     O2::EventVisualisationDetectors
                                     RapidJSON::RapidJSON)

o2_target_root_dictionary(EventVisualisationView
               HEADERS include/EventVisualisationView/MultiView.h
                       include/EventVisualisationView/Initializer.h
                       include/EventVisualisationView/Options.h
                       include/EventVisualisationView/EventManagerFrame.h
               LINKDEF src/EventVisualisationViewLinkDef.h)

o2_add_executable(eve
               SOURCES src/main.cxx
               PUBLIC_LINK_LIBRARIES O2::EventVisualisationView
               )
