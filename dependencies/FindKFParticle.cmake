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

# add KFParticle::KFParticle as library to targets depending on KFParticle

if (NOT DEFINED KFParticle_DIR)
        set(KFParticle_DIR "$ENV{KFPARTICLE_ROOT}")
endif()

find_package(KFParticle NO_MODULE)
if (NOT KFParticle_FOUND)
        return()
endif()

add_library(KFParticle IMPORTED INTERFACE)

set_target_properties(KFParticle
        PROPERTIES
        INTERFACE_LINK_LIBRARIES "${KFPARTICLE_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${KFPARTICLE_INCLUDE_DIR}")

#Promote the imported target to global visibility(so we can alias it)
set_target_properties(KFParticle PROPERTIES IMPORTED_GLOBAL TRUE)
add_library(KFParticle::KFParticle ALIAS KFParticle)
