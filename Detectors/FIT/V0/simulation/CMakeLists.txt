# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(V0Simulation
               SOURCES src/Detector.cxx
               PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat O2::V0Base
                                     O2::DataFormatsFITV0 ROOT::Physics)

o2_target_root_dictionary(V0Simulation HEADERS include/V0Simulation/Detector.h)
