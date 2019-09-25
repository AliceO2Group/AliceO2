# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_test(Geometry
            SOURCES testGeometry.cxx
            PUBLIC_LINK_LIBRARIES O2::MIDSimulation
            COMPONENT_NAME mid
            LABELS mid muon)

o2_add_test(Simulation
            SOURCES testSimulation.cxx
            PUBLIC_LINK_LIBRARIES O2::MIDSimulation O2::MIDTracking
            COMPONENT_NAME mid
            LABELS mid long muon)
