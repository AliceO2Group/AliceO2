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

o2_add_test_root_macro(drawCPVgeometry.C
                       PUBLIC_LINK_LIBRARIES O2::CPVSimulation
                                             O2::DetectorsPassive
                                             O2::PHOSSimulation
                       LABELS cpv)

o2_add_test_root_macro(plot_hit_cpv.C
                       PUBLIC_LINK_LIBRARIES FairRoot::Base O2::CPVSimulation
                       LABELS cpv COMPILE_ONLY)

o2_add_test_root_macro(plot_dig_cpv.C
                       PUBLIC_LINK_LIBRARIES FairRoot::Base O2::CPVSimulation
                       LABELS cpv)
		       
o2_add_test_root_macro(plot_clu_cpv.C
                       PUBLIC_LINK_LIBRARIES FairRoot::Base O2::CPVSimulation
                       LABELS cpv)
		       
