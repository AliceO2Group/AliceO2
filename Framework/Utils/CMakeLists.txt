# Copyright CERN and copyright holders of ALICE O2. This software is
# distributed under the terms of the GNU General Public License v3 (GPL
# Version 3), copied verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/ for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(DPLUtils
        SOURCES
        src/Utils.cxx
        src/DPLBroadcaster.cxx
        src/DPLGatherer.cxx
        src/DPLMerger.cxx
        src/DPLRouter.cxx
        test/DPLBroadcasterMerger.cxx
        test/DPLOutputTest.cxx
        PUBLIC_LINK_LIBRARIES
        O2::Framework
        )


# FIXME: this one seg fault.
# o2_add_test(DPLBroadcasterMerger
#         SOURCES
#         src/Utils.cxx test/test_DPLBroadcasterMerger.cxx test/DPLBroadcasterMerger.cxx src/DPLMerger.cxx src/DPLBroadcaster.cxx
#         PUBLIC_LINK_LIBRARIES
#         O2::DPLUtils
#         COMPONENT_NAME DPLUtils)

# FIXME: this one never ends ? or is doing too much ? check that.
# o2_add_test(DPLOutput
#         SOURCES src/Utils.cxx test/test_DPLOutputTest.cxx test/DPLOutputTest.cxx
#         PUBLIC_LINK_LIBRARIES
#         O2::DPLUtils
#         COMPONENT_NAME DPLUtils
#         LABELS slow)

o2_add_test(RootTreeWriter
        SOURCES test/test_RootTreeWriter.cxx
        PUBLIC_LINK_LIBRARIES
        O2::DPLUtils
        COMPONENT_NAME DPLUtils)

o2_add_test(RootTreeWriterWorkflow
        SOURCES test/test_RootTreeWriterWorkflow.cxx
        PUBLIC_LINK_LIBRARIES
        O2::DPLUtils
        COMPONENT_NAME DPLUtils)
