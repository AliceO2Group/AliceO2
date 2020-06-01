# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
add_compile_options(-O0 -g -fPIC)

add_subdirectory(workflow)

o2_add_test(itsmft
            PUBLIC_LINK_LIBRARIES O2::CTFWorkflow
                                  O2::ITSMFTReconstruction
	                          O2::DataFormatsITSMFT
            SOURCES test/test_ctf_io_itsmft.cxx
            COMPONENT_NAME ctf
            LABELS ctf)

o2_add_test(tpc
            PUBLIC_LINK_LIBRARIES O2::CTFWorkflow
                                  O2::TPCReconstruction
	                          O2::DataFormatsTPC
            SOURCES test/test_ctf_io_tpc.cxx
            COMPONENT_NAME ctf
            LABELS ctf)
