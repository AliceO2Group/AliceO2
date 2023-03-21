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
add_compile_options(-O0 -g -fPIC)
add_subdirectory(workflow)
add_subdirectory(utils)

o2_add_test(itsmft
            PUBLIC_LINK_LIBRARIES O2::ITSMFTReconstruction
                                  O2::DataFormatsITSMFT
            SOURCES test/test_ctf_io_itsmft.cxx
            COMPONENT_NAME ctf
            LABELS ctf)

o2_add_test(tpc
            PUBLIC_LINK_LIBRARIES O2::TPCReconstruction
                                  O2::DataFormatsTPC
            SOURCES test/test_ctf_io_tpc.cxx
            COMPONENT_NAME ctf
            LABELS ctf)

o2_add_test(ft0
            PUBLIC_LINK_LIBRARIES O2::FT0Reconstruction
                                  O2::DataFormatsFT0
            SOURCES test/test_ctf_io_ft0.cxx
            COMPONENT_NAME ctf
            LABELS ctf)

o2_add_test(fv0
            PUBLIC_LINK_LIBRARIES O2::FV0Reconstruction
                                  O2::DataFormatsFV0
            SOURCES test/test_ctf_io_fv0.cxx
            COMPONENT_NAME ctf
            LABELS ctf)

o2_add_test(fdd
            PUBLIC_LINK_LIBRARIES O2::FDDReconstruction
                                  O2::DataFormatsFDD
            SOURCES test/test_ctf_io_fdd.cxx
            COMPONENT_NAME ctf
            LABELS ctf)

o2_add_test(tof
            PUBLIC_LINK_LIBRARIES O2::TOFBase
                                  O2::TOFReconstruction
                                  O2::DataFormatsTOF
            SOURCES test/test_ctf_io_tof.cxx
            COMPONENT_NAME ctf
            LABELS ctf)

o2_add_test(mid
            PUBLIC_LINK_LIBRARIES O2::DataFormatsMID
                                  O2::MIDCTF
            SOURCES test/test_ctf_io_mid.cxx
            COMPONENT_NAME ctf
            LABELS ctf;muon;mid)

o2_add_test(mch
            PUBLIC_LINK_LIBRARIES O2::MCHCTF
            SOURCES test/test_ctf_io_mch.cxx
            COMPONENT_NAME ctf
            LABELS ctf;muon;mch)

o2_add_test(emcal
            PUBLIC_LINK_LIBRARIES O2::DataFormatsEMCAL
                                  O2::EMCALReconstruction
            SOURCES test/test_ctf_io_emcal.cxx
            COMPONENT_NAME ctf
            LABELS ctf)

o2_add_test(phos
            PUBLIC_LINK_LIBRARIES O2::DataFormatsPHOS
                                  O2::PHOSReconstruction
            SOURCES test/test_ctf_io_phos.cxx
            COMPONENT_NAME ctf
            LABELS ctf)

o2_add_test(cpv
            PUBLIC_LINK_LIBRARIES O2::DataFormatsCPV
                                  O2::CPVReconstruction
            SOURCES test/test_ctf_io_cpv.cxx
            COMPONENT_NAME ctf
            LABELS ctf)

o2_add_test(zdc
            PUBLIC_LINK_LIBRARIES O2::DataFormatsZDC
                                  O2::ZDCReconstruction
            SOURCES test/test_ctf_io_zdc.cxx
            COMPONENT_NAME ctf
            LABELS ctf)

o2_add_test(trd
            PUBLIC_LINK_LIBRARIES O2::CTFWorkflow
                                  O2::DataFormatsTRD
                                  O2::TRDReconstruction
            SOURCES test/test_ctf_io_trd.cxx
            COMPONENT_NAME ctf
            LABELS ctf)

o2_add_test(hmpid
            PUBLIC_LINK_LIBRARIES O2::CTFWorkflow
                                  O2::DataFormatsHMP
                                  O2::HMPIDReconstruction
            SOURCES test/test_ctf_io_hmpid.cxx
            COMPONENT_NAME ctf
            LABELS ctf)

o2_add_test(ctp
            PUBLIC_LINK_LIBRARIES O2::CTPWorkflow
                                  O2::DataFormatsCTP
                                  O2::CTPReconstruction
            SOURCES test/test_ctf_io_ctp.cxx
            COMPONENT_NAME ctf
            LABELS ctf)
