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

o2_add_test(dc_primitives
            SOURCES test/test_dc_primitives.cxx
            COMPONENT_NAME DataCompression
            LABELS utils)

o2_add_test(Fifo
            SOURCES test/test_Fifo.cxx
            COMPONENT_NAME DataCompression
            PUBLIC_LINK_LIBRARIES Threads::Threads
            LABELS utils)

o2_add_test(DataGenerator
            SOURCES test/test_DataGenerator.cxx
            COMPONENT_NAME DataCompression
            LABELS utils)

o2_add_test(HuffmanCodec
            SOURCES test/test_HuffmanCodec.cxx
            COMPONENT_NAME DataCompression
            PUBLIC_LINK_LIBRARIES O2::CommonUtils
            LABELS utils)

o2_add_test(DataDeflater
            SOURCES test/test_DataDeflater.cxx
            COMPONENT_NAME DataCompression
            LABELS utils)
