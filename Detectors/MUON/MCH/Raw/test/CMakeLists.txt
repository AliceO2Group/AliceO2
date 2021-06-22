# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# closure tests, thus depending on both Encoder and Decoder

o2_add_test(closure-codec
                SOURCES testClosureCoDec.cxx
                COMPONENT_NAME mchraw
                LABELS "muon;mch;raw"
                PUBLIC_LINK_LIBRARIES O2::MCHRawEncoderPayload O2::MCHRawDecoder)

o2_add_test(closure-codec-digit
                SOURCES testClosureCoDecDigit.cxx
                COMPONENT_NAME mchraw
                LABELS "muon;mch;raw"
                PUBLIC_LINK_LIBRARIES O2::MCHRawEncoderDigit O2::MCHRawDecoder)
