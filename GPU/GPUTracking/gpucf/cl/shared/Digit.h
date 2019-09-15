// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#if !defined(SHARED_DIGIT_H)
#    define  SHARED_DIGIT_H

#include "tpc.h"


typedef struct PackedDigit_s
{
    float charge;
    timestamp time;
    pad_t pad;
    row_t row;
} PackedDigit;

#define PACKED_DIGIT_SIZE 8


#endif //!defined(SHARED_DIGIT_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
