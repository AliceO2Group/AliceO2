// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#if !defined(SHARED_CONSTANTS_H)
#define SHARED_CONSTANTS_H

#include "types.h"

#ifndef CONSTANT
#ifdef __OPENCL__
#define CONSTANT constant
#else
#define CONSTANT static const
#endif
#endif

CONSTANT float CHARGE_THRESHOLD = 0.f;
CONSTANT float OUTER_CHARGE_THRESHOLD = 0.f;
CONSTANT float QTOT_THRESHOLD = 500.f;
CONSTANT int MIN_SPLIT_NUM = 1;

#endif //!defined(SHARED_CONSTANTS_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
