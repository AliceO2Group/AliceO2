// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#pragma once

#include <cmath>

static constexpr float FEQ_EPSILON_SMALL = 0.5;
static constexpr float FEQ_EPSILON_BIG = 1.0;

static inline bool floatEq(float f1, float f2, float epsilon = FEQ_EPSILON_SMALL)
{
  return std::abs(f1 - f2) <= epsilon;
}

bool almostEqual(float, float);

// vim: set ts=4 sw=4 sts=4 expandtab:
