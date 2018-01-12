// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ParameterDetector.cxx
/// \brief Implementation of the parameter class for the detector
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCBase/ParameterDetector.h"

using namespace o2::TPC;

ParameterDetector::ParameterDetector() : mTPClength(0.f), mPadCapacitance(0.f) {}

void ParameterDetector::setDefaultValues()
{
  mTPClength = 250.f;
  mPadCapacitance = 0.1f;
}
