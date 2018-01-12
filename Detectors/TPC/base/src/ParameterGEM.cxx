// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ParameterGEM.cxx
/// \brief Implementation of the parameter class for the GEM stack
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCBase/ParameterGEM.h"

using namespace o2::TPC;

ParameterGEM::ParameterGEM() : mAbsoluteGain(), mCollectionEfficiency(), mExtractionEfficiency() {}

void ParameterGEM::setDefaultValues()
{
  mAbsoluteGain = { { 14.f, 8.f, 53.f, 240.f } };
  mCollectionEfficiency = { { 1.f, 0.2f, 0.25f, 1.f } };
  mExtractionEfficiency = { { 0.65f, 0.55f, 0.12f, 0.6f } };
}
