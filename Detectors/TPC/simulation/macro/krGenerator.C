// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file krGenerator
/// \brief This macro instantiates the compiled 83mKr TPC calibration
///        generator (o2::eventgen::GeneratorKrDecay), for use with
///        o2-sim -g external --extGenFile krGenerator.C --extGenFunc krGenerator
/// \author Ankur Yadav <ankur.yadav@cern.ch>

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "FairGenerator.h"
#include "TPCSimulation/GeneratorKrDecay.h"
#endif

FairGenerator* krGenerator()
{
  auto gen = new o2::eventgen::GeneratorKrDecay();
  return gen;
}
