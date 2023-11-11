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

/// \file  TPCScaler.cxx
/// \brief Definition of TPCScaler class
///
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#include "TPCCalibration/TPCScaler.h"
#include <TFile.h>
#include <TTree.h>
#include "Framework/Logger.h"

using namespace o2::tpc;

void TPCScaler::dumpToFile(const char* file, const char* name)
{
  TFile out(file, "RECREATE");
  TTree tree("TPCScaler", "TPCScaler");
  tree.Branch("TPCScaler", this);
  tree.Fill();
  out.WriteObject(&tree, name);
}

void TPCScaler::loadFromFile(const char* inpf, const char* name)
{
  TFile out(inpf, "READ");
  TTree* tree = (TTree*)out.Get(name);
  setFromTree(*tree);
}

void TPCScaler::setFromTree(TTree& tpcScalerTree)
{
  TPCScaler* scalerTmp = this;
  tpcScalerTree.SetBranchAddress("TPCScaler", &scalerTmp);
  const int entries = tpcScalerTree.GetEntries();
  if (entries > 0) {
    tpcScalerTree.GetEntry(0);
  } else {
    LOGP(error, "TPCScaler not found in input file");
  }
  tpcScalerTree.SetBranchAddress("TPCScaler", nullptr);
}

float TPCScaler::getMeanScaler(double timestamp, o2::tpc::Side side) const
{
  // index to data buffer
  const int idxData = (timestamp - mTimeStampMS) / mIntegrationTimeMS + 0.5;
  const int nVals = getNValuesIonDriftTime();
  const int nValues = getNValues(side);
  if ((nVals == 0) || (nVals > nValues)) {
    return -1;
    LOGP(error, "Empty data provided {}", nValues);
  }

  // clamp indices to min and max
  const int lastIdx = std::clamp(idxData, nVals, nValues);
  const int firstIdx = (lastIdx == nValues) ? (nValues - nVals) : std::clamp(idxData - nVals, 0, nValues);

  // sump up values from last ion drift time
  float sum = 0;
  for (int i = firstIdx; i < lastIdx; ++i) {
    sum += getScalers(i, side);
  }
  return (sum / nVals);
}
