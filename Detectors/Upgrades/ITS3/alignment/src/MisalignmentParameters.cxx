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

/// \file MisalignmentParameter.cxx
/// \brief Implementation of the MisalignmentParameter class

#include "ITS3Align/MisalignmentParameters.h"
#include "Framework/Logger.h"

#include "TFile.h"

#include <memory>

ClassImp(o2::its3::align::MisalignmentParameters);

namespace o2::its3::align
{

MisalignmentParameters::MisalignmentParameters()
{
  SetName("MisalignmentParameters");
  SetTitle("ITS3 MisalignmentParameters");
}

bool MisalignmentParameters::store(const std::string& file) const
{
  std::unique_ptr<TFile> fOut(TFile::Open(file.c_str(), "RECREATE"));
  if (fOut == nullptr || fOut->IsZombie()) {
    LOGP(info, "Unable to save misalignment parameters");
    return false;
  }
  fOut->WriteObjectAny(this, "o2::its3::align::MisalignmentParameters", "ccdb_object");
  return true;
}

MisalignmentParameters* MisalignmentParameters::load(const std::string& file)
{
  std::unique_ptr<TFile> fIn(TFile::Open(file.c_str(), "READ"));
  auto p = fIn->Get<MisalignmentParameters>("ccdb_object");
  if (p == nullptr) {
    LOGP(fatal, "Unable to load parameters from file!");
  }
  return p;
}

void MisalignmentParameters::printParams(unsigned int detID) const
{
  LOGP(info, "Parameters for ID={}:", detID);
  LOGP(info, " - Global Trans: X={} Y={} Z={}", getGloTransX(detID), getGloTransY(detID), getGloTransZ(detID));
  LOGP(info, " - Global Rots: X={} Y={} Z={}", getGloRotX(detID), getGloRotY(detID), getGloRotZ(detID));
  if (constants::detID::isDetITS3(detID)) {
    auto sensorID = constants::detID::getSensorID(detID);
    LOGP(info, " - Legendre Pol X:");
    getLegendreCoeffX(sensorID).Print();
    LOGP(info, " - Legendre Pol Y:");
    getLegendreCoeffY(sensorID).Print();
    LOGP(info, " - Legendre Pol Z:");
    getLegendreCoeffZ(sensorID).Print();
  }
}

void MisalignmentParameters::printLegendreParams(unsigned int sensorID) const
{
  LOGP(info, " - Legendre Pol X:");
  getLegendreCoeffX(sensorID).Print();
  LOGP(info, " - Legendre Pol Y:");
  getLegendreCoeffY(sensorID).Print();
  LOGP(info, " - Legendre Pol Z:");
  getLegendreCoeffZ(sensorID).Print();
}

} // namespace o2::its3::align
