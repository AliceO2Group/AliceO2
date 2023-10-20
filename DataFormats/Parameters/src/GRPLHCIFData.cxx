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

/// \file GRPLHCIFData.cxx
/// \brief Implementation of the LHC InterFace data

#include "DataFormatsParameters/GRPLHCIFData.h"
#include "CommonUtils/NameConf.h"
#include "CommonConstants/PhysicsConstants.h"
#include <ctime>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <Framework/Logger.h>

using namespace o2::parameters;
using namespace o2::constants::physics;
using namespace o2::constants::lhc;

const std::unordered_map<unsigned int, unsigned int> GRPLHCIFData::mZtoA =
  {
    {1, 1},
    {82, 208}};

//_______________________________________________
void GRPLHCIFData::setBeamAZ(beamDirection beam)
{
  // set both A and Z of the beam in direction 'beam'
  if (beam == beamDirection::BeamC) {
    auto atomicNum = mZtoA.find(getAtomicNumberB1());
    if (atomicNum != mZtoA.end()) {
      mBeamAZ[static_cast<int>(beam)] = (atomicNum->second << 16) + getAtomicNumberB1();
    }
  } else {
    auto atomicNum = mZtoA.find(getAtomicNumberB2());
    if (atomicNum != mZtoA.end()) {
      mBeamAZ[static_cast<int>(beam)] = (atomicNum->second << 16) + getAtomicNumberB2();
    }
  }
}

//_______________________________________________
void GRPLHCIFData::setBeamAZ()
{

  // setting A and Z for both beams
  setBeamAZ(BeamC);
  setBeamAZ(BeamA);
}

//_______________________________________________
float GRPLHCIFData::getSqrtS() const
{
  // get center of mass energy
  double e0 = getBeamEnergyPerNucleonInGeV(BeamC);
  double e1 = getBeamEnergyPerNucleonInGeV(BeamA);
  if (e0 <= MassProton || e1 <= MassProton) {
    return 0.f;
  }
  double beta0 = 1. - MassProton * MassProton / (e0 * e0);
  double beta1 = 1. - MassProton * MassProton / (e1 * e1);
  beta0 = beta0 > 0 ? sqrt(beta0) : 0.;
  beta1 = beta1 > 0 ? sqrt(beta1) : 0.;
  double ss = 2. * (MassProton * MassProton + e0 * e1 * (1. + beta0 * beta1 * cos(getCrossingAngle())));
  return ss > 0. ? sqrt(ss) : 0.;
}

//_________________________________________________________________

void GRPLHCIFData::translateBucketsToBCNumbers(std::vector<int32_t>& bcNb, std::vector<int32_t>& buckets, int beam)
{
  // to translate the vector of bucket numbers to BC numbers
  for (auto i : buckets) {
    if (i) {
      bcNb.push_back((i / 10 + o2::constants::lhc::BunchOffsetsP2[beam]) % o2::constants::lhc::LHCMaxBunches);
    }
  }
}

//_______________________________________________
GRPLHCIFData* GRPLHCIFData::loadFrom(const std::string& grpFileName)
{
  // load object from file
  auto fname = o2::base::NameConf::getGRPLHCIFFileName(grpFileName);
  TFile flGRP(fname.c_str());
  if (flGRP.IsZombie()) {
    LOG(error) << "Failed to open " << fname;
    throw std::runtime_error("Failed to open GRPLHCIF file");
  }
  auto grp = reinterpret_cast<o2::parameters::GRPLHCIFData*>(flGRP.GetObjectChecked(o2::base::NameConf::CCDBOBJECT.data(), Class()));
  if (!grp) {
    throw std::runtime_error(fmt::format("Failed to load GRPLHCIF object from {}", fname));
  }
  return grp;
}

//_______________________________________________
void GRPLHCIFData::print() const
{
  // print itself
  auto timeStr = [](long t) -> std::string {
    if (t) {
      std::time_t temp = t / 1000;
      std::tm* tt = std::gmtime(&temp);
      std::stringstream ss;
      ss << std::put_time(tt, "%d/%m/%y %H:%M:%S") << " UTC";
      return ss.str();
    }
    return {"        N / A        "};
  };

  printf("%s: Fill              : %d\n", timeStr(mFillNumber.first).c_str(), mFillNumber.second);
  printf("%s: Injection scheme  : %s\n", timeStr(mInjectionScheme.first).c_str(), mInjectionScheme.second.c_str());
  printf("%s: Beam energy per Z : %d\n", timeStr(mBeamEnergyPerZ.first).c_str(), mBeamEnergyPerZ.second);
  printf("%s: A beam1 (clock)   : %d\n", timeStr(mAtomicNumberB1.first).c_str(), mAtomicNumberB1.second);
  printf("%s: A beam2 (a-clock) : %d\n", timeStr(mAtomicNumberB2.first).c_str(), mAtomicNumberB2.second);
  printf("%s: Bunch filling\n", timeStr(mBunchFilling.first).c_str());
  if (mBunchFilling.first > 0) {
    mBunchFilling.second.print();
  }
}
