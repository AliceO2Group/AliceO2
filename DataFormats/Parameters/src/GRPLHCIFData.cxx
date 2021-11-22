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
#include "DetectorsCommonDataFormats/NameConf.h"
#include "CommonConstants/PhysicsConstants.h"
#include <cmath>
#include <FairLogger.h>

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
  if (beam == beamDirection::BeamClockWise) {
    auto atomicNum = mZtoA.find(getAtomicNumberB1());
    if (atomicNum == mZtoA.end()) {
      LOG(FATAL) << "We don't know the Mass Number for Z = " << getAtomicNumberB1();
    } else {
      mBeamAZ[static_cast<int>(beam)] = (atomicNum->second << 16) + getAtomicNumberB1();
    }
  } else {
    auto atomicNum = mZtoA.find(getAtomicNumberB2());
    if (atomicNum == mZtoA.end()) {
      LOG(FATAL) << "We don't know the Mass Number for Z = " << getAtomicNumberB2();
    } else {
      mBeamAZ[static_cast<int>(beam)] = (atomicNum->second << 16) + getAtomicNumberB2();
    }
  }
}

//_______________________________________________
void GRPLHCIFData::setBeamAZ()
{

  // setting A and Z for both beams
  setBeamAZ(BeamClockWise);
  setBeamAZ(BeamAntiClockWise);
}

//_______________________________________________
float GRPLHCIFData::getSqrtS() const
{
  // get center of mass energy
  double e0 = getBeamEnergyPerNucleon(BeamClockWise);
  double e1 = getBeamEnergyPerNucleon(BeamAntiClockWise);
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
    bcNb.push_back(i = 0 ? 0 : (i / 10 + o2::constants::lhc::BunchOffsetsP2[beam]) % o2::constants::lhc::LHCMaxBunches);
  }
}

//_______________________________________________
GRPLHCIFData* GRPLHCIFData::loadFrom(const std::string& grpFileName)
{
  // load object from file
  auto fname = o2::base::NameConf::getGRPFileName(grpFileName);
  TFile flGRP(fname.c_str());
  if (flGRP.IsZombie()) {
    LOG(ERROR) << "Failed to open " << fname;
    throw std::runtime_error("Failed to open GRPLHCIF file");
  }
  auto grp = reinterpret_cast<o2::parameters::GRPLHCIFData*>(flGRP.GetObjectChecked(o2::base::NameConf::CCDBOBJECT.data(), Class()));
  if (!grp) {
    throw std::runtime_error(fmt::format("Failed to load GRPLHCIF object from {}", fname));
  }
  return grp;
}
