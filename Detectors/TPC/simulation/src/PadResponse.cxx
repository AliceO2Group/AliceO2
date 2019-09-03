// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PadResponse.cxx
/// \brief Implementation of the Pad Response
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/PadResponse.h"

#include "TPCBase/Mapper.h"

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include "FairLogger.h"

using namespace o2::tpc;

PadResponse::PadResponse()
  : mIROC(),
    mOROC12(),
    mOROC3()
{
  mIROC = std::make_unique<TGraph2D>();
  mOROC12 = std::make_unique<TGraph2D>();
  mOROC3 = std::make_unique<TGraph2D>();

  importPRF("PRF_IROC.dat", mIROC);
  importPRF("PRF_OROC1-2.dat", mOROC12);
  importPRF("PRF_OROC3.dat", mOROC3);
}

bool PadResponse::importPRF(std::string file, std::unique_ptr<TGraph2D>& grPRF) const
{
  std::string inputDir;
  const char* aliceO2env = std::getenv("O2_ROOT");
  if (aliceO2env)
    inputDir = aliceO2env;
  inputDir += "/share/Detectors/TPC/files/";

  float x, y, normalizedPadResponse;
  int i = 0;
  std::ifstream prfFile(inputDir + file, std::ifstream::in);
  if (!prfFile) {
    LOG(FATAL) << "tpc::PadResponse - Input file '" << inputDir + file << "' does not exist! No PRF loaded!";
    return false;
  }
  for (std::string line; std::getline(prfFile, line);) {
    std::istringstream is(line);
    while (is >> x >> y >> normalizedPadResponse) {
      grPRF->SetPoint(i++, x, y, normalizedPadResponse);
    }
  }
  return true;
}

float PadResponse::getPadResponse(GlobalPosition3D posEle, DigitPos digiPadPos) const
{
  /// \todo The procedure to find the global position of the pad centre should be made easier
  const Mapper& mapper = Mapper::instance();
  const PadCentre padCentre = mapper.padCentre(mapper.globalPadNumber(digiPadPos.getPadPos()));
  const CRU cru(digiPadPos.getCRU());
  const Sector sector(digiPadPos.getPadSecPos().getSector());
  const LocalPosition3D padCentreLocal(padCentre.X(), padCentre.Y(), posEle.Z());
  const GlobalPosition3D padCentrePos = mapper.LocalToGlobal(padCentreLocal, sector);

  ///std::cout << padCentrePos.X() << " " << posEle.X() << " " << padCentrePos.Y() << " " << posEle.Y() << "\n";

  const int gemStack = int(cru.gemStack());
  const float offsetX = std::abs(posEle.X() - padCentre.X()) * 10.f; /// GlobalPosition3D and DigitPos in cm, PRF in mm
  const float offsetY = std::abs(posEle.Y() - padCentre.Y()) * 10.f; /// GlobalPosition3D and DigitPos in cm, PRF in mm
  float normalizedPadResponse = 0;
  if (gemStack == 0) {
    normalizedPadResponse = mIROC->Interpolate(offsetX, offsetY);
  } else if (gemStack == 1 || gemStack == 2) {
    normalizedPadResponse = mOROC12->Interpolate(offsetX, offsetY);
  } else {
    normalizedPadResponse = mOROC3->Interpolate(offsetX, offsetY);
  }
  return normalizedPadResponse;
}
