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

/// \file SignalInduction.cxx
/// \brief Implementation of the class handling the signal induction
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/SignalInduction.h"
#include <string>
#include "FairLogger.h"

using namespace o2::tpc;

SignalInduction::SignalInduction() : grPRFIROC(),
                                     grPRFOROC12(),
                                     grPRFOROC3()
{
  loadPadResponse();
}

void SignalInduction::loadPadResponse()
{
  std::string inputDir, irocPath, oroc12path, oroc3path;
  const char* aliceO2env = std::getenv("O2_ROOT");
  if (aliceO2env) {
    inputDir = aliceO2env;
  } else {
    LOG(FATAL) << "tpc::SignalInduction - Input dir " << inputDir << " does not exist! No PRF loaded!";
  }
  inputDir += "/share/Detectors/TPC/files/";

  irocPath = inputDir + "PRF_sampa_IROC_all.dat";
  oroc12path = inputDir + "PRF_sampa_OROC1-2_all.dat";
  oroc3path = inputDir + "PRF_sampa_OROC3_all.dat";
  grPRFIROC = new TGraph2D(irocPath.data());
  grPRFIROC->SetName("PRF_IROC");
  if (!grPRFIROC) {
    LOG(FATAL) << "tpc::SignalInduction - Loading IROC PRF from " << irocPath << " failed No PRF loaded!";
  }
  grPRFOROC12 = new TGraph2D(oroc12path.data());
  grPRFOROC12->SetName("PRF_OROC12");
  if (!grPRFOROC12) {
    LOG(FATAL) << "tpc::SignalInduction - Loading OROC 1/2 PRF from " << oroc12path << " failed No PRF loaded!";
  }
  grPRFOROC3 = new TGraph2D(oroc3path.data());
  grPRFOROC3->SetName("PRF_OROC3");
  if (!grPRFOROC3) {
    LOG(FATAL) << "tpc::SignalInduction - Loading OROC 3 PRF from " << oroc3path << " failed No PRF loaded!";
  }
}

PadResponse SignalInduction::computePadResponse(const GlobalPosition3D posEle, const DigitPos digitPos, const double offsetPad, const double offsetRow) const
{
  /// The distance from the electron arrival position to the pad center is computed and the pad response function is evaluated accordingly.
  /// The pad for which this is done can be varied with the variables offsetPad (number of pads left/right of the pad at which the electron arrives) and offsetRow (number of row above/below the row at which the electron arrives) and
  const static Mapper& mapper = Mapper::instance();
  const Sector sector = digitPos.getCRU().sector();
  const GEMstack gemstack = digitPos.getCRU().gemStack();

  /// Compute the cpad and global row coordinates
  const LocalPosition3D elePosLocal = Mapper::GlobalToLocal(posEle, sector);
  const int globalRow = digitPos.getGlobalPadPos().getRow();
  const int globalPad = digitPos.getGlobalPadPos().getPad();
  const int globalCPad = globalPad - 0.5 * mapper.getNumberOfPadsInRowSector(globalRow);

  /// Shift by the offset and compute the offset in x and y
  const PadPos shiftedGlobalPos(globalRow + offsetRow, globalCPad + offsetPad + 0.5 * mapper.getNumberOfPadsInRowSector(globalRow + offsetRow));
  const GlobalPadNumber shiftedGlobalPadNum = mapper.globalPadNumber(shiftedGlobalPos);
  const PadCentre& padCent = mapper.padCentre(shiftedGlobalPadNum);
  const double localYfactor = (sector.side() == Side::A) ? -1.f : 1.f;
  const double offsetX = std::abs(elePosLocal.X() - padCent.X()) * 10.f;
  const double offsetY = std::abs(elePosLocal.Y() - localYfactor * padCent.Y()) * 10.f;

  /// TODO For some reason the rows have a mismatch?
  const DigitPos shiftedDigitPos(mapper.getCRU(sector, shiftedGlobalPadNum), mapper.padPos(shiftedGlobalPadNum));
  //  std::cout << " " << int(digitPos.getPadPos().getPad()) << " " << int(shiftedDigitPos.getPadPos().getPad()) << " " << int(digitPos.getGlobalPadPos().getRow()) << " " << int(shiftedDigitPos.getGlobalPadPos().getRow()) << "\n";

  return {shiftedDigitPos, computePadResponse(offsetX, offsetY, gemstack)};
}

void SignalInduction::getPadResponse(const GlobalPosition3D posEle, const DigitPos digitPos, std::vector<PadResponse>& signalArray, const PadResponseMode& mode) const
{
  signalArray.resize(0);
  switch (mode) {
    case PadResponseMode::FullPadResponse: {
      std::vector<double> offset = {{-1, 0, 1}};
      for (auto offsetRow : offset) {
        for (auto offsetPad : offset) {
          signalArray.push_back(computePadResponse(posEle, digitPos, offsetPad, offsetRow));
        }
      }
      break;
    }
    case PadResponseMode::ProjectivePadResponse: {
      signalArray.push_back({digitPos, 1});
      break;
    }
  }
}
