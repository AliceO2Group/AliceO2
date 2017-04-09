/// \file PadResponse.cxx
/// \brief Implementation of the Pad Response
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/PadResponse.h"

#include "TPCBase/Mapper.h"

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include "FairLogger.h"

using namespace o2::TPC;

PadResponse::PadResponse()
  : mIROC()
  , mOROC12()
  , mOROC3()
{
  mIROC   = std::unique_ptr<TGraph2D> (new TGraph2D());
  mOROC12 = std::unique_ptr<TGraph2D> (new TGraph2D());
  mOROC3  = std::unique_ptr<TGraph2D> (new TGraph2D());
  
  importPRF("PRF_IROC.dat", mIROC);
  importPRF("PRF_OROC1-2.dat", mOROC12);
  importPRF("PRF_OROC3.dat", mOROC3);
}

bool PadResponse::importPRF(std::string file, std::unique_ptr<TGraph2D> & grPRF) const
{
  std::string inputDir;
  const char* aliceO2env=std::getenv("O2_ROOT");
  if (aliceO2env) inputDir=aliceO2env;
  inputDir+="/share/Detectors/TPC/files/";

  float x, y, normalizedPadResponse;
  int i=0;
  std::ifstream prfFile(inputDir+file, std::ifstream::in);
  if(!prfFile) {
    LOG(FATAL) << "TPC::PadResponse - Input file '" << inputDir+file << "' does not exist! No PRF loaded!" << FairLogger::endl;
    return false;
  }
  for (std::string line; std::getline(prfFile, line); ) {
    std::istringstream is(line);
    while(is >> x >> y >> normalizedPadResponse) {
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
  const LocalPosition3D padCentreLocal(padCentre.getX(), padCentre.getY(), posEle.getZ());
  const GlobalPosition3D padCentrePos = mapper.LocalToGlobal(padCentreLocal, sector);

  ///std::cout << padCentrePos.getX() << " " << posEle.getX() << " " << padCentrePos.getY() << " " << posEle.getY() << "\n";

  const int gemStack = int(cru.gemStack());
  const float offsetX = std::fabs(posEle.getX() - padCentre.getX())*10.f; /// GlobalPosition3D and DigitPos in cm, PRF in mm
  const float offsetY = std::fabs(posEle.getY() - padCentre.getY())*10.f; /// GlobalPosition3D and DigitPos in cm, PRF in mm
  float normalizedPadResponse = 0;
  if(gemStack == 0) {
    normalizedPadResponse = mIROC->Interpolate(offsetX, offsetY);
  }
  else if(gemStack == 1 || gemStack == 2) {
    normalizedPadResponse = mOROC12->Interpolate(offsetX, offsetY);
  }
  else {
    normalizedPadResponse = mOROC3->Interpolate(offsetX, offsetY);
  }
  return normalizedPadResponse;
}
