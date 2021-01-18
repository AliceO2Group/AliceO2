// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "HMPIDBase/Digit.h"
#include "HMPIDBase/Param.h"
#include "TRandom.h"
#include "TMath.h"

using namespace o2::hmpid;

ClassImp(o2::hmpid::Digit);

void Digit::getPadAndTotalCharge(HitType const& hit, int& chamber, int& pc, int& px, int& py, float& totalcharge)
{
  float localX;
  float localY;
  chamber = hit.GetDetectorID();
  double tmp[3] = {hit.GetX(), hit.GetY(), hit.GetZ()};
  Param::Instance()->Mars2Lors(chamber, tmp, localX, localY);
  Param::Lors2Pad(localX, localY, pc, px, py);

  totalcharge = Digit::QdcTot(hit.GetEnergyLoss(), hit.GetTime(), pc, px, py, localX, localY);
}

float Digit::getFractionalContributionForPad(HitType const& hit, int somepad)
{
  float localX;
  float localY;

  // chamber number is in detID
  const auto chamber = hit.GetDetectorID();
  double tmp[3] = {hit.GetX(), hit.GetY(), hit.GetZ()};
  // converting chamber id and hit coordiates to local coordinates
  Param::Instance()->Mars2Lors(chamber, tmp, localX, localY);
  // calculate charge fraction in given pad
  return Digit::InMathieson(localX, localY, somepad);
}


float Digit::QdcTot(float e, float time, int pc, int px, int py, float& localX, float& localY)
{
  // Samples total charge associated to a hit
  // Arguments: e- hit energy [GeV] for mip Eloss for photon Etot
  //  Returns: total QDC
  float Q = 0;
  if (time > 1.2e-6) {
    Q = 0;
  }
  if (py < 0) {
    return 0;
  } else {
    float y = Param::LorsY(pc, py);
    localY = ((y - localY) > 0) ? y - 0.2 : y + 0.2; //shift to the nearest anod wire

    float x = (localX > 66.6) ? localX - 66.6 : localX;                                                                      //sagita is for PC (0-64) and not for chamber
    float qdcEle = 34.06311 + 0.2337070 * x + 5.807476e-3 * x * x - 2.956471e-04 * x * x * x + 2.310001e-06 * x * x * x * x; //reparametrised from DiMauro

    int iNele = int((e / 26e-9) * 0.8);
    if (iNele < 1) {
      iNele = 1; //number of electrons created by hit, if photon e=0 implies iNele=1
    }
    for (Int_t i = 1; i <= iNele; i++) {
      double rnd = gRandom->Rndm();
      if (rnd == 0) {
        rnd = 1e-12; //1e-12 is a protection against 0 from rndm
      }
      Q -= qdcEle * TMath::Log(rnd);
    }
  }
  return Q;
}

float Digit::IntPartMathiX(float x, int pad)
{
  // Integration of Mathieson.
  // This is the answer to electrostatic problem of charge distrubution in MWPC described elsewhere. (NIM A370(1988)602-603)
  // Arguments: x,y- position of the center of Mathieson distribution
  //  Returns: a charge fraction [0-1] imposed into the pad
  auto shift1 = -LorsX(pad) + 0.5 * Param::SizePadX();
  auto shift2 = -LorsX(pad) - 0.5 * Param::SizePadX();

  auto ux1 = Param::SqrtK3x() * TMath::TanH(Param::K2x() * (x + shift1) / Param::PitchAnodeCathode());
  auto ux2 = Param::SqrtK3x() * TMath::TanH(Param::K2x() * (x + shift2) / o2::hmpid::Param::PitchAnodeCathode());

  return o2::hmpid::Param::K4x() * (TMath::ATan(ux2) - TMath::ATan(ux1));
}


Double_t Digit::IntPartMathiY(Double_t y, int pad)
{
  // Integration of Mathieson.
  // This is the answer to electrostatic problem of charge distrubution in MWPC described elsewhere. (NIM A370(1988)602-603)
  // Arguments: x,y- position of the center of Mathieson distribution
  //  Returns: a charge fraction [0-1] imposed into the pad
  Double_t shift1 = -LorsY(pad) + 0.5 * o2::hmpid::Param::SizePadY();
  Double_t shift2 = -LorsY(pad) - 0.5 * o2::hmpid::Param::SizePadY();

  Double_t uy1 = Param::SqrtK3y() * TMath::TanH(Param::K2y() * (y + shift1) / Param::PitchAnodeCathode());
  Double_t uy2 = Param::SqrtK3y() * TMath::TanH(Param::K2y() * (y + shift2) / Param::PitchAnodeCathode());

  return Param::K4y() * (TMath::ATan(uy2) - TMath::ATan(uy1));
}

float Digit::InMathieson(float localX, float localY, int pad)
{
  return 4. * Digit::IntPartMathiX(localX, pad) * Digit::IntPartMathiY(localY, pad);
}

