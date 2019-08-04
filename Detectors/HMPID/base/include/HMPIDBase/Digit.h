// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef DETECTORS_HMPID_BASE_INCLUDE_HMPIDBASE_DIGIT_H_
#define DETECTORS_HMPID_BASE_INCLUDE_HMPIDBASE_DIGIT_H_

#include "CommonDataFormat/TimeStamp.h"
#include "HMPIDBase/Hit.h"   // for hit
#include "HMPIDBase/Param.h" // for param
#include "TMath.h"

namespace o2
{
namespace hmpid
{

/// \class Digit
/// \brief HMPID digit implementation
using DigitBase = o2::dataformats::TimeStamp<double>;
class Digit : public DigitBase
{
 public:
  Digit() = default;
  Digit(double time, int pad, float charge) : DigitBase(time), mPad(pad), mQ(charge) {}

  float getCharge() const { return mQ; }
  int getPadID() const { return mPad; }
  // convenience conversion to x-y pad coordinates
  int getPx() const { return Param::A2X(mPad); }
  int getPy() const { return Param::A2Y(mPad); }

  static void getPadAndTotalCharge(HitType const& hit, int& chamber, int& pc, int& px, int& py, float& totalcharge)
  {
    float localX;
    float localY;
    chamber = hit.GetDetectorID();
    double tmp[3] = {hit.GetX(), hit.GetY(), hit.GetZ()};
    Param::Instance()->Mars2Lors(chamber, tmp, localX, localY);
    Param::Lors2Pad(localX, localY, pc, px, py);

    totalcharge = Digit::QdcTot(hit.GetEnergyLoss(), hit.GetTime(), pc, px, py, localX, localY);
  }

  static float getFractionalContributionForPad(HitType const& hit, int somepad)
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

  // add charge to existing digit
  void addCharge(float q) { mQ += q; }

 private:
  float mQ = 0.;
  int mPad = 0.; // -1 indicates invalid digit

  static float LorsX(int pad) { return Param::LorsX(Param::A2P(pad), Param::A2X(pad)); } //center of the pad x, [cm]
  static float LorsY(int pad) { return Param::LorsY(Param::A2P(pad), Param::A2Y(pad)); } //center of the pad y, [cm]

  // determines the total charge created by a hit
  // might modify the localX, localY coordiates associated to the hit
  static float QdcTot(float e, float time, int pc, int px, int py, float& localX, float& localY);

  static float IntPartMathiX(float x, int pad)
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
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  static Double_t IntPartMathiY(Double_t y, int pad)
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

  static float InMathieson(float localX, float localY, int pad)
  {
    return 4. * Digit::IntPartMathiX(localX, pad) * Digit::IntPartMathiY(localY, pad);
  }

  ClassDefNV(Digit, 2);
};

} // namespace hmpid
} // namespace o2

#endif /* DETECTORS_HMPID_BASE_INCLUDE_HMPIDBASE_DIGIT_H_ */
