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
//using DigitBase = o2::dataformats::TimeStamp<double>;
class Digit
{
  public:
    Digit() = default;
    Digit(int bc, int orbit, int pad, float charge) : mBc(bc), mOrbit(orbit), mPad(pad), mQ(charge) {};

    float getCharge() const { return mQ; }
    int getPadID() const { return mPad; }
    int getOrbit() const { return mOrbit; }
    int getBC() const { return mBc; }

    // convenience conversion to x-y pad coordinates
    int getPx() const { return Param::A2X(mPad); }
    int getPy() const { return Param::A2Y(mPad); }

    static void getPadAndTotalCharge(HitType const& hit, int& chamber, int& pc, int& px, int& py, float& totalcharge);
    static float getFractionalContributionForPad(HitType const& hit, int somepad);

    // add charge to existing digit
    void addCharge(float q) { mQ += q; }

  private:
    float mQ = 0.;
    int mPad = 0.; // -1 indicates invalid digit
    int mBc = 0.;
    int mOrbit = 0;

    static float LorsX(int pad) { return Param::LorsX(Param::A2P(pad), Param::A2X(pad)); } //center of the pad x, [cm]
    static float LorsY(int pad) { return Param::LorsY(Param::A2P(pad), Param::A2Y(pad)); } //center of the pad y, [cm]

    // determines the total charge created by a hit
    // might modify the localX, localY coordiates associated to the hit
    static float QdcTot(float e, float time, int pc, int px, int py, float& localX, float& localY);
    static float IntPartMathiX(float x, int pad);
    static Double_t IntPartMathiY(Double_t y, int pad);
    static float InMathieson(float localX, float localY, int pad);

    ClassDefNV(Digit, 2);
};

} // namespace hmpid
} // namespace o2

#endif /* DETECTORS_HMPID_BASE_INCLUDE_HMPIDBASE_DIGIT_H_ */
