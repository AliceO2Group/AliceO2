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
  
  // Conversion functions
    static inline uint32_t Abs(int ch, int pc, int x, int y) { return ch << 20 | pc << 16 | x << 8 | y; }
    static inline int DDL2C(int ddl) { return ddl >> 1; }                 //ddl -> chamber
    static inline int A2C(uint32_t pad) { return (pad & 0x00F00000) >> 20; }           //abs pad -> chamber
    static inline int A2P(uint32_t pad) { return (pad & 0x000F0000) >> 16; } //abs pad -> pc
    static inline int A2X(uint32_t pad) { return (pad & 0x0000FF00) >> 8; }      //abs pad -> pad X
    static inline int A2Y(uint32_t pad) { return (pad & 0x000000FF); }                //abs pad -> pad Y
    static inline uint32_t Photo2Pad(int ch, int pc, int x, int y) { return Abs(ch, pc, x, y); }
    static uint32_t Equipment2Pad(int Equi, int Colu, int Dilo, int Chan);
    static void Pad2Equipment(uint32_t pad, int *Equi, int *Colu, int *Dilo, int *Chan);
    static void Pad2Absolute(uint32_t pad, int *Module, int *x, int *y);
    static uint32_t Absolute2Pad(int Module, int x, int y);
    static void Pad2Photo(uint32_t pad, int *chamber, int *photo, int *x, int *y);

    // Operators definition !
    inline bool operator<(const Digit& l, const Digit& r) { return l.mPad < r.mPad; };
    inline bool operator==(const Digit& l, const Digit& r) { return l.mPad == r.mPad; };
    inline bool operator> (const Digit& l, const Digit& r){ return r < l; };
    inline bool operator<=(const Digit& l, const Digit& r){ return !(l > r); };
    inline bool operator>=(const Digit& l, const Digit& r){ return !(l < r); };
    inline bool operator!=(const Digit& l, const Digit& r) { return !(l == r); };

  public:
    Digit() = default;
    Digit(int bc, int orbit, int pad, float charge) : mBc(bc), mOrbit(orbit), mQ(charge), mPad(pad) {};
    Digit(int bc, int orbit, int chamber, int photo, int x, int y, float charge);
    Digit(int bc, int orbit, float charge, int equipment, int column, int dilogic, int channel) ;
    Digit(int bc, int orbit, float charge, int module, int x, int y);

    float getCharge() const { return mQ; }
    int getPadID() const { return mPad; }
    int getOrbit() const { return mOrbit; }
    int getBC() const { return mBc; }

    // convenience conversion to x-y pad coordinates
    int getPx() const { return A2X(mPad); }
    int getPy() const { return A2Y(mPad); }
    int getPhC() const { return A2P(mPad); }
    int getCh() const { return A2C(mPad); }


    static void getPadAndTotalCharge(HitType const& hit, int& chamber, int& pc, int& px, int& py, float& totalcharge);
    static float getFractionalContributionForPad(HitType const& hit, int somepad);

    // add charge to existing digit
    void addCharge(float q) { mQ += q; }

  private:
    float mQ = 0.;
    uint32_t mPad = 0.; // -1 indicates invalid digit
    int mBc = 0.;
    int mOrbit = 0;

    static float LorsX(int pad) { return Param::LorsX(A2P(pad), A2X(pad)); } //center of the pad x, [cm]
    static float LorsY(int pad) { return Param::LorsY(A2P(pad), A2Y(pad)); } //center of the pad y, [cm]

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
