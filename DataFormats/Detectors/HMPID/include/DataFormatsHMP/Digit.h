// Copyright 2020-2022 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   Digit.h
/// \author Antonio Franco - INFN Bari
/// \version 1.0
/// \date 15/02/2021

//    History
//  10/03/2021   Complete review
//  05/11/2021   Add and review for the Cluster class

#ifndef DETECTORS_HMPID_BASE_INCLUDE_HMPIDDATAFORMAT_DIGIT_H_
#define DETECTORS_HMPID_BASE_INCLUDE_HMPIDDATAFORMAT_DIGIT_H_

#include <iosfwd>
#include <vector>
#include "DataFormatsHMP/Hit.h" // for hit
#include "HMPIDBase/Param.h"    // for param

namespace o2
{
namespace hmpid
{
/// \class Digit
/// \brief HMPID Digit declaration
class Digit
{
 public:
  // Coordinates Conversion Functions
  static inline uint32_t abs(int ch, int pc, int x, int y) { return ch << 24 | pc << 16 | x << 8 | y; }
  static inline int ddl2C(int ddl) { return ddl >> 1; }                    // ddl -> chamber
  static inline int a2C(uint32_t pad) { return (pad & 0xFF000000) >> 24; } // abs pad -> chamber
  static inline int a2P(uint32_t pad) { return (pad & 0x00FF0000) >> 16; } // abs pad -> pc
  static inline int a2X(uint32_t pad) { return (pad & 0x0000FF00) >> 8; }  // abs pad -> pad X
  static inline int a2Y(uint32_t pad) { return (pad & 0x000000FF); }       // abs pad -> pad Y
  static inline uint32_t photo2Pad(int ch, int pc, int x, int y) { return abs(ch, pc, x, y); }
  static uint32_t equipment2Pad(int Equi, int Colu, int Dilo, int Chan);
  static uint32_t absolute2Pad(int Module, int x, int y);
  static void pad2Equipment(uint32_t pad, int* Equi, int* Colu, int* Dilo, int* Chan);
  static void pad2Absolute(uint32_t pad, int* Module, int* x, int* y);
  static void pad2Photo(uint32_t pad, uint8_t* chamber, uint8_t* photo, uint8_t* x, uint8_t* y);
  static void absolute2Equipment(int Module, int x, int y, int* Equi, int* Colu, int* Dilo, int* Chan);
  static void equipment2Absolute(int Equi, int Colu, int Dilo, int Chan, int* Module, int* x, int* y);

  // Trigger time Conversion Functions
  //  static inline uint64_t orbitBcToEventId(uint32_t Orbit, uint16_t BC) { return ((Orbit << 12) | (0x0FFF & BC)); };
  //  static inline uint32_t eventIdToOrbit(uint64_t EventId) { return (EventId >> 12); };
  //  static inline uint16_t EventIdToBc(uint64_t EventId) { return (EventId & 0x0FFF); };
  //  static double OrbitBcToTimeNs(uint32_t Orbit, uint16_t BC);
  //  static uint32_t TimeNsToOrbit(double TimeNs);
  //  static uint16_t TimeNsToBc(double TimeNs);
  //  static void TimeNsToOrbitBc(double TimeNs, uint32_t& Orbit, uint16_t& Bc);

  // Operators definition !
  friend inline bool operator<(const Digit& l, const Digit& r) { return l.getPadID() < r.getPadID(); };
  friend inline bool operator==(const Digit& l, const Digit& r) { return l.getPadID() == r.getPadID(); };
  friend inline bool operator>(const Digit& l, const Digit& r) { return r < l; };
  friend inline bool operator<=(const Digit& l, const Digit& r) { return !(l > r); };
  friend inline bool operator>=(const Digit& l, const Digit& r) { return !(l < r); };
  friend inline bool operator!=(const Digit& l, const Digit& r) { return !(l == r); };

  friend std::ostream& operator<<(std::ostream& os, const Digit& d);

 public:
  Digit() = default;
  Digit(int pad, uint16_t charge);
  Digit(int chamber, int photo, int x, int y, uint16_t charge);
  Digit(uint16_t charge, int equipment, int column, int dilogic, int channel);
  Digit(uint16_t charge, int module, int x, int y);

  // Getter & Setters
  uint16_t getCharge() const { return mQ; }
  void setCharge(uint16_t Q) { mQ = Q; };

  int getPadID() const { return mCh << 24 | mPh << 16 | mX << 8 | mY; }
  void setPadID(uint32_t pad)
  {
    mCh = pad >> 24;
    mPh = (pad & 0x00FF0000) >> 16;
    mX = (pad & 0x0000FF00) >> 8;
    mY = (pad & 0x000000FF);
  };

  bool isValid() { return (mCh == 0xFF ? true : false); };
  void setInvalid()
  {
    mCh = 0xFF;
    return;
  };

  // // convenience wrapper function for conversion to x-y pad coordinates
  // int getPx() const { return A2X(mPad); }
  // int getPy() const { return A2Y(mPad); }
  // int getPhC() const { return A2P(mPad); }
  // int getCh() const { return A2C(mPad); }

  // Charge management functions
  static void getPadAndTotalCharge(o2::hmpid::HitType const& hit, int& chamber, int& pc, int& px, int& py, float& totalcharge);
  static float getFractionalContributionForPad(o2::hmpid::HitType const& hit, int somepad);
  void addCharge(float q)
  {
    mQ += q;
    if (mQ > 0x0FFF) {
      mQ = 0x0FFF;
    }
  }
  void subCharge(float q) { mQ -= q; }

  uint16_t getQ() const { return mQ; }
  uint8_t getCh() const { return mCh; }
  uint8_t getPh() const { return mPh; }
  uint8_t getX() const { return mX; }
  uint8_t getY() const { return mY; }

 public:
  // Members
  uint16_t mQ = 0;
  uint8_t mCh = 0; // 0xFF indicates invalid digit
  uint8_t mPh = 0;
  uint8_t mX = 0;
  uint8_t mY = 0;

  // The Pad Unique Id, code a pad inside one HMPID chamber.
  // Bit Map : 0000.0000.cccc.pppp.xxxx.xxxx.yyyy.yyyy
  //           cccc := chamber [0..6]
  //           pppp := photo cathode [0..5]
  //           xxxx.xxxx := horizontal displacement [0..79]
  //           yyyy.yyyy := vertical displacement [0..47]
  // uint32_t mPad = 0; // 0xFFFFFFFF indicates invalid digit

  // Get the Geometric center of the pad
  static float lorsX(int pad) { return Param::lorsX(a2P(pad), a2X(pad)); } // center of the pad x, [cm]
  static float lorsY(int pad) { return Param::lorsY(a2P(pad), a2Y(pad)); } // center of the pad y, [cm]

  // determines the total charge created by a hit
  // might modify the localX, localY coordinates associated to the hit
  static double qdcTot(double e, double time, int pc, int px, int py, double& localX, double& localY);
  static double intPartMathiX(double x, int pad);
  static double intPartMathiY(double y, int pad);
  static double intMathieson(double localX, double localY, int pad);
  static double mathiesonX(double x); // Mathieson distribution along wires X
  static double mathiesonY(double x); // Mathieson distribution perp to wires Y

  ClassDefNV(Digit, 2);
};

} // namespace hmpid
} // namespace o2

#endif /* DETECTORS_HMPID_BASE_INCLUDE_HMPIDDATAFORMAT_DIGIT_H_ */
