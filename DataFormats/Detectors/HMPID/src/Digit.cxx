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

/// \file   Digit.cxx
/// \author Antonio Franco - INFN Bari
/// \brief Base Class to manage HMPID Digit data
/// \version 1.0
/// \date 15/02/2021

/* ------ HISTORY ---------
  10/03/2021   /  complete review
  05/11/2021   Add and review for the Cluster class
*/

#include <iostream>
#include <TRandom.h>
#include "CommonConstants/LHCConstants.h"
#include "HMPIDBase/Geo.h"
#include "HMPIDBase/Param.h"
#include "DataFormatsHMP/Digit.h"

ClassImp(o2::hmpid::Digit);

namespace o2
{
namespace hmpid
{

// ============= Digit Class implementation =======
/// Constructor : Create the Digit structure. Accepts the trigger time (Orbit,BC)
///               The mapping of the digit is in the Photo Cathod coords
///               (Chamber, PhotoCathod, X, Y)
/// @param[in] pad : the Digit Unique Id [0x00CPXXYY]
/// @param[in] charge : the value of the charge [0 .. 2^12-1]
Digit::Digit(int pad, uint16_t charge)
{
  mQ = charge > 0x0FFF ? 0x0FFF : charge;
  mCh = a2C(pad);
  mPh = a2P(pad);
  mX = a2X(pad);
  mY = a2Y(pad);
}

/// Constructor : Create the Digit structure. Accepts the trigger time (Orbit,BC)
///               The mapping of the digit is in the Photo Cathod coords
///               (Chamber, PhotoCathod, X, Y)
/// @param[in] chamber : the HMPID module [0 .. 6]
/// @param[in] photo : the photo cathode number [0 .. 5] (left-down to right-up)
/// @param[in] x : the horizontal in cathode displacement [0 .. 79]
/// @param[in] y : the vertical in cathode displacement [0 .. 47]
/// @param[in] charge : the value of the charge [0 .. 2^12-1]
Digit::Digit(int chamber, int photo, int x, int y, uint16_t charge)
{
  mQ = charge > 0x0FFF ? 0x0FFF : charge;
  mCh = chamber;
  mPh = photo;
  mX = x;
  mY = y;
}

/// Constructor : Create the Digit structure. Accepts the trigger time (Orbit,BC)
///               The mapping of the digit is in the Hardware coords
///               (Equipment, Column, Dilogic, Channel)
/// @param[in] charge : the value of the charge [0 .. 2^12-1]
/// @param[in] equipment : the HMPID DDL link [0 .. 13]
/// @param[in] column : the readout column number [0 .. 23]
/// @param[in] dilogic : the displacement in the Dilogics chain [0 .. 9]
/// @param[in] channel : the number of gassiplexes channels [0 .. 47]
Digit::Digit(uint16_t charge, int equipment, int column, int dilogic, int channel)
{
  mQ = charge > 0x0FFF ? 0x0FFF : charge;
  pad2Photo(equipment2Pad(equipment, column, dilogic, channel), &mCh, &mPh, &mX, &mY);
}

/// Constructor : Create the Digit structure. Accepts the trigger time (Orbit,BC)
///               The mapping of the digit is in the Logical coords
///               (Module, X, Y)
/// @param[in] charge : the value of the charge [0 .. 2^12-1]
/// @param[in] module : the HMPID Module [0 .. 6]
/// @param[in] x : the horizontal in Module displacement [0 .. 159]
/// @param[in] y : the vertical in Module displacement [0 .. 143]
Digit::Digit(uint16_t charge, int module, int x, int y)
{
  mQ = charge > 0x0FFF ? 0x0FFF : charge;
  pad2Photo(absolute2Pad(module, x, y), &mCh, &mPh, &mX, &mY);
}

// Digit ASCCI format Dump := [Chamber,PhotoCathod,X,Y]@(Orbit,BunchCrossing)=Charge
std::ostream& operator<<(std::ostream& os, const o2::hmpid::Digit& d)
{
  os << "[" << (int)d.mCh << "," << (int)d.mPh << "," << (int)d.mX << "," << (int)d.mY << "]=" << d.mQ;
  return os;
};

// -----  Coordinate Conversion -----

/// Equipment2Pad : Converts the coords from Hardware to Digit Unique Id
/// @param[in] Equi : the equipment [0 .. 13]
/// @param[in] Colu : the readout column number [0 .. 23]
/// @param[in] Dilo : the displacement in the Dilogics chain [0 .. 9]
/// @param[in] Chan : the number of gassiplexes channels [0 .. 47]
/// @return uint32_t : the Digit Unique Id [0x00CPXXYY]
uint32_t Digit::equipment2Pad(int Equi, int Colu, int Dilo, int Chan)
{
  // Check the input data
  if (Equi < 0 || Equi >= Geo::MAXEQUIPMENTS || Colu < 0 || Colu >= Geo::N_COLUMNS ||
      Dilo < 0 || Dilo >= Geo::N_DILOGICS || Chan < 0 || Chan >= Geo::N_CHANNELS) {
    return -1;
  }

  int chan2y[6] = {3, 2, 4, 1, 5, 0}; // y coordinate translation for a channel address (index position) for even chamber

  bool isEven = (Equi % Geo::EQUIPMENTSPERMODULE) == 0 ? true : false; // Calculate the odd/even of geometry
  int ch = Equi / Geo::EQUIPMENTSPERMODULE;                            // The Module

  // Calculate the x,y photo cathode relative coords For Odd equipment
  int pc = (Colu / Geo::N_COLXSEGMENT) * 2 + 1; // col [0..23] -> [1,3,5]
  int px = Geo::MAXXPHOTO - ((Dilo * Geo::DILOPADSROWS) + (Chan / Geo::DILOPADSCOLS));
  int py = (Colu % Geo::DILOPADSROWS) * Geo::DILOPADSCOLS + chan2y[Chan % Geo::DILOPADSCOLS];
  if (isEven) {
    pc = 5 - pc;
    py = Geo::MAXYPHOTO - py;
  }
  return abs(ch, pc, px, py); // Pack the coords into the PadID word
}

/// Pad2Equipment : Converts the Digit Unique Id to Hardware coords
/// @param[in] pad : the Digit Unique Id [0x00CPXXYY]
/// @param[out] Equi : the equipment [0 .. 13]
/// @param[out] Colu : the readout column number [0 .. 23]
/// @param[out] Dilo : the displacement in the Dilogics chain [0 .. 9]
/// @param[out] Chan : the number of gassiplexes channels [0 .. 47]
void Digit::pad2Equipment(uint32_t pad, int* Equi, int* Colu, int* Dilo, int* Chan)
{
  uint8_t ch, ph, px, py;
  int y2chan[6] = {5, 3, 1, 0, 2, 4};

  pad2Photo(pad, &ch, &ph, &px, &py); // Unpak the pad ID in the photo cathode coords

  bool isEven = (ph % 2) == 0 ? true : false;
  int eq = ch * Geo::EQUIPMENTSPERMODULE + 1;
  px = Geo::MAXXPHOTO - px; // revert the X coord
  if (isEven) {
    eq--;                     // Correct the equipment number
    py = Geo::MAXYPHOTO - py; // revert the Y coord
    ph = 5 - ph;              // revert the photo cathode index [0,2,4] -> [5,3,1]
  }
  *Dilo = px / Geo::DILOPADSROWS;                                     // Calculate the Dilogic x [0..79] -> dil [0..9]
  *Colu = ((ph / 2) * Geo::N_COLXSEGMENT) + (py / Geo::DILOPADSCOLS); // calculate the column  (ph [1,3,5], y [0..47]) -> col [0..23]
  *Chan = ((px % Geo::DILOPADSROWS) * Geo::DILOPADSCOLS) + y2chan[py % Geo::DILOPADSCOLS];
  *Equi = eq;
  return;
}

/// Absolute2Equipment : Converts the Module coords to Hardware coords
/// @param[in] Module : the HMPID Module number [0..6]
/// @param[in] x : the horizontal displacement [0..159]
/// @param[in] y : the vertical displacement [0..143]
/// @param[out] Equi : the equipment [0 .. 13]
/// @param[out] Colu : the readout column number [0 .. 23]
/// @param[out] Dilo : the displacement in the Dilogics chain [0 .. 9]
/// @param[out] Chan : the number of gassiplexes channels [0 .. 47]
void Digit::absolute2Equipment(int Module, int x, int y, int* Equi, int* Colu, int* Dilo, int* Chan)
{
  uint32_t pad = absolute2Pad(Module, x, y);
  pad2Equipment(pad, Equi, Colu, Dilo, Chan);
  return;
}

/// Equipment2Absolute : Converts the Module coords to Hardware coords
/// @param[in] Equi : the equipment [0 .. 13]
/// @param[in] Colu : the readout column number [0 .. 23]
/// @param[in] Dilo : the displacement in the Dilogics chain [0 .. 9]
/// @param[in] Chan : the number of gassiplexes channels [0 .. 47]
/// @param[out] Module : the HMPID Module number [0..6]
/// @param[out] x : the horizontal displacement [0..159]
/// @param[out] y : the vertical displacement [0..143]
void Digit::equipment2Absolute(int Equi, int Colu, int Dilo, int Chan, int* Module, int* x, int* y)
{
  uint32_t pad = equipment2Pad(Equi, Colu, Dilo, Chan);
  pad2Absolute(pad, Module, x, y);
  return;
}

/// Absolute2Pad : Converts the Module coords in the Digit Unique Id
/// @param[in] Module : the HMPID Module number [0..6]
/// @param[in] x : the horizontal displacement [0..159]
/// @param[in] y : the vertical displacement [0..143]
/// @return uint32_t : the Digit Unique Id [0x00CPXXYY]
uint32_t Digit::absolute2Pad(int Module, int x, int y)
{
  int ph = (y / Geo::N_PHOTOCATODSY) * 2 + ((x >= Geo::HALFXROWS) ? 1 : 0);
  int px = x % Geo::HALFXROWS;
  int py = y % Geo::N_PHOTOCATODSY;
  return abs(Module, ph, px, py);
}

/// Pad2Absolute : Converts the the Digit Unique Id to Module coords
/// @param[in] pad : the Digit Unique Id [0x00CPXXYY]
/// @param[out] Module : the HMPID Module number [0..6]
/// @param[out] x : the horizontal displacement [0..159]
/// @param[out] y : the vertical displacement [0..143]
void Digit::pad2Absolute(uint32_t pad, int* Module, int* x, int* y)
{
  *Module = a2C(pad);
  int ph = a2P(pad);
  int px = a2X(pad);
  int py = a2Y(pad);
  *x = px + ((ph % 2 == 1) ? Geo::HALFXROWS : 0);
  *y = ((ph >> 1) * Geo::N_PHOTOCATODSY) + py;
  return;
}

/// Pad2Photo : Converts the the Digit Unique Id to Photo Cathode coords
/// @param[in] pad : the Digit Unique Id [0x00CPXXYY]
/// @param[out] chamber : the HMPID chamber number [0..6]
/// @param[out] photo : the photo cathode number [0..5]
/// @param[out] x : the horizontal displacement [0..79]
/// @param[out] y : the vertical displacement [0..47]
void Digit::pad2Photo(uint32_t pad, uint8_t* chamber, uint8_t* photo, uint8_t* x, uint8_t* y)
{
  *chamber = a2C(pad);
  *photo = a2P(pad);
  *x = a2X(pad);
  *y = a2Y(pad);
  return;
}

/// getPadAndTotalCharge : Extract all the info from the Hit structure
/// and returns they in the Photo Cathode coords
/// @param[in] hit : the HMPID Hit
/// @param[out] chamber : the HMPID chamber number [0..6]
/// @param[out] pc : the photo cathode number [0..5]
/// @param[out] px : the horizontal displacement [0..79]
/// @param[out] py : the vertical displacement [0..47]
/// @param[out] totalcharge : the charge of the hit [0..2^12-1]
void Digit::getPadAndTotalCharge(HitType const& hit, int& chamber, int& pc, int& px, int& py, float& totalcharge)
{
  double localX;
  double localY;
  chamber = hit.GetDetectorID();
  double tmp[3] = {hit.GetX(), hit.GetY(), hit.GetZ()};
  Param::instance()->mars2Lors(chamber, tmp, localX, localY);
  Param::lors2Pad(localX, localY, pc, px, py);

  totalcharge = Digit::qdcTot(hit.GetEnergyLoss(), hit.GetTime(), pc, px, py, localX, localY);
  return;
}

/// getFractionalContributionForPad : ...
///
/// @param[in] hit : the HMPID Hit
/// @param[in] somepad : the Digit Unique Id [0x00CPXXYY]
/// @return : the fraction of the charge ...
float Digit::getFractionalContributionForPad(HitType const& hit, int somepad)
{
  double localX;
  double localY;

  const auto chamber = hit.GetDetectorID(); // chamber number is in detID
  double tmp[3] = {hit.GetX(), hit.GetY(), hit.GetZ()};
  // converting chamber id and hit coordiates to local coordinates
  Param::instance()->mars2Lors(chamber, tmp, localX, localY);
  // calculate charge fraction in given pad
  return Digit::intMathieson(localX, localY, somepad);
}

/// QdcTot : Samples total charge associated to a hit
///
/// @param[in] e : hit energy [GeV] for mip Eloss for photon Etot
/// @param[in] time : ...
/// @param[in] pc : the photo cathode number [0..5]
/// @param[in] px : the horizontal displacement [0..79]
/// @param[in] py : the vertical displacement [0..47]
/// @param[out] localX : the horizontal displacement related to Anode Wires
/// @param[out] localY : the vertical displacement  related to Anode Wires
/// @return : total QDC
Double_t Digit::qdcTot(Double_t e, Double_t time, Int_t pc, Int_t px, Int_t py, Double_t& localX, Double_t& localY)
{
  //
  // Arguments: e-
  //  Returns:
  double Q = 0;
  if (time > 1.2e-6) {
    Q = 0;
  }
  if (py < 0) {
    return 0;
  } else {
    double y = Param::lorsY(pc, py);
    localY = ((y - localY) > 0) ? y - 0.2 : y + 0.2; // shift to the nearest anod wire

    double x = (localX > 66.6) ? localX - 66.6 : localX;                                                                      // sagita is for PC (0-64) and not for chamber
    double qdcEle = 34.06311 + 0.2337070 * x + 5.807476e-3 * x * x - 2.956471e-04 * x * x * x + 2.310001e-06 * x * x * x * x; // reparametrised from DiMauro

    int iNele = int((e / 26e-9) * 0.8);
    if (iNele < 1) {
      iNele = 1; // number of electrons created by hit, if photon e=0 implies iNele=1
    }
    for (Int_t i = 1; i <= iNele; i++) {
      double rnd = gRandom->Rndm();
      if (rnd == 0) {
        rnd = 1e-12; // 1e-12 is a protection against 0 from rndm
      }
      Q -= qdcEle * TMath::Log(rnd);
    }
  }
  return Q;
}

/// IntPartMathiX : Integration of Mathieson.
/// This is the answer to electrostatic problem of charge distrubution in MWPC
/// described elsewhere. (NIM A370(1988)602-603)
///
/// @param[in] x : position of the center of Mathieson distribution
/// @param[in] pad : the Digit Unique Id [0x00CPXXYY]
/// @return : a charge fraction [0-1] imposed into the pad
double Digit::intPartMathiX(double x, int pad)
{
  double shift1 = -lorsX(pad) + o2::hmpid::Param::sizeHalfPadX();
  double shift2 = -lorsX(pad) - o2::hmpid::Param::sizeHalfPadX();

  double ux1 = o2::hmpid::Param::sqrtK3x() * tanh(o2::hmpid::Param::k2x() * (x + shift1) / o2::hmpid::Param::pitchAnodeCathode());
  double ux2 = o2::hmpid::Param::sqrtK3x() * tanh(o2::hmpid::Param::k2x() * (x + shift2) / o2::hmpid::Param::pitchAnodeCathode());

  return o2::hmpid::Param::k4x() * (atan(ux2) - atan(ux1));
}

/// IntPartMathiY : Integration of Mathieson.
/// This is the answer to electrostatic problem of charge distrubution in MWPC
/// described elsewhere. (NIM A370(1988)602-603)
///
/// @param[in] y : position of the center of Mathieson distribution
/// @param[in] pad : the Digit Unique Id [0x00CPXXYY]
/// @return : a charge fraction [0-1] imposed into the pad
double Digit::intPartMathiY(double y, int pad)
{
  double shift1 = -lorsY(pad) + o2::hmpid::Param::sizeHalfPadY();
  double shift2 = -lorsY(pad) - o2::hmpid::Param::sizeHalfPadY();

  double uy1 = o2::hmpid::Param::sqrtK3y() * tanh(o2::hmpid::Param::k2y() * (y + shift1) / o2::hmpid::Param::pitchAnodeCathode());
  double uy2 = o2::hmpid::Param::sqrtK3y() * tanh(o2::hmpid::Param::k2y() * (y + shift2) / o2::hmpid::Param::pitchAnodeCathode());

  return o2::hmpid::Param::k4y() * (atan(uy2) - atan(uy1));
}

/// IntMathieson : Integration of Mathieson.
/// This is the answer to electrostatic problem of charge distrubution in MWPC
/// described elsewhere. (NIM A370(1988)602-603)
///
/// @param[in] localX : X position of the center of Mathieson distribution
/// @param[in] localY : Y position of the center of Mathieson distribution
/// @param[in] pad : the Digit Unique Id [0x00CPXXYY]
/// @return : a charge fraction [0-1] imposed into the pad
double Digit::intMathieson(double localX, double localY, int pad)
{
  return 4. * intPartMathiX(localX, pad) * intPartMathiY(localY, pad);
}

// Mathieson function.
// This is the answer to electrostatic problem of charge distrubution in MWPC described elsewhere. (NIM A370(1988)602-603)
// Arguments: x- position of the center of Mathieson distribution
//  Returns: value of the Mathieson function
double Digit::mathiesonX(double x)
{
  double lambda = x / o2::hmpid::Param::pitchAnodeCathode();
  double tanh_v = tanh(o2::hmpid::Param::k2x() * lambda);
  double a = 1 - tanh_v * tanh_v;
  double b = 1 + o2::hmpid::Param::sqrtK3x() * o2::hmpid::Param::sqrtK3x() * tanh_v * tanh_v;
  double mathi = o2::hmpid::Param::k1x() * a / b;
  return mathi;
}

// Mathieson function.
// This is the answer to electrostatic problem of charge distrubution in MWPC described elsewhere. (NIM A370(1988)602-603)
// Arguments: x- position of the center of Mathieson distribution
//  Returns: value of the Mathieson function
double Digit::mathiesonY(double y)
{
  double lambda = y / o2::hmpid::Param::pitchAnodeCathode();
  double tanh_v = tanh(o2::hmpid::Param::k2y() * lambda);
  double a = 1 - tanh_v * tanh_v;
  double b = 1 + o2::hmpid::Param::sqrtK3y() * o2::hmpid::Param::sqrtK3y() * tanh_v * tanh_v;
  double mathi = o2::hmpid::Param::k1y() * a / b;
  return mathi;
}

/*
// ---- Time conversion functions ----

/// OrbitBcToTimeNs : Converts the Orbit,BC pair in absolute
/// nanoseconds time.
///
/// @param[in] Orbit : the Orbit number [0..2^32-1]
/// @param[in] BC : the Bunch Crossing Number [0..2^12-1]
/// @return : the absolute time in nanoseconds
Double_t Digit::OrbitBcToTimeNs(uint32_t Orbit, uint16_t BC)
{
  return (BC * o2::constants::lhc::LHCBunchSpacingNS + Orbit * o2::constants::lhc::LHCOrbitNS);
}

/// TimeNsToOrbit : Extracts the Orbit number from the absolute
/// nanoseconds time.
///
/// @param[in] TimeNs : the absolute nanoseconds time
/// @return : the Orbit number [0..2^32-1]
uint32_t Digit::TimeNsToOrbit(Double_t TimeNs)
{
  return (uint32_t)(TimeNs / o2::constants::lhc::LHCOrbitNS);
}

/// TimeNsToBc : Extracts the Bunch Crossing number from the absolute
/// nanoseconds time.
///
/// @param[in] TimeNs : the absolute nanoseconds time
/// @return : the Bunch Crossing number [0..2^12-1]
uint16_t Digit::TimeNsToBc(Double_t TimeNs)
{
  return (uint16_t)(std::fmod(TimeNs, o2::constants::lhc::LHCOrbitNS) / o2::constants::lhc::LHCBunchSpacingNS);
}

/// TimeNsToOrbitBc : Extracts the (Orbit,BC) pair from the absolute
/// nanoseconds time.
///
/// @param[in] TimeNs : the absolute nanoseconds time
/// @param[out] Orbit : the Orbit number [0..2^32-1]
/// @param[out] Bc : the Bunch Crossing number [0..2^12-1]
void Digit::TimeNsToOrbitBc(double TimeNs, uint32_t& Orbit, uint16_t& Bc)
{
  Orbit = TimeNsToOrbit(TimeNs);
  Bc = TimeNsToBc(TimeNs);
  return;
}


// ---- Functions to manage Digit vectors ----

/// eventEquipPadsComp : Function for order digits (Event,Chamber,Photo,x,y)
/// to use in sort method function overload
/// @param[in] d1 : one Digit
/// @param[in] d2 : one Digit
/// @return : true if event of d1 comes before the event of d2, for
///           same events evaluates the position into the detector
bool Digit::eventEquipPadsComp(Digit& d1, Digit& d2)
{
  uint64_t t1, t2;
  t1 = d1.getTriggerID();
  t2 = d2.getTriggerID();
  if (t1 < t2) {
    return true;
  }
  if (t2 < t1) {
    return false;
  }
  if (d1.getPadID() < d2.getPadID()) {
    return true;
  }
  return false;
};


/// extractDigitsPerEvent : Function for select a sub vector of Digits of the
/// same event
/// @param[in] Digits : one vector of Digits
/// @param[in] EventID : the Trigger ID [ 0000.0000.0000.0000.0000.oooo.oooo.oooo.oooo.oooo.oooo.oooo.oooo.bbbb.bbbb.bbbb ]
/// @return : the subvector of Digits that have the same EventID
std::vector<o2::hmpid::Digit>* Digit::extractDigitsPerEvent(std::vector<o2::hmpid::Digit>& Digits, uint64_t EventID)
{
  std::vector<o2::hmpid::Digit>* subVector = new std::vector<o2::hmpid::Digit>();
  for (const auto& digit : Digits) {
    if (digit.getTriggerID() == EventID) {
      subVector->push_back(digit);
    }
  }
  return (subVector);
};

/// extractEvents : Function that returns the list of Event IDs from a
/// vector of Digits
/// @param[in] Digits : one vector of Digits
/// @return : the vector of Event IDs
std::vector<uint64_t>* Digit::extractEvents(std::vector<o2::hmpid::Digit>& Digits)
{
  std::vector<uint64_t>* eventIds = new std::vector<uint64_t>();
  for (const auto& digit : Digits) {
    if (find(eventIds->begin(), eventIds->end(), digit.getTriggerID()) == eventIds->end()) {
      eventIds->push_back(digit.getTriggerID());
    }
  }
  return (eventIds);
};
*/

} // namespace hmpid
} // namespace o2
