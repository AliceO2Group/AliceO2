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
#include "HMPIDBase/Geo.h"
#include "HMPIDBase/Param.h"
#include "TRandom.h"
#include "TMath.h"
#include "CommonConstants/LHCConstants.h"

using namespace o2::hmpid;

ClassImp(o2::hmpid::Digit);

// ----- Constructors ------------
 Digit::Digit(uint16_t bc, uint32_t orbit, int chamber, int photo, int x, int y, uint16_t charge)
{
  mBc = bc;
  mOrbit = orbit;
  mQ =  charge;
  mPad = Abs(chamber, photo, x, y);
}

 Digit::Digit(uint16_t bc, uint32_t orbit, uint16_t charge, int equipment, int column, int dilogic, int channel)
{
  mBc = bc;
  mOrbit = orbit;
  mQ =  charge;
  mPad = Equipment2Pad(equipment, column, dilogic, channel);
}

 Digit::Digit(uint16_t bc, uint32_t orbit, uint16_t charge, int module, int x, int y)
{
  mBc = bc;
  mOrbit = orbit;
  mQ =  charge;
  mPad = Absolute2Pad(module, x, y);
}

// -----  Coordinate Conversion ----
uint32_t Digit::Equipment2Pad(int Equi, int Colu, int Dilo, int Chan)
{
  // Check the input data
  if(Equi<0 || Equi >= Geo::MAXEQUIPMENTS || Colu<0 || Colu >= Geo::N_COLUMNS ||
      Dilo<0 || Dilo >= Geo::N_DILOGICS || Chan<0 || Chan >= Geo::N_CHANNELS ) return -1;

  int chan2y[6]={3,2,4,1,5,0}; // y coordinate translation for a channel address (index position) for even chamber

  bool isEven = (Equi % Geo::EQUIPMENTSPERMODULE) == 0 ? true : false; // Calculate the odd/even of geometry
  int ch = Equi / Geo::EQUIPMENTSPERMODULE; // The Module

  // Calculate the x,y photo cathode relative coords For Odd equipment
  int pc = (Colu / Geo::N_COLXSEGMENT) * 2 + 1; // col [0..23] -> [1,3,5]
  int px = Geo::MAXXPHOTO - ((Dilo * Geo::DILOPADSROWS) + (Chan / Geo::DILOPADSCOLS));
  int py = (Colu % Geo::DILOPADSROWS) * Geo::DILOPADSCOLS + chan2y[Chan % Geo::DILOPADSCOLS];
  if(isEven) {
    pc = 5 - pc;
    py = Geo::MAXYPHOTO - py;
  }
  return Abs(ch,pc,px,py);  // Pack the coords into the PadID word
}

void Digit::Pad2Equipment(uint32_t pad, int *Equi, int *Colu, int *Dilo, int *Chan)
{
  int ch, ph, px, py;
  int y2chan[6]={5,3,1,0,2,4};

  Pad2Photo(pad, &ch, &ph, &px, &py); // Unpak the pad ID in the photo cathode coords

  bool isEven = (ph % 2) == 0 ? true : false;
  int eq = ch * Geo::EQUIPMENTSPERMODULE +1;
  px = Geo::MAXXPHOTO - px; // revert the X coord
  if(isEven) {
    eq--; // Correct the equipment number
    py = Geo::MAXYPHOTO - py; // revert the Y coord
    ph = 5 - ph; // revert the photo cathode index [0,2,4] -> [5,3,1]
  }
  *Dilo = px / Geo::DILOPADSROWS; // Calculate the Dilogic x [0..79] -> dil [0..9]
  *Colu = ((ph / 2) * Geo::N_COLXSEGMENT) + (py / Geo::DILOPADSCOLS);  // calculate the column  (ph [1,3,5], y [0..47]) -> col [0..23]
  *Chan = ((px % Geo::DILOPADSROWS) * Geo::DILOPADSCOLS) + y2chan[py % Geo::DILOPADSCOLS];
  *Equi = eq;
  return;
}

void Digit::Absolute2Equipment(int Module, int x, int y, int *Equi, int *Colu, int *Dilo, int *Chan)
{
  uint32_t pad = Absolute2Pad(Module, x, y);
  Pad2Equipment(pad, Equi, Colu, Dilo, Chan);
  return;
}

void Digit::Equipment2Absolute(int Equi, int Colu, int Dilo, int Chan, int *Module, int *x, int *y)
{
  uint32_t pad = Equipment2Pad(Equi, Colu, Dilo, Chan);
  Pad2Absolute(pad, Module, x, y);
  return;
}

uint32_t Digit::Absolute2Pad(int Module, int x, int y)
{
  int ph = (y/Geo::N_PHOTOCATODSY)*2+((x >= Geo::HALFXROWS ) ? 1 : 0);
  int px  = x % Geo::HALFXROWS;
  int py  = y % Geo::N_PHOTOCATODSY;
  return Abs(Module,ph,px,py);
}

void Digit::Pad2Absolute(uint32_t pad, int *Module, int *x, int *y)
{
  *Module = A2C(pad);
  int ph = A2P(pad);
  int px  = A2X(pad);
  int py  = A2Y(pad);
  *x = px + ((ph % 2 == 1) ? Geo::HALFXROWS : 0);
  *y = ((ph >> 1) * Geo::N_PHOTOCATODSY) + py;
  return;
}

void Digit::Pad2Photo(uint32_t pad, int *chamber, int *photo, int *x, int *y)
{
  *chamber = A2C(pad);
  *photo = A2P(pad);
  *x  = A2X(pad);
  *y  = A2Y(pad);
  return;
}

// -----  Getter Methods ---------
void Digit::getPadAndTotalCharge(HitType const& hit, int& chamber, int& pc, int& px, int& py, float& totalcharge)
{
  float localX;
  float localY;
  chamber = hit.GetDetectorID();
  double tmp[3] = {hit.GetX(), hit.GetY(), hit.GetZ()};
  Param::Instance()->Mars2Lors(chamber, tmp, localX, localY);
  Param::Lors2Pad(localX, localY, pc, px, py);

  totalcharge = Digit::QdcTot(hit.GetEnergyLoss(), hit.GetTime(), pc, px, py, localX, localY);
  return;
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

// Time conversion functions
double Digit::OrbitBcToTimeNs(uint32_t Orbit, uint16_t BC)
{
  return( BC * o2::constants::lhc::LHCBunchSpacingNS + Orbit * o2::constants::lhc::LHCOrbitNS);
}

uint32_t Digit::TimeNsToOrbit(double TimeNs)
{
  return(TimeNs / o2::constants::lhc::LHCOrbitNS);
}

uint16_t Digit::TimeNsToBc(double TimeNs)
{
  return(std::fmod(TimeNs , o2::constants::lhc::LHCOrbitNS) / o2::constants::lhc::LHCBunchSpacingNS);
}

void Digit::TimeNsToOrbitBc(double TimeNs, uint32_t &Orbit, uint16_t &Bc)
{
  Orbit = TimeNsToOrbit(TimeNs);
  Bc = TimeNsToBc(TimeNs);
  return;
}

// Functions to manage Digit vectors

// Function for order digits (Event,Chamber,Photo,x,y)
bool Digit::eventEquipPadsComp(Digit &d1, Digit &d2)
{
  uint64_t t1,t2;
  t1 = d1.getTriggerID();
  t2 = d2.getTriggerID();
  if (t1 < t2) return true;
  if (t2 < t1) return false;
  if (d1.getPadID() < d2.getPadID()) return true;
  return false;
};

std::vector<o2::hmpid::Digit>* Digit::extractDigitsPerEvent(std::vector<o2::hmpid::Digit> &Digits, uint64_t EventID)
{
  std::vector<o2::hmpid::Digit>* subVector = new std::vector<o2::hmpid::Digit>();
  for(const auto & digit : Digits) {
    if(digit.getTriggerID() == EventID) {
        subVector->push_back(digit);
    }
  }
  return(subVector);
};

std::vector<uint64_t>* Digit::extractEvents(std::vector<o2::hmpid::Digit> &Digits)
{
  std::vector<uint64_t>* subVector = new std::vector<uint64_t>();
  for(const auto & digit : Digits) {
    if(find(subVector->begin(), subVector->end(), digit.getTriggerID()) == subVector->end()) {
      subVector->push_back(digit.getTriggerID());
    }
  }
  return(subVector);
};


