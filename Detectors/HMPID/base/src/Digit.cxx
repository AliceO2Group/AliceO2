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
  /*
   *       Original algorithm in AliROOT Run 1/2
   *
   * kNDILOGICAdd = 10
   *
    if(ddl<0 || ddl >13 || row<1 || row >25 || dil<1 || dil >10 || pad<0 || pad >47 ) return -1;
  Int_t a2y[6]={3,2,4,1,5,0};     //pady for a given padress (for single DILOGIC chip)
  Int_t ch=ddl/2;
  Int_t tmp=(24-row)/8;
  Int_t pc=(ddl%2)?5-2*tmp:2*tmp;
  Int_t px=(kNDILOGICAdd+1 - dil)*8-pad/6-1;  //flip according to Paolo (2-9-2008)

  tmp=(ddl%2)?row-1:(24-row);
  Int_t py=6*(tmp%8)+a2y[pad%6];
   */

  // Check the input data
  if(Equi<0 || Equi >= Geo::MAXEQUIPMENTS || Colu<0 || Colu >= Geo::N_COLUMNS ||
      Dilo<0 || Dilo >= Geo::N_DILOGICS || Chan<0 || Chan >= Geo::N_CHANNELS ) return -1;

  int chan2y[6]={3,2,4,1,5,0}; // y coordinate translation for a channel address (index position) for even chamber

  // Calculate the odd/even of geometry
  bool isEven = (Equi % Geo::EQUIPMENTSPERMODULE) == 0 ? true : false;
  int ch = Equi / Geo::EQUIPMENTSPERMODULE; // The Module

  // Calculate the x,y photo cathode relative coords
  int pc = (23-Colu) / Geo::N_COLXSEGMENT * 2;  // pho = [0,2,4]
  int px = (Geo::N_DILOGICS - Dilo) * Geo::DILOPADSROWS - Chan / Geo::DILOPADSCOLS - 1; // (dil [0..9] ,chan[0..47]) -> x [79..0]
  int py = Geo::DILOPADSCOLS * ((23-Colu) % Geo::N_COLXSEGMENT) + chan2y[Chan % Geo::DILOPADSCOLS];// Col[0..23] -> y [47..0]

  if(!isEven) {
    pc = 5 - pc; // revert the photo cathode index
    px = Geo::MAXXPHOTO - px; // revert the x coordinate
    py = Geo::MAXYPHOTO - py; // revert the y coordinate
  }

  /* ---- Last good algorithm -------

  int a2y[6]={3,2,4,1,5,0};     //pady for a given padress (for single DILOGIC chip)
  int ch = Equi / Geo::EQUIPMENTSPERMODULE; // The Module
  int tmp = (23 - Colu) / Geo::N_COLXSEGMENT;

  int py;
  int pc;
  int px;
  if((Equi % Geo::EQUIPMENTSPERMODULE) != 0 ) {
    pc = 5-2*tmp;
    px = Dilo * Geo::DILOPADSROWS + Chan / Geo::DILOPADSCOLS;
    tmp = Colu;
  }else {
    pc = 2*tmp;
    px = (Geo::N_DILOGICS - Dilo) * Geo::DILOPADSROWS - Chan / Geo::DILOPADSCOLS - 1;
    tmp = (23-Colu);

  }
  py = Geo::DILOPADSCOLS * (tmp % Geo::DILOPADSROWS)+a2y[Chan % Geo::DILOPADSCOLS];
  py = py % 48;
  */

  /* Direct Translation from Aliroot
  int pc = (Equi % Geo::EQUIPMENTSPERMODULE) ? 5-2*tmp : 2*tmp; // The PhotoCatode
  int px = (Geo::N_DILOGICS - Dilo) * Geo::DILOPADSROWS - Chan / Geo::DILOPADSCOLS - 1;
  tmp = (Equi % Geo::EQUIPMENTSPERMODULE) ? Colu : (23-Colu);
  int py = Geo::DILOPADSCOLS * (tmp % Geo::DILOPADSROWS)+a2y[Chan % Geo::DILOPADSCOLS];

 */

  return Abs(ch,pc,px,py);  // Pack the coords into the PadID word
}

void Digit::Pad2Equipment(uint32_t pad, int *Equi, int *Colu, int *Dilo, int *Chan)
{
  int ch, ph, px, py;
  int y2chan[6]={5,3,1,0,2,4};

  Pad2Photo(pad, &ch, &ph, &px, &py); // Unpak the pad ID in the photo cathode coords

  bool isEven = (ph % 2) == 0 ? true : false;
  int eq = ch * Geo::EQUIPMENTSPERMODULE;

  if(!isEven) {
    eq++; // Correct the equipment number
    px = Geo::MAXXPHOTO - px; // revert the X coord
    py = Geo::MAXYPHOTO - py; // revert the Y coord
    ph = 5 - ph; // revert the photo cathode index
  }
  *Dilo = Geo::N_DILOGICS - (px / Geo::DILOPADSROWS) -1;  // Calculate the Dilogic x [0..79] -> dil [9..0]
  *Colu = 23 - (((ph / 2) * Geo::N_COLXSEGMENT) + (py / Geo::DILOPADSCOLS));  // calculate the column  (ph [0,2,4], y [0..47]) -> col [23..0]
  *Chan = ((Geo::MAXXPHOTO-px) % Geo::DILOPADSROWS) * Geo::DILOPADSCOLS + y2chan[py % Geo::DILOPADSCOLS];
  *Equi = eq;

  return;
  /*  last good algorithm
  int ch, ax, ay;
  int y2a[6]={4,2,0,1,3,5};
  Pad2Absolute(pad, &ch, &ax, &ay);
  if (ax > Geo::MAXHALFXROWS) {
    *Equi = ch * Geo::EQUIPMENTSPERMODULE + 1;
    ax = ax - Geo::HALFXROWS;
    *Chan = (ax % Geo::DILOPADSROWS) * Geo::DILOPADSCOLS + y2a[5 -(ay % Geo::DILOPADSCOLS)];
  } else {
    *Equi = ch * Geo::EQUIPMENTSPERMODULE;
    ax = Geo::MAXHALFXROWS - ax;
    ay = Geo::MAXYCOLS - ay;
    *Chan = (ax % Geo::DILOPADSROWS) * Geo::DILOPADSCOLS + y2a[ay % Geo::DILOPADSCOLS];
  }
  *Dilo = ax / Geo::DILOPADSROWS;
  *Colu = ay / Geo::DILOPADSCOLS;
  return;
  */
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
  *x = (ph % 2 == 0) ? px : (px + Geo::HALFXROWS);
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

