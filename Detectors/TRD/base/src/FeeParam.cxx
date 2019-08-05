// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/////////////////////////////////////////////////////////////////////
//
//  TRD front end electronics parameters class
//  Contains all FEE (MCM, TRAP, PASA) related
//  parameters, constants, and mapping.
//
//   2007/08/17:
//   The default raw data version (now mRAWversion ) is set to 3
//   in the constructor because version 3 raw data read and write
//   are fully debugged.
//
//  Author:
//    Ken Oyama (oyama@physi.uni-heidelberg.de)
//
//  many things now configured by AliTRDtrapConfig reflecting
//  the real memory structure of the TRAP (Jochen)
//
//  Now has the mcm to pad lookup table mapping
//////////////////////////////////////////////////////////////////

#include <TGeoManager.h>
#include <TGeoPhysicalNode.h>
#include <TMath.h>
#include <TVirtualMC.h>
#include <fairlogger/Logger.h>

#include "DetectorsBase/GeometryManager.h"
#include "TRDBase/TRDGeometry.h"
#include "TRDBase/TRDPadPlane.h"
#include "TRDBase/FeeParam.h"
#include "TRDBase/TRDCommonParam.h"

using namespace o2::trd;

//_____________________________________________________________________________

FeeParam* FeeParam::mgInstance = nullptr;
bool FeeParam::mgTerminated = kFALSE;
bool FeeParam::mgTracklet = kTRUE;
bool FeeParam::mgRejectMultipleTracklets = kFALSE;
bool FeeParam::mgUseMisalignCorr = kFALSE;
bool FeeParam::mgUseTimeOffset = kFALSE;
bool FeeParam::mgLUTPadNumberingFilled = kFALSE;
std::vector<short> FeeParam::mgLUTPadNumbering;

// definition of geometry constants
std::array<float, 30> FeeParam::mgZrow = {
  301, 177, 53, -57, -181,
  301, 177, 53, -57, -181,
  315, 184, 53, -57, -188,
  329, 191, 53, -57, -195,
  343, 198, 53, -57, -202,
  347, 200, 53, -57, -204};
std::array<float, 6> FeeParam::mgX = {300.65, 313.25, 325.85, 338.45, 351.05, 363.65};
std::array<float, 6> FeeParam::mgTiltingAngle = {-2., 2., -2., 2., -2., 2.};
int FeeParam::mgDyMax = 63;
int FeeParam::mgDyMin = -64;
float FeeParam::mgBinDy = 140e-4;
std::array<float, 6> FeeParam::mgWidthPad = {0.635, 0.665, 0.695, 0.725, 0.755, 0.785};
std::array<float, 6> FeeParam::mgLengthInnerPadC1 = {7.5, 7.5, 8.0, 8.5, 9.0, 9.0};
std::array<float, 6> FeeParam::mgLengthOuterPadC1 = {7.5, 7.5, 7.5, 7.5, 7.5, 8.5};
std::array<float, 6> FeeParam::mgInvX;
std::array<float, 6> FeeParam::mgTiltingAngleTan;
std::array<float, 6> FeeParam::mgInvWidthPad;

float FeeParam::mgLengthInnerPadC0 = 9.0;
float FeeParam::mgLengthOuterPadC0 = 8.0;
float FeeParam::mgScalePad = 256. * 32.;
float FeeParam::mgDriftLength = 3.;

//_____________________________________________________________________________
FeeParam* FeeParam::instance()
{
  //
  // Instance constructor
  //

  if (mgTerminated != kFALSE) {
    return nullptr;
  }

  if (mgInstance == nullptr) {
    mgInstance = new FeeParam();
  }

  return mgInstance;
}

//_____________________________________________________________________________
void FeeParam::terminate()
{
  //
  // Terminate the class and release memory
  //

  mgTerminated = kTRUE;

  if (mgInstance != nullptr) {
    delete mgInstance;
    mgInstance = nullptr;
  }
}

//_____________________________________________________________________________
FeeParam::FeeParam() : mMagField(0.),
                       mOmegaTau(0.),
                       mPtMin(0.1),
                       mNtimebins(20 << 5),
                       mScaleQ0(0),
                       mScaleQ1(0),
                       mPidTracklengthCorr(false),
                       mTiltCorr(false),
                       mPidGainCorr(false)
{
  //
  // Default constructor
  //

  mCP = TRDCommonParam::Instance();
  createPad2MCMLookUpTable();

  // These variables are used internally in the class to elliminate divisions.
  // putting them at the top was messy.
  int j = 0;
  std::for_each(mgInvX.begin(), mgInvX.end(), [&j](float& x) { x = 1. / mgX[j]; });
  j = 0;
  std::for_each(mgInvWidthPad.begin(), mgInvWidthPad.end(), [&j](float& x) { x = 1. / mgWidthPad[j]; });
  j = 0;
  std::for_each(mgTiltingAngleTan.begin(), mgTiltingAngleTan.end(), [&j](float& x) { x = std::tan(mgTiltingAngle[j] * M_PI / 180.0); });

  mInvPtMin = 1 / mPtMin;
}

//_____________________________________________________________________________
//FeeParam::FeeParam(TRootIoCtor*)
//{
//
// IO constructor
//
//}

//_____________________________________________________________________________
FeeParam::FeeParam(const FeeParam& p)
{
  //
  // FeeParam copy constructor
  //
  mRAWversion = p.mRAWversion;
  mCP = p.mCP;
  createPad2MCMLookUpTable();
}

//_____________________________________________________________________________
FeeParam::~FeeParam() = default;

//_____________________________________________________________________________
FeeParam& FeeParam::operator=(const FeeParam& p)
{
  //
  // Assignment operator
  //

  if (this != &p) {
    ((FeeParam&)p).Copy(*this);
  }

  return *this;
}

//_____________________________________________________________________________
void FeeParam::Copy(FeeParam& p) const
{
  //
  // Copy function
  //

  p.mCP = mCP;
  p.mRAWversion = mRAWversion;
}

//_____________________________________________________________________________
int FeeParam::getPadRowFromMCM(int irob, int imcm) const
{
  //
  // Return on which pad row this mcm sits
  //

  return mgkNmcmRobInRow * (irob / 2) + imcm / mgkNmcmRobInCol;
}

//_____________________________________________________________________________
int FeeParam::getPadColFromADC(int irob, int imcm, int iadc) const
{
  //
  // Return which pad is connected to this adc channel.
  //
  // Return virtual pad number even if ADC is outside chamber
  // to keep compatibility of data processing at the edge MCM.
  // User has to check that this is in the chamber if it is essential.
  // Return -100 if iadc is invalid.
  //
  // Caution: ADC ordering in the online data is opposite to the pad column ordering.
  // And it is not one-by-one correspondence. Precise drawing can be found in:
  // http://wiki.kip.uni-heidelberg.de/ti/TRD/index.php/Image:ROB_MCM_numbering.pdf
  //

  if (iadc < 0 || iadc > mgkNadcMcm)
    return -100;
  int mcmcol = imcm % mgkNmcmRobInCol + getRobSide(irob) * mgkNmcmRobInCol; // MCM column number on ROC [0..7]
  int padcol = mcmcol * mgkNcolMcm + mgkNcolMcm + 1 - iadc;
  if (padcol < 0 || padcol >= mgkNcol)
    return -1; // this is commented because of reason above OK

  return padcol;
}

//_____________________________________________________________________________
int FeeParam::getExtendedPadColFromADC(int irob, int imcm, int iadc) const
{
  //
  // Return which pad coresponds to the extended digit container pad numbering
  // Extended digit container is designed to store all pad data including shared pad,
  // so we have to introduce new virtual pad numbering scheme for this purpose.
  //

  if (iadc < 0 || iadc > mgkNadcMcm)
    return -100;
  int mcmcol = imcm % mgkNmcmRobInCol + getRobSide(irob) * mgkNmcmRobInCol; // MCM column number on ROC [0..7]
  int padcol = mcmcol * mgkNadcMcm + mgkNcolMcm + 2 - iadc;

  return padcol;
}

//_____________________________________________________________________________
int FeeParam::getMCMfromPad(int irow, int icol) const
{
  //
  // Return on which MCM this pad is directry connected.
  // Return -1 for error.
  //

  if (irow < 0 || icol < 0 || irow > mgkNrowC1 || icol > mgkNcol)
    return -1;

  return (icol % (mgkNcol / 2)) / mgkNcolMcm + mgkNmcmRobInCol * (irow % mgkNmcmRobInRow);
}

//_____________________________________________________________________________
int FeeParam::getMCMfromSharedPad(int irow, int icol) const
{
  //
  // Return on which MCM this pad is directry connected.
  // Return -1 for error.
  //

  if (irow < 0 || icol < 0 || irow > mgkNrowC1 || icol > mgkNcol + 8 * 3)
    return -1;

  int adc = 20 - (icol % 18) - 1;
  switch (adc) {
    case 2:
      icol += 5;
      break;
    case 18:
      icol -= 5;
      break;
    case 19:
      icol -= 5;
      break;
    default:
      icol += 0;
      break;
  }

  return (icol % (mgkNcol / 2)) / mgkNcolMcm + mgkNmcmRobInCol * (irow % mgkNmcmRobInRow);
}

//_____________________________________________________________________________
int FeeParam::getROBfromPad(int irow, int icol) const
{
  //
  // Return on which rob this pad is
  //

  return (irow / mgkNmcmRobInRow) * 2 + getColSide(icol);
}

//_____________________________________________________________________________
int FeeParam::getROBfromSharedPad(int irow, int icol) const
{
  //
  // Return on which rob this pad is for shared pads
  //

  if (icol < 72)
    return (irow / mgkNmcmRobInRow) * 2 + getColSide(icol + 5);
  else
    return (irow / mgkNmcmRobInRow) * 2 + getColSide(icol - 5);
}

//_____________________________________________________________________________
int FeeParam::getRobSide(int irob) const
{
  //
  // Return on which side this rob sits (A side = 0, B side = 1)
  //

  if (irob < 0 || irob >= mgkNrobC1)
    return -1;

  return irob % 2;
}

//_____________________________________________________________________________
int FeeParam::getColSide(int icol) const
{
  //
  // Return on which side this column sits (A side = 0, B side = 1)
  //

  if (icol < 0 || icol >= mgkNcol)
    return -1;

  return icol / (mgkNcol / 2);
}

unsigned int FeeParam::aliToExtAli(int rob, int aliid)
{
  if (aliid != 127)
    return ((1 << 10) | (rob << 7) | aliid);

  return 127;
}

int FeeParam::extAliToAli(unsigned int dest, unsigned short linkpair, unsigned short rocType, int* mcmList, int listSize)
{
  // Converts an extended ALICE ID which identifies a single MCM or a group of MCMs to
  // the corresponding list of MCMs. Only broadcasts (127) are encoded as 127
  // The return value is the number of MCMs in the list

  mcmList[0] = -1;

  short nmcm = 0;
  unsigned int mcm, rob, robAB;
  unsigned int cmA = 0, cmB = 0; // Chipmask for each A and B side

  // Default chipmask for 4 linkpairs (each bit correponds each alice-mcm)
  static const unsigned int gkChipmaskDefLp[4] = {0x1FFFF, 0x1FFFF, 0x3FFFF, 0x1FFFF};

  rob = dest >> 7;                 // Extract ROB pattern from dest.
  mcm = dest & 0x07F;              // Extract MCM pattern from dest.
  robAB = getRobAB(rob, linkpair); // Get which ROB sides are selected.

  // Abort if no ROB is selected
  if (robAB == 0) {
    return 0;
  }

  // Special case
  if (mcm == 127) {
    if (robAB == 3) {   // This is very special 127 can stay only if two ROBs are selected
      mcmList[0] = 127; // broadcase to ALL
      mcmList[1] = -1;
      return 1;
    }
    cmA = cmB = 0x3FFFF;
  } else if ((mcm & 0x40) != 0) { // If top bit is 1 but not 127, this is chip group.
    if ((mcm & 0x01) != 0) {
      cmA |= 0x04444;
      cmB |= 0x04444;
    } // chip_cmrg
    if ((mcm & 0x02) != 0) {
      cmA |= 0x10000;
      cmB |= 0x10000;
    } // chip_bmrg
    if ((mcm & 0x04) != 0 && rocType == 0) {
      cmA |= 0x20000;
      cmB |= 0x20000;
    } // chip_hm3
    if ((mcm & 0x08) != 0 && rocType == 1) {
      cmA |= 0x20000;
      cmB |= 0x20000;
    } // chip_hm4
    if ((mcm & 0x10) != 0) {
      cmA |= 0x01111;
      cmB |= 0x08888;
    } // chip_edge
    if ((mcm & 0x20) != 0) {
      cmA |= 0x0aaaa;
      cmB |= 0x03333;
    }      // chip_norm
  } else { // Otherwise, this is normal chip ID, turn on only one chip.
    cmA = 1 << mcm;
    cmB = 1 << mcm;
  }

  // Mask non-existing MCMs
  cmA &= gkChipmaskDefLp[linkpair];
  cmB &= gkChipmaskDefLp[linkpair];
  // Remove if only one side is selected
  if (robAB == 1)
    cmB = 0;
  if (robAB == 2)
    cmA = 0;
  if (robAB == 4 && linkpair != 2)
    cmA = cmB = 0; // Restrict to only T3A and T3B

  // Finally convert chipmask to list of slaves
  nmcm = chipmaskToMCMlist(cmA, cmB, linkpair, mcmList, listSize);

  return nmcm;
}

short FeeParam::getRobAB(unsigned short robsel, unsigned short linkpair)
{
  // Converts the ROB part of the extended ALICE ID to robs

  if ((robsel & 0x8) != 0) { // 1000 .. direct ROB selection. Only one of the 8 ROBs are used.
    robsel = robsel & 7;
    if ((robsel % 2) == 0 && (robsel / 2) == linkpair)
      return 1; // Even means A side (position 0,2,4,6)
    if ((robsel % 2) == 1 && (robsel / 2) == linkpair)
      return 2; // Odd  means B side (position 1,3,5,7)
    return 0;
  }

  // ROB group
  if (robsel == 0) {
    return 3;
  } // Both   ROB
  if (robsel == 1) {
    return 1;
  } // A-side ROB
  if (robsel == 2) {
    return 2;
  } // B-side ROB
  if (robsel == 3) {
    return 3;
  } // Both   ROB
  if (robsel == 4) {
    return 4;
  } // Only T3A and T3B
  // Other number 5 to 7 are ignored (not defined)

  return 0;
}

short FeeParam::chipmaskToMCMlist(unsigned int cmA, unsigned int cmB, unsigned short linkpair, int* mcmList, int listSize)
{
  // Converts the chipmask to a list of MCMs

  short nmcm = 0;
  short i;
  for (i = 0; i < 18; i++) { // 18: number of MCMs on a ROB
    if ((cmA & (1 << i)) != 0 && nmcm < listSize) {
      mcmList[nmcm] = ((linkpair * 2) << 7) | i;
      ++nmcm;
    }
    if ((cmB & (1 << i)) != 0 && nmcm < listSize) {
      mcmList[nmcm] = ((linkpair * 2 + 1) << 7) | i;
      ++nmcm;
    }
  }

  mcmList[nmcm] = -1;
  return nmcm;
}

//_____________________________________________________________________________
void FeeParam::setRAWversion(int rawver)
{
  //
  // Set raw data version (major number only)
  // Maximum available number is preset in mgkMaxRAWversion
  //

  if (rawver >= 0 && rawver <= mgkMaxRAWversion) {
    mRAWversion = rawver;
  } else {
    LOG(error) << "Raw version is out of range: " << rawver;
  }
}

void FeeParam::createPad2MCMLookUpTable()
{

  //
  // Initializes the Look Up Table to relate
  // pad numbering and mcm channel numbering
  //

  if (!mgLUTPadNumberingFilled) {

    //   mgLUTPadNumbering.resize(FeeParam::getNcol());
    //  memset(&mgLUTPadNumbering[0], 0, sizeof(mgLUTPadNumbering[0]) * FeeParam::getNcol());

    for (int mcm = 0; mcm < 8; mcm++) {
      int lowerlimit = 0 + mcm * 18;
      int upperlimit = 18 + mcm * 18;
      int shiftposition = 1 + 3 * mcm;
      for (int index = lowerlimit; index < upperlimit; index++) {
        FeeParam::instance()->mgLUTPadNumbering[index] = index + shiftposition;
      }
    }
    mgLUTPadNumberingFilled = kTRUE;
  }
}

int FeeParam::getDyCorrection(int det, int rob, int mcm) const
{
  // calculate the correction of the deflection
  // i.e. Lorentz angle and tilt correction (if active)

  int layer = det % 6;

  float dyTilt = (mgDriftLength * std::tan(mgTiltingAngle[layer] * M_PI / 180.) *
                  getLocalZ(det, rob, mcm) * mgInvX[layer]);

  // calculate Lorentz correction
  float dyCorr = -mOmegaTau * mgDriftLength;

  if (mTiltCorr)
    dyCorr += dyTilt; // add tilt correction

  return (int)TMath::Nint(dyCorr * mgScalePad * mgInvWidthPad[layer]);
}

void FeeParam::getDyRange(int det, int rob, int mcm, int ch,
                          int& dyMinInt, int& dyMaxInt) const
{
  // calculate the deflection range in which tracklets are accepted

  dyMinInt = mgDyMin;
  dyMaxInt = mgDyMax;

  // deflection cut is considered for |B| > 0.1 T only
  if (std::abs(mMagField) < 0.1)
    return;

  float e = 0.30;

  float maxDeflTemp = getPerp(det, rob, mcm, ch) / 2. *             // Sekante/2 (cm)
                      (e * 1e-2 * std::abs(mMagField) * mInvPtMin); // 1/R (1/cm)

  float phi = getPhi(det, rob, mcm, ch);
  if (maxDeflTemp < std::cos(phi)) {
    float maxDeflAngle = std::asin(maxDeflTemp);

    float dyMin = (mgDriftLength *
                   std::tan(phi - maxDeflAngle));

    dyMinInt = int(dyMin / mgBinDy);
    // clipping to allowed range
    if (dyMinInt < mgDyMin)
      dyMinInt = mgDyMin;
    else if (dyMinInt > mgDyMax)
      dyMinInt = mgDyMax;

    float dyMax = (mgDriftLength *
                   std::tan(phi + maxDeflAngle));

    dyMaxInt = int(dyMax / mgBinDy);
    // clipping to allowed range
    if (dyMaxInt > mgDyMax)
      dyMaxInt = mgDyMax;
    else if (dyMaxInt < mgDyMin)
      dyMaxInt = mgDyMin;
  } else if (maxDeflTemp < 0.) {
    // this must not happen
    printf("Inconsistent calculation of sin(alpha): %f\n", maxDeflTemp);
  } else {
    // TRD is not reached at the given pt threshold
    // max range
  }

  if ((dyMaxInt - dyMinInt) <= 0) {
    LOG(info) << "strange dy range: [" << dyMinInt << "," << dyMaxInt << "], using max range now";
    dyMaxInt = mgDyMax;
    dyMinInt = mgDyMin;
  }
}

float FeeParam::getElongation(int det, int rob, int mcm, int ch) const
{
  // calculate the ratio of the distance to the primary vertex and the
  // distance in x-direction for the given ADC channel

  int layer = det % 6;

  float elongation = std::abs(getDist(det, rob, mcm, ch) * mgInvX[layer]);

  // sanity check
  if (elongation < 0.001) {
    elongation = 1.;
  }
  return elongation;
}

void FeeParam::getCorrectionFactors(int det, int rob, int mcm, int ch,
                                    unsigned int& cor0, unsigned int& cor1, float gain) const
{
  // calculate the gain correction factors for the given ADC channel
  float Invgain = 1.0;
  if (mPidGainCorr == true)
    Invgain = 1 / gain;

  if (mPidTracklengthCorr == true) {
    float InvElongationOverGain = 1 / getElongation(det, rob, mcm, ch) * Invgain;
    cor0 = (unsigned int)(mScaleQ0 * InvElongationOverGain);
    cor1 = (unsigned int)(mScaleQ1 * InvElongationOverGain);
  } else {
    cor0 = (unsigned int)(mScaleQ0 * Invgain);
    cor1 = (unsigned int)(mScaleQ1 * Invgain);
  }
}

int FeeParam::getNtimebins() const
{
  // return the number of timebins used

  return mNtimebins;
}

float FeeParam::getX(int det, int /* rob */, int /* mcm */) const
{
  // return the distance to the beam axis in x-direction

  int layer = det % 6;
  return mgX[layer];
}

float FeeParam::getLocalY(int det, int rob, int mcm, int ch) const
{
  // get local y-position (r-phi) w.r.t. the chamber centre

  int layer = det % 6;
  // calculate the pad position as in the TRAP
  float ypos = (-4 + 1 + (rob & 0x1) * 4 + (mcm & 0x3)) * 18 - ch - 0.5; // y position in bins of pad widths
  return ypos * mgWidthPad[layer];
}

float FeeParam::getLocalZ(int det, int rob, int mcm) const
{
  // get local z-position w.r.t. to the chamber boundary

  int stack = (det % 30) / 6;
  int layer = det % 6;
  int row = (rob / 2) * 4 + mcm / 4;

  if (stack == 2) {
    if (row == 0)
      return (mgZrow[layer * 6 + stack] - 0.5 * mgLengthOuterPadC0);
    else if (row == 11)
      return (mgZrow[layer * 6 + stack] - 1.5 * mgLengthOuterPadC0 - (row - 1) * mgLengthInnerPadC0);
    else
      return (mgZrow[layer * 6 + stack] - mgLengthOuterPadC0 - (row - 0.5) * mgLengthInnerPadC0);
  } else {
    if (row == 0)
      return (mgZrow[layer * 6 + stack] - 0.5 * mgLengthOuterPadC1[layer]);
    else if (row == 15)
      return (mgZrow[layer * 6 + stack] - 1.5 * mgLengthOuterPadC1[layer] - (row - 1) * mgLengthInnerPadC1[layer]);
    else
      return (mgZrow[layer * 6 + stack] - mgLengthOuterPadC1[layer] - (row - 0.5) * mgLengthInnerPadC1[layer]);
  }
}

float FeeParam::getPerp(int det, int rob, int mcm, int ch) const
{
  // get transverse distance to the beam axis
  float y;
  float x;
  x = getX(det, rob, mcm);
  y = getLocalY(det, rob, mcm, ch);
  return std::sqrt(y * y + x * x);
}

float FeeParam::getPhi(int det, int rob, int mcm, int ch) const
{
  // calculate the azimuthal angle for the given ADC channel

  return std::atan2(getLocalY(det, rob, mcm, ch), getX(det, rob, mcm));
}

float FeeParam::getDist(int det, int rob, int mcm, int ch) const
{
  // calculate the distance from the origin for the given ADC channel
  float x, y, z;
  x = getX(det, rob, mcm);
  y = getLocalY(det, rob, mcm, ch);
  z = getLocalZ(det, rob, mcm);

  return std::sqrt(y * y + x * x + z * z);
}
