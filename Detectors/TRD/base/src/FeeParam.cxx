// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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
#include <array>

#include "DataFormatsTRD/HelperMethods.h"
#include "DetectorsBase/GeometryManager.h"
#include "TRDBase/Geometry.h"
#include "TRDBase/PadPlane.h"
#include "TRDBase/FeeParam.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace o2::trd;
using namespace o2::trd::constants;

//_____________________________________________________________________________

FeeParam* FeeParam::mgInstance = nullptr;

//_____________________________________________________________________________
FeeParam* FeeParam::instance()
{
  //
  // Instance constructor
  //
#ifdef WITH_OPENMP
#pragma omp critical
#endif
  { // start omp critical block
    if (mgInstance == nullptr) {
      mgInstance = new FeeParam();
    }

    return mgInstance;
  } // end omp critical block
}


//_____________________________________________________________________________
FeeParam::FeeParam()
{
  //
  // Default constructor
  //

  for (int j = 0; j < constants::NLAYER; ++j) {
    mInvX[j] = 1. / mX[j];
    mInvWidthPad[j] = 1. / mWidthPad[j];
    mTiltingAngleTan[j] = std::tan(mTiltingAngle[j] * M_PI / 180.0);
  }

  fillPad2MCMLookUpTable();
}



//_____________________________________________________________________________
int FeeParam::getPadRowFromMCM(int irob, int imcm)
{
  //
  // Return on which pad row this mcm sits
  //

  return HelperMethods::getPadRowFromMCM(irob, imcm);
}

//_____________________________________________________________________________
int FeeParam::getPadColFromADC(int irob, int imcm, int iadc)
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

  return HelperMethods::getPadColFromADC(irob, imcm, iadc);
}

//_____________________________________________________________________________
int FeeParam::getExtendedPadColFromADC(int irob, int imcm, int iadc)
{
  //
  // Return which pad coresponds to the extended digit container pad numbering
  // Extended digit container is designed to store all pad data including shared pad,
  // so we have to introduce new virtual pad numbering scheme for this purpose.
  //

  if (iadc < 0 || iadc > NADCMCM) {
    return -100;
  }
  int mcmcol = imcm % NMCMROBINCOL + getROBSide(irob) * NMCMROBINCOL; // MCM column number on ROC [0..7]
  int padcol = mcmcol * NADCMCM + NCOLMCM + 2 - iadc;

  return padcol;
}

//_____________________________________________________________________________
int FeeParam::getMCMfromPad(int irow, int icol)
{
  //
  // Return on which MCM this pad is directry connected.
  // Return -1 for error.
  //

  return HelperMethods::getMCMfromPad(irow, icol);
}

//_____________________________________________________________________________
int FeeParam::getMCMfromSharedPad(int irow, int icol)
{
  //
  // Return on which MCM this pad is directry connected.
  // Return -1 for error.
  //

  if (irow < 0 || icol < 0 || irow > NROWC1 || icol > NCOLUMN + 8 * 3) {
    return -1;
  }

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

  return (icol % (NCOLUMN / 2)) / NCOLMCM + NMCMROBINCOL * (irow % NMCMROBINROW);
}

//_____________________________________________________________________________
int FeeParam::getROBfromPad(int irow, int icol)
{
  //
  // Return on which rob this pad is
  //
  return HelperMethods::getROBfromPad(irow, icol);
}

//_____________________________________________________________________________
int FeeParam::getROBfromSharedPad(int irow, int icol)
{
  //
  // Return on which rob this pad is for shared pads
  //

  if (icol < 72) {
    return (irow / NMCMROBINROW) * 2 + getColSide(icol + 5);
  } else {
    return (irow / NMCMROBINROW) * 2 + getColSide(icol - 5);
  }
}

//_____________________________________________________________________________
int FeeParam::getROBSide(int irob)
{
  //
  // Return on which side this rob sits (A side = 0, B side = 1)
  //
  return HelperMethods::getROBSide(irob);
}

//_____________________________________________________________________________
int FeeParam::getColSide(int icol)
{
  //
  // Return on which side this column sits (A side = 0, B side = 1)
  //

  return HelperMethods::getColSide(icol);
}

unsigned int FeeParam::aliToExtAli(int rob, int aliid)
{
  if (aliid != 127) {
    return ((1 << 10) | (rob << 7) | aliid);
  }

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
  const unsigned int gkChipmaskDefLp[4] = {0x1FFFF, 0x1FFFF, 0x3FFFF, 0x1FFFF};

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
  if (robAB == 1) {
    cmB = 0;
  }
  if (robAB == 2) {
    cmA = 0;
  }
  if (robAB == 4 && linkpair != 2) {
    cmA = cmB = 0; // Restrict to only T3A and T3B
  }

  // Finally convert chipmask to list of slaves
  nmcm = chipmaskToMCMlist(cmA, cmB, linkpair, mcmList, listSize);

  return nmcm;
}

short FeeParam::getRobAB(unsigned short robsel, unsigned short linkpair)
{
  // Converts the ROB part of the extended ALICE ID to robs

  if ((robsel & 0x8) != 0) { // 1000 .. direct ROB selection. Only one of the 8 ROBs are used.
    robsel = robsel & 7;
    if ((robsel % 2) == 0 && (robsel / 2) == linkpair) {
      return 1; // Even means A side (position 0,2,4,6)
    }
    if ((robsel % 2) == 1 && (robsel / 2) == linkpair) {
      return 2; // Odd  means B side (position 1,3,5,7)
    }
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
/*
void FeeParam::createORILookUpTable()
{
    int ori;
    for(int trdstack=0;trdstack<3;trdstack++)
    {
        for(int side=0;side<2;side++)
        {

            for(int trdlayer=5;trdlayer>=0;trdlayer++)
            {
                ori=trdstack*12  + (5-trdlayer + side*6) +trdlayer/6 + side;
                mAsideLUT[ori]= (trdstack<<8) + (trdlayer<<4) + side;           // A side LUT to map ORI to stack/layer/side
                if(ori==29) break;

            }
                if(ori==29) break;
        }
                if(ori==29) break;
    }
    for(int trdstack=4;trdstack>1;trdstack--)
    {
        for(int side=0;side<2;side++)
        {

            for(int trdlayer=5;trdlayer>=0;trdlayer++)
            {
                ori = (4-trdstack)*12  + (5-trdlayer + side*5) +trdlayer/6 + side;
                int newside;
                if(ori >=24) newside=1; else newside=side; // a hack as I am not typing this all out.
                mCsideLUT[ori]= (trdstack<<8) + (trdlayer<<4) + newside;           // A side LUT to map ORI to stack/layer/side
                if(ori==29) break;
            }
                if(ori==29) break;
        }
                if(ori==29) break;
    }
}
*/

void FeeParam::unpackORI(int link, int side, int& stack, int& layer, int& halfchamberside)
{
  // given the link 0-29 and the side A/C 0/1 return the stack layer and halfchamber side
  if (side == 0) {
    stack = link / 12;
    layer = 5 - link % 6;
    halfchamberside = (link / 6) % 2;
  } else {
    //c side
    stack = 4 - link / 12;
    layer = 5 - link % 6;
    halfchamberside = (link / 6) % 2;
    if (link > 23) {
      halfchamberside = 1;
    }
  }
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
  // Maximum available number is preset in mkMaxRAWversion
  //

  if (rawver >= 0 && rawver <= mkMaxRAWversion) {
    mRAWversion = rawver;
  } else {
    LOG(error) << "Raw version is out of range: " << rawver;
  }
}

/*
 * This was originally moved here from arrayADC, signalADC etc. We now longer use those classes
 * so removing this for now as its crashing.
 */
void FeeParam::fillPad2MCMLookUpTable()
{
  //
  // Initializes the Look Up Table to relate
  // pad numbering and mcm channel numbering
  //
  for (int mcm = 0; mcm < NMCMROBINCOL * 2; mcm++) {
    int lowerlimit = mcm * NCOLMCM;
    int upperlimit = NCOLMCM + mcm * NCOLMCM;
    int shiftposition = 1 + 3 * mcm;
    for (int index = lowerlimit; index < upperlimit; index++) {
      mLUTPadNumbering[index] = index + shiftposition;
    }
  }
}

int FeeParam::getDyCorrection(int det, int rob, int mcm) const
{
  // calculate the correction of the deflection
  // i.e. Lorentz angle and tilt correction (if active)

  int layer = det % NLAYER;

  float dyTilt = (mDriftLength * std::tan(mTiltingAngle[layer] * M_PI / 180.) *
                  getLocalZ(det, rob, mcm) * mInvX[layer]);

  // calculate Lorentz correction
  float dyCorr = -mOmegaTau * mDriftLength;

  if (mTiltCorr) {
    dyCorr += dyTilt; // add tilt correction
  }

  return (int)TMath::Nint(dyCorr * mScalePad * mInvWidthPad[layer]);
}

void FeeParam::getDyRange(int det, int rob, int mcm, int ch,
                          int& dyMinInt, int& dyMaxInt) const
{
  // calculate the deflection range in which tracklets are accepted

  dyMinInt = mDyMin;
  dyMaxInt = mDyMax;

  // deflection cut is considered for |B| > 0.1 T only
  if (std::abs(mMagField) < 0.1) {
    return;
  }

  float e = 0.30;

  float maxDeflTemp = getPerp(det, rob, mcm, ch) / 2. *             // Sekante/2 (cm)
                      (e * 1e-2 * std::abs(mMagField) * mInvPtMin); // 1/R (1/cm)

  float phi = getPhi(det, rob, mcm, ch);
  if (maxDeflTemp < std::cos(phi)) {
    float maxDeflAngle = std::asin(maxDeflTemp);

    float dyMin = (mDriftLength *
                   std::tan(phi - maxDeflAngle));

    dyMinInt = int(dyMin / mBinDy);
    // clipping to allowed range
    if (dyMinInt < mDyMin) {
      dyMinInt = mDyMin;
    } else if (dyMinInt > mDyMax) {
      dyMinInt = mDyMax;
    }

    float dyMax = (mDriftLength *
                   std::tan(phi + maxDeflAngle));

    dyMaxInt = int(dyMax / mBinDy);
    // clipping to allowed range
    if (dyMaxInt > mDyMax) {
      dyMaxInt = mDyMax;
    } else if (dyMaxInt < mDyMin) {
      dyMaxInt = mDyMin;
    }
  } else if (maxDeflTemp < 0.) {
    // this must not happen
    printf("Inconsistent calculation of sin(alpha): %f\n", maxDeflTemp);
  } else {
    // TRD is not reached at the given pt threshold
    // max range
  }

  if ((dyMaxInt - dyMinInt) <= 0) {
    LOG(debug) << "strange dy range: [" << dyMinInt << "," << dyMaxInt << "], using max range now";
    dyMaxInt = mDyMax;
    dyMinInt = mDyMin;
  }
}

float FeeParam::getElongation(int det, int rob, int mcm, int ch) const
{
  // calculate the ratio of the distance to the primary vertex and the
  // distance in x-direction for the given ADC channel

  int layer = det % NLAYER;

  float elongation = std::abs(getDist(det, rob, mcm, ch) * mInvX[layer]);

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
  if (mPidGainCorr == true) {
    Invgain = 1 / gain;
  }

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

  int layer = det % NLAYER;
  return mX[layer];
}

float FeeParam::getLocalY(int det, int rob, int mcm, int ch) const
{
  // get local y-position (r-phi) w.r.t. the chamber centre

  int layer = det % NLAYER;
  // calculate the pad position as in the TRAP
  float ypos = (-4 + 1 + (rob & 0x1) * 4 + (mcm & 0x3)) * 18 - ch - 0.5; // y position in bins of pad widths
  return ypos * mWidthPad[layer];
}

float FeeParam::getLocalZ(int det, int rob, int mcm) const
{
  // get local z-position w.r.t. to the chamber boundary

  int stack = (det % NCHAMBERPERSEC) / NLAYER;
  int layer = det % NLAYER;
  int row = (rob / 2) * 4 + mcm / 4;

  if (stack == 2) {
    if (row == 0) {
      return (mZrow[layer * NLAYER + stack] - 0.5 * mLengthOuterPadC0);
    } else if (row == 11) {
      return (mZrow[layer * NLAYER + stack] - 1.5 * mLengthOuterPadC0 - (row - 1) * mLengthInnerPadC0);
    } else {
      return (mZrow[layer * NLAYER + stack] - mLengthOuterPadC0 - (row - 0.5) * mLengthInnerPadC0);
    }
  } else {
    if (row == 0) {
      return (mZrow[layer * NLAYER + stack] - 0.5 * mLengthOuterPadC1[layer]);
    } else if (row == 15) {
      return (mZrow[layer * NLAYER + stack] - 1.5 * mLengthOuterPadC1[layer] - (row - 1) * mLengthInnerPadC1[layer]);
    } else {
      return (mZrow[layer * NLAYER + stack] - mLengthOuterPadC1[layer] - (row - 0.5) * mLengthInnerPadC1[layer]);
    }
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
