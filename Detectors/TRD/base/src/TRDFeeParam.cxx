// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

////////////////////////////////////////////////////////////////////////////
//                                                                        //
//  TRD front end electronics parameters class                            //
//  Contains all FEE (MCM, TRAP, PASA) related                            //
//  parameters, constants, and mapping.                                   //
//                                                                        //
//  New release on 2007/08/17:                                            //
//   The default raw data version (now mRAWversion ) is set to 3          //
//   in the constructor because version 3 raw data read and write         //
//   are fully debugged.                                                  //
//                                                                        //
//  Author:                                                               //
//    Ken Oyama (oyama@physi.uni-heidelberg.de)                           //
//                                                                        //
//  many things now configured by AliTRDtrapConfig reflecting             //
//  the real memory structure of the TRAP (Jochen)                        //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include <TGeoManager.h>
#include <TGeoPhysicalNode.h>
#include <TMath.h>
#include <TVirtualMC.h>
#include <FairLogger.h>

#include "DetectorsBase/GeometryManager.h"
#include "TRDBase/TRDGeometry.h"
#include "TRDBase/TRDPadPlane.h"
#include "TRDBase/TRDFeeParam.h"
#include "TRDBase/TRDCommonParam.h"

using namespace o2::trd;

//_____________________________________________________________________________

TRDFeeParam* TRDFeeParam::mgInstance = nullptr;
Bool_t TRDFeeParam::mgTerminated = kFALSE;
Bool_t TRDFeeParam::mgTracklet = kTRUE;
Bool_t TRDFeeParam::mgRejectMultipleTracklets = kFALSE;
Bool_t TRDFeeParam::mgUseMisalignCorr = kFALSE;
Bool_t TRDFeeParam::mgUseTimeOffset = kFALSE;

//_____________________________________________________________________________
TRDFeeParam* TRDFeeParam::Instance()
{
  //
  // Instance constructor
  //

  if (mgTerminated != kFALSE) {
    return nullptr;
  }

  if (mgInstance == nullptr) {
    mgInstance = new TRDFeeParam();
  }

  return mgInstance;
}

//_____________________________________________________________________________
void TRDFeeParam::Terminate()
{
  //
  // Terminate the class and release memory
  //

  mgTerminated = kTRUE;

  if (mgInstance != nullptr) {
    delete mgInstance;
    mgInstance = 0;
  }
}

//_____________________________________________________________________________
TRDFeeParam::TRDFeeParam()
{
  //
  // Default constructor
  //

  mCP = TRDCommonParam::Instance();
}

//_____________________________________________________________________________
TRDFeeParam::TRDFeeParam(TRootIoCtor*)
{
  //
  // IO constructor
  //
}

//_____________________________________________________________________________
TRDFeeParam::TRDFeeParam(const TRDFeeParam& p)
{
  //
  // TRDFeeParam copy constructor
  //
  mRAWversion=p.mRAWversion;
  mCP=p.mCP;
}

//_____________________________________________________________________________
TRDFeeParam::~TRDFeeParam() = default;

//_____________________________________________________________________________
TRDFeeParam& TRDFeeParam::operator=(const TRDFeeParam& p)
{
  //
  // Assignment operator
  //

  if (this != &p) {
    ((TRDFeeParam&)p).Copy(*this);
  }

  return *this;
}

//_____________________________________________________________________________
void TRDFeeParam::Copy(TRDFeeParam& p) const
{
  //
  // Copy function
  //

  ((TRDFeeParam&)p).mCP = mCP;
  ((TRDFeeParam&)p).mRAWversion = mRAWversion;
}

//_____________________________________________________________________________
Int_t TRDFeeParam::getPadRowFromMCM(Int_t irob, Int_t imcm) const
{
  //
  // Return on which pad row this mcm sits
  //

  return mgkNmcmRobInRow * (irob / 2) + imcm / mgkNmcmRobInCol;
}

//_____________________________________________________________________________
Int_t TRDFeeParam::getPadColFromADC(Int_t irob, Int_t imcm, Int_t iadc) const
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
  Int_t mcmcol = imcm % mgkNmcmRobInCol + getRobSide(irob) * mgkNmcmRobInCol; // MCM column number on ROC [0..7]
  Int_t padcol = mcmcol * mgkNcolMcm + mgkNcolMcm + 1 - iadc;
  if (padcol < 0 || padcol >= mgkNcol)
    return -1; // this is commented because of reason above OK

  return padcol;
}

//_____________________________________________________________________________
Int_t TRDFeeParam::getExtendedPadColFromADC(Int_t irob, Int_t imcm, Int_t iadc) const
{
  //
  // Return which pad coresponds to the extended digit container pad numbering
  // Extended digit container is designed to store all pad data including shared pad,
  // so we have to introduce new virtual pad numbering scheme for this purpose.
  //

  if (iadc < 0 || iadc > mgkNadcMcm)
    return -100;
  Int_t mcmcol = imcm % mgkNmcmRobInCol + getRobSide(irob) * mgkNmcmRobInCol; // MCM column number on ROC [0..7]
  Int_t padcol = mcmcol * mgkNadcMcm + mgkNcolMcm + 2 - iadc;

  return padcol;
}

//_____________________________________________________________________________
Int_t TRDFeeParam::getMCMfromPad(Int_t irow, Int_t icol) const
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
Int_t TRDFeeParam::getMCMfromSharedPad(Int_t irow, Int_t icol) const
{
  //
  // Return on which MCM this pad is directry connected.
  // Return -1 for error.
  //

  if (irow < 0 || icol < 0 || irow > mgkNrowC1 || icol > mgkNcol + 8 * 3)
    return -1;

  Int_t adc = 20 - (icol % 18) - 1;
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
Int_t TRDFeeParam::getROBfromPad(Int_t irow, Int_t icol) const
{
  //
  // Return on which rob this pad is
  //

  return (irow / mgkNmcmRobInRow) * 2 + getColSide(icol);
}

//_____________________________________________________________________________
Int_t TRDFeeParam::getROBfromSharedPad(Int_t irow, Int_t icol) const
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
Int_t TRDFeeParam::getRobSide(Int_t irob) const
{
  //
  // Return on which side this rob sits (A side = 0, B side = 1)
  //

  if (irob < 0 || irob >= mgkNrobC1)
    return -1;

  return irob % 2;
}

//_____________________________________________________________________________
Int_t TRDFeeParam::getColSide(Int_t icol) const
{
  //
  // Return on which side this column sits (A side = 0, B side = 1)
  //

  if (icol < 0 || icol >= mgkNcol)
    return -1;

  return icol / (mgkNcol / 2);
}

UInt_t TRDFeeParam::AliToExtAli(Int_t rob, Int_t aliid)
{
  if (aliid != 127)
    return ((1 << 10) | (rob << 7) | aliid);

  return 127;
}

Int_t TRDFeeParam::ExtAliToAli(UInt_t dest, UShort_t linkpair, UShort_t rocType, Int_t* mcmList, Int_t listSize)
{
  // Converts an extended ALICE ID which identifies a single MCM or a group of MCMs to
  // the corresponding list of MCMs. Only broadcasts (127) are encoded as 127
  // The return value is the number of MCMs in the list

  mcmList[0] = -1;

  Short_t nmcm = 0;
  UInt_t mcm, rob, robAB;
  UInt_t cmA = 0, cmB = 0; // Chipmask for each A and B side

  // Default chipmask for 4 linkpairs (each bit correponds each alice-mcm)
  static const UInt_t gkChipmaskDefLp[4] = { 0x1FFFF, 0x1FFFF, 0x3FFFF, 0x1FFFF };

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
  nmcm = ChipmaskToMCMlist(cmA, cmB, linkpair, mcmList, listSize);

  return nmcm;
}

Short_t TRDFeeParam::getRobAB(UShort_t robsel, UShort_t linkpair)
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

Short_t TRDFeeParam::ChipmaskToMCMlist(UInt_t cmA, UInt_t cmB, UShort_t linkpair, Int_t* mcmList, Int_t listSize)
{
  // Converts the chipmask to a list of MCMs

  Short_t nmcm = 0;
  Short_t i;
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
void TRDFeeParam::setRAWversion(Int_t rawver)
{
  //
  // Set raw data version (major number only)
  // Maximum available number is preset in mgkMaxRAWversion
  //

  if (rawver >= 0 && rawver <= mgkMaxRAWversion) {
    mRAWversion = rawver;
  } else {
    LOG(ERROR) << "Raw version is out of range: " << rawver;
  }
}
