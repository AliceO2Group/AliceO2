// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_HELPERMETHODS_HH
#define ALICEO2_TRD_HELPERMETHODS_HH

#include "DataFormatsTRD/Constants.h"

namespace o2
{
namespace trd
{

struct HelperMethods {
  static int getROBfromPad(int irow, int icol)
  {
    return (irow / constants::NMCMROBINROW) * 2 + getColSide(icol);
  }

  static int getMCMfromPad(int irow, int icol)
  {
    if (irow < 0 || icol < 0 || irow > constants::NROWC1 || icol > constants::NCOLUMN) {
      return -1;
    }
    return (icol % (constants::NCOLUMN / 2)) / constants::NCOLMCM + constants::NMCMROBINCOL * (irow % constants::NMCMROBINROW);
  }

  static int getColSide(int icol)
  {
    if (icol < 0 || icol >= constants::NCOLUMN) {
      return -1;
    }

    return icol / (constants::NCOLUMN / 2);
  }

  static int getPadRowFromMCM(int irob, int imcm)
  {
    return constants::NMCMROBINROW * (irob / 2) + imcm / constants::NMCMROBINCOL;
  }

  static int getPadColFromADC(int irob, int imcm, int iadc)
  {
    if (iadc < 0 || iadc > constants::NADCMCM) {
      return -100;
    }
    int mcmcol = imcm % constants::NMCMROBINCOL + getROBSide(irob) * constants::NMCMROBINCOL; // MCM column number on ROC [0..7]
    int padcol = mcmcol * constants::NCOLMCM + constants::NCOLMCM + 1 - iadc;
    if (padcol < 0 || padcol >= constants::NCOLUMN) {
      return -1; // this is commented because of reason above OK
    }
    return padcol;
  }

  static int getROBSide(int irob)
  {
    if (irob < 0 || irob >= constants::NROBC1) {
      return -1;
    }
    return irob % 2;
  }
};

} // namespace trd
} // namespace o2

#endif
