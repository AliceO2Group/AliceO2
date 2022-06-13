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

#ifndef ALICEO2_TRD_HELPERMETHODS_HH
#define ALICEO2_TRD_HELPERMETHODS_HH

#include "DataFormatsTRD/Constants.h"
#include <iostream>

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

  static void printSectorStackLayer(int det)
  {
    // for a given chamber number prints SECTOR_STACK_LAYER
    printf("%02i_%i_%i\n", det / constants::NCHAMBERPERSEC, (det % constants::NCHAMBERPERSEC) / constants::NLAYER, det % constants::NLAYER);
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

  static int getSector(int det)
  {
    return det / constants::NCHAMBERPERSEC;
  }

  static int getStack(int det)
  {
    return det % (constants::NSTACK * constants::NLAYER) / constants::NLAYER;
  }
  static int getLayer(int det)
  {
    return det % constants::NLAYER;
  }

  static int getDetector(int sector, int stack, int layer)
  {
    return (layer + stack * constants::NLAYER + sector * constants::NLAYER * constants::NSTACK);
  }

  static int getORIinSuperModule(int detector, int readoutboard)
  {
    //given a detector and readoutboard
    int ori = -1;
    int trdstack = HelperMethods::getStack(detector);
    int trdlayer = HelperMethods::getLayer(detector);
    int side = HelperMethods::getROBSide(readoutboard);
    //TODO ccdb lookup of detector/stack/layer/side for link id.
    bool aside = false;
    if (trdstack == 0 || trdstack == 1) {
      aside = true; //aside
    } else {
      if (trdstack != 2) {
        aside = false; //cside
      } else {
        if (side == 0) {
          aside = true; //stack
        } else {
          aside = false; //stack2, halfchamber 1
        }
      }
    }
    if (aside) {
      ori = trdstack * 12 + (5 - trdlayer + side * 5) + trdlayer / 6 + side; // <- that is correct for A side at least for now, probably not for very long LUT as that will come form CCDB ni anycase.
    } else {
      //cside
      int newside = side;
      if (trdstack == 2) {
        newside = 0; // the last part of C side CRU is a special case.
      }
      ori = (4 - trdstack) * 12 + (5 - trdlayer + newside * 5) + trdlayer / 6 + newside;
      ori += 30; // 30 to offset if from the a side link , 69 links in total
    }
    //see TDP for explanation of mapping TODO should probably come from CCDB
    return ori;
  }

  static int getLinkIDfromHCID(int hcid)
  {
    //return a number in range [0:29] for the link related to this hcid with in its respective CRU
    //lower 15 is endpoint 0 and upper 15 is endpoint 1
    //a side has 30, c side has 30 to give 60 links for a supermodule
    int detector = hcid / 2;
    int supermodule = hcid / 60;
    int chamberside = hcid % 2; // 0 for side 0, 1 for side 1;
    // now offset for supermodule (+60*supermodule);
    return HelperMethods::getORIinSuperModule(detector, chamberside) + 60 * supermodule; // it takes readoutboard but only cares if its odd or even hence side here.
  }

  inline static void swapByteOrder(unsigned int& word)
  {
    word = (word >> 24) |
           ((word << 8) & 0x00FF0000) |
           ((word >> 8) & 0x0000FF00) |
           (word << 24);
  }
  inline static unsigned int swapByteOrderreturn(unsigned int word)
  {
    //    word = (word >> 24) |
    //         ((word << 8) & 0x00FF0000) |
    //        ((word >> 8) & 0x0000FF00) |
    //       (word << 24);
    return word;
  }
};

} // namespace trd
} // namespace o2

#endif
