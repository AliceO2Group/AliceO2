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

  static void printSectorStackLayerSide(int hcid)
  {
    // for a given half-chamber number prints SECTOR_STACK_LAYER_side
    int det = hcid / 2;
    std::string side = (hcid % 2 == 0) ? "A" : "B";
    printf("%02i_%i_%i%s\n", det / constants::NCHAMBERPERSEC, (det % constants::NCHAMBERPERSEC) / constants::NLAYER, det % constants::NLAYER, side.c_str());
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

  static int getORIinSuperModule(int hcid)
  {
    // given a half chamber ID compute the link ID from [0..59]
    // where link ID [0..29] is for A-side CRU and [30..59] for C-side CRU
    int ori = -1;
    int stack = getStack(hcid / 2);
    int layer = getLayer(hcid / 2);
    int side = (hcid % 2 == 0) ? 0 : 1;
    bool isAside = false;
    if (stack < 2 || (stack == 2 && side == 0)) {
      isAside = true;
    }
    if (isAside) {
      ori = stack * constants::NLAYER * 2 + side * constants::NLAYER + 5 - layer;
    } else {
      // C-side
      ori = (4 - stack) * constants::NLAYER * 2 + side * constants::NLAYER + 5 - layer;
      if (stack == 2) {
        ori -= constants::NLAYER;
      }
      ori += constants::NLINKSPERCRU;
    }
    // TODO: put mapping into TDP and prepare for alternative mapping (CCDB?)
    return ori;
  }

  static int getHCIDFromLinkID(int link)
  {
    // link = halfcrulink [0..14] + halfcru [0..71] * constants::NLINKSPERHALFCRU (15) -> [0..1079]

    int sector = link / constants::NHCPERSEC;
    int linkSector = link % constants::NHCPERSEC;       // [0..59]
    int linkCRU = linkSector % constants::NLINKSPERCRU; // [0..29]
    int stack = linkCRU / (constants::NLAYER * 2);
    int layer = 5 - (linkCRU % constants::NLAYER);
    int side = (linkCRU / constants::NLAYER) % 2;
    if (linkSector >= constants::NLINKSPERCRU) {
      // C-side
      stack = 4 - stack;
      if (stack == 2) {
        side = 1;
      }
    }
    return sector * constants::NHCPERSEC + stack * constants::NLAYER * 2 + layer * 2 + side;
  }

  static int getLinkIDfromHCID(int hcid)
  {
    //return a number in range [0:29] for the link related to this hcid with in its respective CRU
    //lower 15 is endpoint 0 and upper 15 is endpoint 1
    //a side has 30, c side has 30 to give 60 links for a supermodule
    int sector = hcid / constants::NHCPERSEC;
    return getORIinSuperModule(hcid) + constants::NHCPERSEC * sector;
  }
};

} // namespace trd
} // namespace o2

#endif
