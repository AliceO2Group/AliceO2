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

#include "PWGDQCore/MCProng.h"

#include <cmath>
#include <iostream>

ClassImp(MCProng);

//________________________________________________________________________________________________________________
MCProng::MCProng() : fNGenerations(0),
                     fPDGcodes({}),
                     fCheckBothCharges({}),
                     fExcludePDG({}),
                     fSourceBits({}),
                     fExcludeSource({}),
                     fUseANDonSourceBitMap({})
{
}

//________________________________________________________________________________________________________________
MCProng::MCProng(int n) : fNGenerations(n)
{
  fPDGcodes.reserve(n);
  fCheckBothCharges.reserve(n);
  fExcludePDG.reserve(n);
  fSourceBits.reserve(n);
  fExcludeSource.reserve(n);
  fUseANDonSourceBitMap.reserve(n);
  for (int i = 0; i < n; i++) {
    fPDGcodes.push_back(kPDGCodeNotAssigned);
    fCheckBothCharges.push_back(false);
    fExcludePDG.push_back(false);
    fSourceBits.push_back(0);
    fExcludeSource.push_back(0);
    fUseANDonSourceBitMap.push_back(true);
  }
}

//________________________________________________________________________________________________________________
MCProng::MCProng(int n, const std::vector<int> pdgs, const std::vector<bool> checkBothCharges, const std::vector<bool> excludePDG,
                 const std::vector<uint64_t> sourceBits, const std::vector<uint64_t> excludeSource,
                 const std::vector<bool> useANDonSourceBitMap) : fPDGcodes(pdgs),
                                                                 fCheckBothCharges(checkBothCharges),
                                                                 fExcludePDG(excludePDG),
                                                                 fSourceBits(sourceBits),
                                                                 fExcludeSource(excludeSource),
                                                                 fUseANDonSourceBitMap(useANDonSourceBitMap),
                                                                 fNGenerations(n){};

//________________________________________________________________________________________________________________
void MCProng::SetPDGcode(int generation, int code, bool checkBothCharges /*= false*/, bool exclude /*= false*/)
{
  if (generation < 0 || generation >= fNGenerations) {
    return;
  }
  fPDGcodes[generation] = code;
  fCheckBothCharges[generation] = checkBothCharges;
  fExcludePDG[generation] = exclude;
}

//________________________________________________________________________________________________________________
void MCProng::SetSources(int generation, uint64_t bits, uint64_t exclude /*=0*/, bool useANDonSourceBits /*=true*/)
{
  if (generation < 0 || generation >= fNGenerations) {
    return;
  }
  fSourceBits[generation] = bits;
  fExcludeSource[generation] = exclude;
  fUseANDonSourceBitMap[generation] = useANDonSourceBits;
}

//________________________________________________________________________________________________________________
void MCProng::SetSourceBit(int generation, int sourceBit, bool exclude /*=false*/)
{
  if (generation < 0 || generation >= fNGenerations) {
    return;
  }
  fSourceBits[generation] |= (uint64_t(1) << sourceBit);
  if (exclude) {
    fExcludeSource[generation] |= (uint64_t(1) << sourceBit);
  }
}

//________________________________________________________________________________________________________________
void MCProng::SetUseANDonSourceBits(int generation, bool option /*=true*/)
{
  if (generation < 0 || generation >= fNGenerations) {
    return;
  }
  fUseANDonSourceBitMap[generation] = option;
}

//________________________________________________________________________________________________________________
void MCProng::Print() const
{
  for (int i = 0; i < fNGenerations; i++) {
    std::cout << "Generation #" << i << " PDGcode(" << fPDGcodes[i] << ") CheckBothCharges(" << fCheckBothCharges[i]
              << ") ExcludePDG(" << fExcludePDG[i] << ")  SourceBits(" << fSourceBits[i] << ") ExcludeSource(" << fExcludeSource[i]
              << ") UseANDonSource(" << fUseANDonSourceBitMap[i] << ")" << std::endl;
  }
}

//________________________________________________________________________________________________________________
bool MCProng::TestPDG(int i, int pdgCode) const
{
  if (i < 0 || i >= fNGenerations) { // TODO: need an error message
    return false;
  }
  if (!ComparePDG(pdgCode, fPDGcodes[i], fCheckBothCharges[i], fExcludePDG[i])) {
    return false;
  }
  return true;
}

//________________________________________________________________________________________________________________
bool MCProng::ComparePDG(int pdg, int prongPDG, bool checkBothCharges, bool exclude) const
{
  //
  // test whether the particle pdg ("pdg") matches this MC signal prong ("prongPDG")
  //
  bool decision = true;
  // compute the absolute value of the PDG code
  int absPDG = std::abs(pdg);
  switch (std::abs(prongPDG)) {
    case kPDGCodeNotAssigned:
      // PDG not required (any code will do fine)
      break;
    case 100: // light flavoured mesons
      if (checkBothCharges) {
        decision = absPDG >= 100 && absPDG <= 199;
      } else {
        decision = (prongPDG > 0 ? pdg >= 100 && pdg <= 199 : pdg >= -199 && pdg <= -100);
      }
      break;
    case 1000: // light flavoured baryons
      if (checkBothCharges) {
        decision = absPDG >= 1000 && absPDG <= 1999;
      } else {
        decision = (prongPDG > 0 ? pdg >= 1000 && pdg <= 1999 : pdg >= -1999 && pdg <= -1000);
      }
      break;
    case 200: // light flavoured mesons
      if (checkBothCharges) {
        decision = absPDG >= 200 && absPDG <= 299;
      } else {
        decision = (prongPDG > 0 ? pdg >= 200 && pdg <= 299 : pdg >= -299 && pdg <= -200);
      }
      break;
    case 2000: // light flavoured baryons
      if (checkBothCharges) {
        decision = absPDG >= 2000 && absPDG <= 2999;
      } else {
        decision = (prongPDG > 0 ? pdg >= 2000 && pdg <= 2999 : pdg >= -2999 && pdg <= -2000);
      }
      break;
    case 300: // all strange mesons
      if (checkBothCharges) {
        decision = absPDG >= 300 && absPDG <= 399;
      } else {
        decision = (prongPDG > 0 ? pdg >= 300 && pdg <= 399 : pdg >= -399 && pdg <= -300);
      }
      break;
    case 3000: // all strange baryons
      if (checkBothCharges) {
        decision = absPDG >= 3000 && absPDG <= 3999;
      } else {
        decision = (prongPDG > 0 ? pdg >= 3000 && pdg <= 3999 : pdg >= -3999 && pdg <= -3000);
      }
      break;
    case 400: // all charmed mesons
      if (checkBothCharges) {
        decision = absPDG >= 400 && absPDG <= 499;
      } else {
        decision = (prongPDG > 0 ? pdg >= 400 && pdg <= 499 : pdg >= -499 && pdg <= -400);
      }
      break;
    case 401: // open charm mesons
      if (checkBothCharges) {
        decision = absPDG >= 400 && absPDG <= 439;
      } else {
        decision = (prongPDG > 0 ? pdg >= 400 && pdg <= 439 : pdg >= -439 && pdg <= -400);
      }
      break;
    case 402: // open charm mesons and baryons together
      if (checkBothCharges) {
        decision = (absPDG >= 400 && absPDG <= 439) || (absPDG >= 4000 && absPDG <= 4399);
      } else {
        if (prongPDG > 0) {
          decision = (pdg >= 400 && pdg <= 439) || (pdg >= 4000 && pdg <= 4399);
        }
        if (prongPDG < 0) {
          decision = (pdg >= -439 && pdg <= -400) || (pdg >= -4399 && pdg <= -4000);
        }
      }
      break;
    case 403: // all charm hadrons
      if (checkBothCharges) {
        decision = (absPDG >= 400 && absPDG <= 499) || (absPDG >= 4000 && absPDG <= 4999);
      } else {
        if (prongPDG > 0) {
          decision = (pdg >= 400 && pdg <= 499) || (pdg >= 4000 && pdg <= 4999);
        }
        if (prongPDG < 0) {
          decision = (pdg >= -499 && pdg <= -400) || (pdg >= -4999 && pdg <= -4000);
        }
      }
      break;
    case 404: // charged open charmed mesons w/o s-quark
      if (checkBothCharges) {
        decision = (absPDG >= 410 && absPDG <= 419);
      } else {
        decision = (prongPDG > 0 ? pdg >= 410 && pdg <= 419 : pdg >= -419 && pdg <= -410);
      }
      break;
    case 405: // neutral open charmed mesons
      if (checkBothCharges) {
        decision = absPDG >= 420 && absPDG <= 429;
      } else {
        decision = (prongPDG > 0 ? pdg >= 420 && pdg <= 429 : pdg >= -429 && pdg <= -420);
      }
      break;
    case 406: // charged open charmed mesons with s-quark
      if (checkBothCharges) {
        decision = (absPDG >= 430 && absPDG <= 439);
      } else {
        decision = (prongPDG > 0 ? pdg >= 430 && pdg <= 439 : pdg >= -439 && pdg <= -430);
      }
      break;
    case 4000: // all charmed baryons
      if (checkBothCharges) {
        decision = absPDG >= 4000 && absPDG <= 4999;
      } else {
        decision = (prongPDG > 0 ? pdg >= 4000 && pdg <= 4999 : pdg >= -4999 && pdg <= -4000);
      }
      break;
    case 4001: // open charm baryons
      if (checkBothCharges) {
        decision = absPDG >= 4000 && absPDG <= 4399;
      } else {
        decision = (prongPDG > 0 ? pdg >= 4000 && pdg <= 4399 : pdg >= -4399 && pdg <= -4000);
      }
      break;
    case 500: // all beauty mesons
      if (checkBothCharges) {
        decision = absPDG >= 500 && absPDG <= 599;
      } else {
        decision = (prongPDG > 0 ? pdg >= 500 && pdg <= 599 : pdg >= -599 && pdg <= -500);
      }
      break;
    case 501: // open beauty mesons
      if (checkBothCharges) {
        decision = absPDG >= 500 && absPDG <= 549;
      } else {
        decision = (prongPDG > 0 ? pdg >= 500 && pdg <= 549 : pdg >= -549 && pdg <= -500);
      }
      break;
    case 502: // open beauty mesons and baryons
      if (checkBothCharges) {
        decision = (absPDG >= 500 && absPDG <= 549) || (absPDG >= 5000 && absPDG <= 5499);
      } else {
        if (prongPDG > 0) {
          decision = (pdg >= 500 && pdg <= 549) || (pdg >= 5000 && pdg <= 5499);
        }
        if (prongPDG < 0) {
          decision = (pdg >= -549 && pdg <= -500) || (pdg >= -5499 && pdg <= -5000);
        }
      }
      break;
    case 503: // all beauty hadrons
      if (checkBothCharges) {
        decision = (absPDG >= 500 && absPDG <= 599) || (absPDG >= 5000 && absPDG <= 5999);
      } else {
        if (prongPDG > 0) {
          decision = (pdg >= 500 && pdg <= 599) || (pdg >= 5000 && pdg <= 5999);
        }
        if (prongPDG < 0) {
          decision = (pdg >= -599 && pdg <= -500) || (pdg >= -5999 && pdg <= -5000);
        }
      }
      break;
    case 504: // neutral open beauty mesons w/o s-quark
      if (checkBothCharges) {
        decision = (absPDG >= 510 && absPDG <= 519);
      } else {
        decision = (prongPDG > 0 ? pdg >= 510 && pdg <= 519 : pdg >= -519 && pdg <= -510);
      }
      break;
    case 505: // charged open beauty mesons
      if (checkBothCharges) {
        decision = absPDG >= 520 && absPDG <= 529;
      } else {
        decision = (prongPDG > 0 ? pdg >= 520 && pdg <= 529 : pdg >= -529 && pdg <= -520);
      }
      break;
    case 506: // charged open beauty mesons with s-quark
      if (checkBothCharges) {
        decision = (absPDG >= 530 && absPDG <= 539);
      } else {
        decision = (prongPDG > 0 ? pdg >= 530 && pdg <= 539 : pdg >= -539 && pdg <= -530);
      }
      break;
    case 5000: // all beauty baryons
      if (checkBothCharges) {
        decision = absPDG >= 5000 && absPDG <= 5999;
      } else {
        decision = (prongPDG > 0 ? pdg >= 5000 && pdg <= 5999 : pdg >= -5999 && pdg <= -5000);
      }
      break;
    case 5001: // open beauty baryons
      if (checkBothCharges) {
        decision = absPDG >= 5000 && absPDG <= 5499;
      } else {
        decision = (prongPDG > 0 ? pdg >= 5000 && pdg <= 5499 : pdg >= -5499 && pdg <= -5000);
      }
      break;
    case 902: // open charm,beauty  mesons and baryons together
      if (checkBothCharges) {
        decision = (absPDG >= 400 && absPDG <= 439) || (absPDG >= 4000 && absPDG <= 4399) ||
                   (absPDG >= 500 && absPDG <= 549) || (absPDG >= 5000 && absPDG <= 5499);
      } else {
        if (prongPDG > 0) {
          decision = (pdg >= 400 && pdg <= 439) || (pdg >= 4000 && pdg <= 4399) ||
                     (pdg >= 500 && pdg <= 549) || (pdg >= 5000 && pdg <= 5499);
        }
        if (prongPDG < 0) {
          decision = (pdg >= -439 && pdg <= -400) || (pdg >= -4399 && pdg <= -4000) ||
                     (pdg >= -549 && pdg <= -500) || (pdg >= -5499 && pdg <= -5000);
        }
      }
      break;
    case 903: // all hadrons in the code range 100-599, 1000-5999
      if (checkBothCharges) {
        decision = (absPDG >= 100 && absPDG <= 599) || (absPDG >= 1000 && absPDG <= 5999);
      } else {
        if (prongPDG > 0) {
          decision = (pdg >= 100 && pdg <= 599) || (pdg >= 1000 && pdg <= 5999);
        }
        if (prongPDG < 0) {
          decision = (pdg >= -599 && pdg <= -100) || (pdg >= -5999 && pdg <= -1000);
        }
      }
      break;
    default: // all explicit PDG code cases
      if (checkBothCharges) {
        decision = (std::abs(prongPDG) == absPDG);
      } else {
        decision = (prongPDG == pdg);
      }
  }

  // if exclude is specified, then reverse decision
  if (prongPDG && exclude) {
    decision = !decision;
  }
  return decision;
};
