// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FITSimulation/Digitizer.h"

#include "TCanvas.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLeaf.h"
#include "TMath.h"
#include "TProfile2D.h"
#include "TRandom.h"
#include <algorithm>
#include <cassert>

using namespace o2::fit;

ClassImp(Digitizer);

void Digitizer::process(const std::vector<HitType>* hits, std::vector<Digit>* digits)
{

  std::cout << "@@@@  Digitizer::process " << std::endl;
  // hits array of FIT hits for a given simulated event
  mDigits = digits;
  Int_t timeframe = 0;
  Int_t bc = 0;
  Int_t ampthreshold = 100;
  Float_t lowTimeA = 10000, lowTimeC = 2500, highTimeA = 12500, highTimeC = 4500;
  Int_t mcp, trackID;
  Int_t amp[208], cfd[208];
  for (auto& hit : *hits) {
    // TODO: put timeframe counting/selection
    // if (timeframe == mTimeFrameCurrent) {
    timeframe = Int_t((mEventTime + hit.GetTime())); // to be replaced with uncalibrated time
    mcp = hit.GetDetectorID();
    if ((mcp < 96 && hit.GetTime() > lowTimeA && hit.GetTime() < highTimeA) ||
        (mcp > 95 && hit.GetTime() > lowTimeC && hit.GetTime() < highTimeC)) {
      cfd[mcp] += hit.GetTime();
      amp[mcp]++;
    }
    trackID = hit.GetTrackID();
  }
  // extract trackID

  Int_t ndigits = 0; // Number of digits added
  for (Int_t ipmt = 0; ipmt < 208; ipmt++) {
    if (amp[ipmt] > ampthreshold) {
      cfd[ipmt] = cfd[ipmt] / Float_t(amp[ipmt]);
      ndigits++;
      addDigit(Double_t(timeframe), ipmt, cfd[ipmt], amp[ipmt], bc, trackID);
      printf(" @@@@ mcp %i  Time %i amp %i track %i\n", ipmt, cfd[ipmt], amp[ipmt]);
      //}
    } // end loop over hits
  }
}

void Digitizer::addDigit(Double_t time, Int_t channel, Int_t cfd, Int_t amp, Int_t bc, Int_t trackID)
{
  // FIT digit requires: channel, time and number of photons

  Digit newdigit(time, channel, cfd, amp, bc);
  // Int_t nbc = Int_t(time * Geo::BC_TIME_INPS_INV); // time elapsed in number of bunch crossing
  //  Digit newdigit(time, channel, (time - Geo::BC_TIME_INPS * nbc) * Geo::NTDCBIN_PER_PS, tot * Geo::NTOTBIN_PER_NS,
  //  nbc);
  if (mMCTruthContainer) {
    auto ndigits = mDigits->size() - 1;
    o2::fit::MCLabel label(trackID, mEventID, mSrcID, cfd);
    mMCTruthContainer->addElement(ndigits, label);
  }

}

void Digitizer::initParameters()
{
  Int_t ampthreshold = 100;
  Float_t lowTimeA = 10000, lowTimeC = 2500, highTimeA = 12500, highTimeC = 4500;
  // murmur
}
/*
void Digitizer::printParameters()
{
  //murmur
}
*/
