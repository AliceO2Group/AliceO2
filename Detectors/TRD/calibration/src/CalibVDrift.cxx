// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   CalibVDrift.cxx
/// \author Ole Schmidt, ole.schmidt@cern.ch

#include "TFile.h"
#include "TH2F.h"

#include <fairlogger/Logger.h>

#include "TRDCalibration/CalibVDrift.h"

using namespace o2::trd;
using namespace o2::trd::constants;

void CalibVDrift::process()
{
  LOG(info) << "Started processing for vDrift calibration";

  for (int iDet = 0; iDet < constants::MAXCHAMBER; ++iDet) {
    for (int iBin = 0; iBin < constants::NBINSANGLEDIFF; ++iBin) { // note: iBin = constants::NBINSANGLEDIFF - 1 is under-/overflow bin
      short nEntries = mAngleDiffCounters[iDet * constants::NBINSANGLEDIFF + iBin];
      float angleDiffSum = mAngleDiffSums[iDet * constants::NBINSANGLEDIFF + iBin];
      if (nEntries > 0) {
        LOGF(INFO, "Found %i entrie(s) in chamber %i, bin %i. Average angular deviation: %f", nEntries, iDet, iBin, angleDiffSum / nEntries);
      }
    }
  }

  /*
  // as an example I loop over the input, create a histogram and write it to a file
  auto fOut = TFile::Open("trdcalibdummy.root", "recreate");
  auto hXY = std::make_unique<TH2F>("histDummy", "foo", 100, -60, 60, 100, 250, 400); // xy distribution of TRD space points
  for (int i = 0; i < mAngulerDeviationProf.size(); ++i) {
    hXY->Fill(mAngulerDeviationProf[i].mX[0], mAngulerDeviationProf[i].mR);
  }
  fOut->cd();
  hXY->Write();
  hXY.reset(); // delete the histogram before closing the output file
  fOut->Close();
  */
}
