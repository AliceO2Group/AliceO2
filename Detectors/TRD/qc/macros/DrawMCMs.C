// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// This macro demonstrates how to use the o2::trd::RawDataManager and
/// o2::trd:MCMDisplay to visualize TRD digits and tracklets.
/// You probably want to copy it and adjust it for your use case.
void DrawMCMs(std::string dirname = ".")
{
  // instantiate the class that handles all the data access
  auto dman = o2::trd::RawDataManager(dirname);
  cout << dman.describeFiles() << endl;

  // Here we can specify how many plots we want for each MCMs. The default is 0.
  std::map<array<int, 3>, int> ndraw;
  ndraw[{13, 6, 1}] = 1;
  ndraw[{21, 2, 13}] = 1;
  ndraw[{47, 0, 2}] = 1;
  ndraw[{52, 3, 12}] = 1;
  ndraw[{537, 5, 11}] = 1;

  // Set a number of plots you want for any MCM, not specified above
  int ndrawany = 10;

  // --------------------------------------------------------------------
  // loop over timeframes
  while (dman.nextTimeFrame()) {
    cout << dman.describeTimeFrame() << endl;

    // loop over events
    while (dman.nextEvent()) {

      auto ev = dman.getEvent();

      // skip events without digits immediately
      if (ev.digits.size() == 0) {
        continue;
      }
      // if (ev.digits.length() > 800000) { continue; }

      cout << dman.describeEvent() << endl;

      for (auto& mcm : dman.getEvent().iterateByMCM()) {
        // skip MCMs without digits
        if (mcm.digits.size() == 0) {
          continue;
        }
        if (mcm.tracklets.size() == 0) {
          continue;
        }

        // we skipped MCMs without digits, so we can use the first digit to find out where we are
        auto firstdigit = *mcm.digits.begin();
        array<int, 3> key = {firstdigit.getDetector(), firstdigit.getROB(), firstdigit.getMCM()};

        // only draw MCMs that have not reached their desired count yet
        if (ndraw[key] > 0) {
          // decrement the number of plots we need for this MCM
          --ndraw[key];
        } else if (ndrawany > 0) {
          --ndrawany;
        } else {
          continue;
        }

        cout << "==============================================================" << endl;
        // for (auto hit : mcm.hits) { cout << hit << endl; }
        for (auto digit : mcm.digits) {
          cout << digit << endl;
        }
        for (auto tracklet : mcm.tracklets) {
          cout << tracklet << endl;
        }

        // the actual drawing
        o2::trd::MCMDisplay disp(mcm);
        // disp.Draw();
        disp.drawDigits("colz");
        disp.drawDigits("text,same");
        disp.drawClusters();
        disp.drawTracklets();

        disp.drawHits();
        disp.drawMCTrackSegments();
      }

    } // event/trigger record loop
  }   // time frame loop
}
