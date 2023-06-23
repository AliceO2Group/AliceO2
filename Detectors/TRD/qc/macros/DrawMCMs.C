
/// This macro demonstrates how to use the o2::trd::RawDataManager and
/// o2::trd:MCMDisplay to visualize TRD digits and tracklets.
/// You probably want to copy it and adjust it for your use case.
void DrawMCMs(std::string dirname = ".")
{
  // instantiate the class that handles all the data access
  auto dman = o2::trd::RawDataManager(dirname);
  cout << dman.DescribeFiles() << endl;

  // Here we can specify how many plots we want for each MCMs. The default is 0.
  std::map<array<int,3>, int> ndraw;
  ndraw[{  13, 6,  1 }] = 1;
  ndraw[{  21, 2, 13 }] = 1;
  ndraw[{  47, 0,  2 }] = 1;
  ndraw[{  52, 3, 12 }] = 1;
  ndraw[{ 537, 5, 11 }] = 1;

  // --------------------------------------------------------------------
  // loop over timeframes
  while (dman.NextTimeFrame()) {
    cout << dman.DescribeTimeFrame() << endl;

    // loop over events
    while (dman.NextEvent()) {

      auto ev = dman.GetEvent();

      // skip events without digits immediately
      if (ev.digits.length() == 0) {
        continue;
      }
      // if (ev.digits.length() > 800000) { continue; }

      cout << dman.DescribeEvent() << endl;

      for (auto& mcm : dman.GetEvent().IterateBy<o2::trd::MCM_ID>()) {
        // skip MCMs without digits
        if (mcm.digits.length()==0) {
          continue;
        }

        // we skipped MCMs without digits, so we can use the first one to find out where we are
        auto firstdigit = *mcm.digits.begin();
        array<int,3> key = { firstdigit.getDetector(), firstdigit.getROB(), firstdigit.getMCM() };

        // only draw MCMs that have not reached their desired count yet
        if (ndraw[key] == 0) {
          continue;
        }

        // the actual drawing
        o2::trd::MCMDisplay disp(mcm);
        // disp.Draw();
        disp.DrawDigits("colz");
        disp.DrawDigits("text,same");
        disp.DrawTracklets();

        // decrement the number of plots we need for this MCM
        --ndraw[key];
      }

    } // event/trigger record loop
  }   // time frame loop

}
