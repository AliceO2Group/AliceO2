#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TOFReconstruction/EventTimeMaker.h"
#endif

using namespace o2::tof;
void macroEvTime(bool removebias = true)
{

  TH1F* htest = new TH1F("htest", ";N#sigma (t0_{reco} - t0_{true})", 100, -10, 10);
  TH1F* htest2 = new TH1F("htest2", ";t0_{reco} - t0_{true} ps", 100, -100, 100);

  TH1F* hkaon = new TH1F("hkaon", ";N#sigma^{K}", 100, -10, 10);
  TH1F* hkaonT = new TH1F("hkaonT", ";N#sigma^{K}", 100, -10, 10);
  hkaonT->SetLineColor(2);

  std::vector<eventTimeTrackTest> tracks;
  for (int i = 0; i < 1000; i++) {
    tracks.clear();
    generateEvTimeTracks(tracks, 5);
    auto evtime = evTimeMaker<std::vector<eventTimeTrackTest>, eventTimeTrackTest, filterDummy>(tracks);
    //    Printf("Ev time %f +-%f", evtime.eventTime, evtime.eventTimeError);
    htest->Fill(evtime.eventTime / evtime.eventTimeError);
    htest2->Fill(evtime.eventTime);

    int nt = 0;
    for (auto track : tracks) {

      if (track.p() > 2) {
        nt++;
        continue;
      }

      float et = evtime.eventTime;
      float erret = evtime.eventTimeError;

      if (removebias) {
        float sumw = 1. / erret / erret;
        et *= sumw;
        et -= evtime.weights[nt] * evtime.tracktime[nt];
        sumw -= evtime.weights[nt];
        et /= sumw;
        erret = sqrt(1. / sumw);
      }

      nt++;

      float res = sqrt(100 * 100 + erret * erret);
      hkaon->Fill((track.tofSignal() - track.tofExpSignalKa() - et) / res);

      if (track.masshypo() == 1) {
        hkaonT->Fill((track.tofSignal() - track.tofExpSignalKa() - et) / res);
      }
    }
  }
  htest->Draw();
  new TCanvas;
  htest2->Draw();
  new TCanvas;
  hkaon->Draw();
  hkaonT->Draw("SAME");
}
