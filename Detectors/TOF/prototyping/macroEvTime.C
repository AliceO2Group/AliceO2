#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TOFBase/EventTimeMaker.h"
#endif

///
/// \brief Simple macro to test the TOF event time maker routine
///

using namespace o2::tof;
void macroEvTime(bool removebias = true,
                 int nEvents = 1000,
                 int nTracks = 5)
{

  TH1F* htest = new TH1F("htest", ";N#sigma (t0_{reco} - t0_{true})", 100, -10, 10);
  TH1F* htest2 = new TH1F("htest2", ";t0_{reco} - t0_{true} ps", 100, -100, 100);
  TH1F* hdiff = new TH1F("hdiff", ";t0 - t0 (central)", 100, -100, 100);
  TH1F* herrdiff = new TH1F("herrdiff", ";#sigmat0 - #sigmat0 (central)", 100, -100, 100);

  TH1F* hkaon = new TH1F("hkaon", ";N#sigma^{K}", 100, -10, 10);
  TH1F* hkaonT = new TH1F("hkaonT", ";N#sigma^{K}", 100, -10, 10);
  hkaonT->SetLineColor(2);

  std::vector<eventTimeTrackTest> tracks;
  for (int i = 0; i < nEvents; i++) {
    tracks.clear();
    generateEvTimeTracks(tracks, nTracks);
    auto evtime = evTimeMaker<std::vector<eventTimeTrackTest>, eventTimeTrackTest, filterDummy>(tracks);
    //    Printf("Ev time %f +-%f", evtime.mEventTime, evtime.mEventTimeError);
    htest->Fill(evtime.mEventTime / evtime.mEventTimeError);
    htest2->Fill(evtime.mEventTime);

    int nt = 0;
    for (auto track : tracks) {
      float et = evtime.mEventTime;
      float erret = evtime.mEventTimeError;

      float et2 = et;
      float erret2 = erret;
      if (removebias) {
        evtime.removeBias<eventTimeTrackTest, filterDummy>(track, nt, et2, erret2, -2);
        nt--;
      }

      if (!filterDummy(track)) { // Only tracks that were used for TOF-T0
        nt++;
        continue;
      }

      if (removebias) {
        float sumw = 1. / erret / erret;
        et *= sumw;
        et -= evtime.mWeights[nt] * evtime.mTrackTimes[nt];
        sumw -= evtime.mWeights[nt];
        et /= sumw;
        erret = sqrt(1. / sumw);
      }

      nt++;

      hdiff->Fill(et - et2);
      herrdiff->Fill(erret - erret2);

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
  hdiff->Draw();
  new TCanvas;
  herrdiff->Draw();
  new TCanvas;
  hkaon->Draw();
  hkaonT->Draw("SAME");
}
