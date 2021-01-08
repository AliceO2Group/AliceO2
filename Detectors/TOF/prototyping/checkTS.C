#include "TFile.h"
#include "TTree.h"
#include "TGraph.h"
#include "TProfile.h"
#include "DataFormatsTOF/CalibTimeSlewingParamTOF.h"

void checkTS(int ch = 0)
{

  // macro to verify the evalTimeSlewing function

  TFile* f = new TFile("outputCCDBfromOCDB.root");
  f->ls();
  o2::dataformats::CalibTimeSlewingParamTOF* obj = reinterpret_cast<o2::dataformats::CalibTimeSlewingParamTOF*>(f->GetObjectChecked("TimeSlewing", o2::dataformats::CalibTimeSlewingParamTOF::Class()));
  int nbin = 1000;

  // where the entries for the tot vs timeSlewing correction for the desired channel are stored in the sector-wise vector
  int istart = obj->getStartIndexForChannel(ch / 8736, ch % 8736);
  int istop = obj->getStopIndexForChannel(ch / 8736, ch % 8736);
  ;
  printf("istart = %d -- istop = %d\n", istart, istop);

  float a[1000], b[1000];
  int np = 0;
  float range = 100;

  // corrections of the desidered channel
  const std::vector<std::pair<unsigned short, short>> vect = obj->getVector(ch / 8736);
  for (int ii = istart; ii <= istop; ii++) {
    a[np] = (float)vect.at(ii).first;
    b[np] = (float)vect.at(ii).second;
    range = a[np] * 1.3;
    np++;
  }

  TGraph* g = new TGraph(np, a, b);
  TProfile* h = new TProfile("h", "", nbin, 0, range);
  float inv = range / nbin;

  for (int i = 1; i <= nbin; i++) {
    float ts = obj->evalTimeSlewing(ch, (i - 0.5) * inv * 1e-3); // I need to multiply by 1e-3 because the TimeSLewing object is stored in ps, so "inv" will be in ps, but when I call "evalTimeSlewing" the argument is expected in ns
    h->Fill((i - 0.5) * inv, ts);
  }

  // comparing if the stored corrections match with what evalTimeSlewing returns
  h->Draw("P");
  h->SetMarkerStyle(20);

  g->SetLineColor(2);
  g->Draw("L");
}
