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

  TTree* t = (TTree*)f->Get("tree");

  o2::dataformats::CalibTimeSlewingParamTOF* obj = new o2::dataformats::CalibTimeSlewingParamTOF();
  t->SetBranchAddress("CalibTimeSlewingParamTOF", &obj);
  t->GetEvent(0);

  int nbin = 1000;

  // where the entries for the tot vs timeSlewing correction for the desired channel are stored in the sector-wise vector
  int istart = obj->getStartIndexForChannel(ch / 8736, ch % 8736);
  int istop = obj->getStartIndexForChannel(ch / 8736, ch % 8736 + 1) - 1;
  printf("istart = %d -- istop = %d\n", istart, istop);

  float a[1000], b[1000];
  int np = 0;
  float range = 100;

  // corrections of the desidered channel
  const std::vector<std::pair<float, float>>* vect = obj->getVector(ch / 8736);
  for (int ii = istart; ii <= istop; ii++) {
    a[np] = vect->at(ii).first;
    b[np] = vect->at(ii).second;
    range = a[np] * 1.3;
    np++;
    printf("tot=%f -- time = %f\n", (*vect)[ii].first, (*vect)[ii].second);
  }

  TGraph* g = new TGraph(np, a, b);
  TProfile* h = new TProfile("h", "", nbin, 0, range);
  float inv = range / nbin;

  for (int i = 1; i <= nbin; i++) {
    float ts = obj->evalTimeSlewing(ch, (i - 0.5) * inv);
    printf("bin = %i, tot = %f, timeSlewing correction = %f\n", i, (i - 0.5) * inv, ts);
    h->Fill((i - 0.5) * inv, ts);
  }

  // comparing if the stored corrections match with what evalTimeSlewing returns
  h->Draw("P");
  h->SetMarkerStyle(20);

  g->SetLineColor(2);
  g->Draw("L");
}
