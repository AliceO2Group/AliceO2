/*
   root -l -q SplineDemo.C
 */

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "TFile.h"
#include "TRandom.h"
#include "TNtuple.h"
#include "Riostream.h"
#include "TSystem.h"
#include "TH1.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TMarker.h"
#include "TLine.h"
#include "GPU/Spline1D.h"
#include "GPU/IrregularSpline1D.h"
#include "GPU/SplineHelper1D.h"

#endif

const int Fdegree = 5;
static double Fcoeff[2 * (Fdegree + 1)];

int nKnots = 4;

void F(float u, float f[])
{
  double uu = u * TMath::Pi() / (nKnots - 1);
  f[0] = 0; //Fcoeff[0]/2;
  for (int i = 1; i <= Fdegree; i++) {
    f[0] += Fcoeff[2 * i] * TMath::Cos(i * uu) + Fcoeff[2 * i + 1] * TMath::Sin(i * uu);
  }
}

float Flocal(float u)
{
  float f = 0;
  F(u * (nKnots - 1), &f);
  return f;
}

TCanvas* canv = new TCanvas("cQA", "Spline Demo", 2000, 800);

bool doAskSteps = 1;
bool drawLocal = 1;

bool ask()
{
  canv->Update();
  cout << "type 'q ' to exit";
  if (doAskSteps) {
    cout << ", 's' to skip individual steps";
  } else {
    cout << ", 's' to stop at individual steps";
  }
  if (drawLocal) {
    cout << ", 'l' to skip local splines";
  } else {
    cout << ", 'l' to draw local splines";
  }
  cout << endl;

  std::string str;
  std::getline(std::cin, str);
  if (str == "s") {
    doAskSteps = !doAskSteps;
  } else if (str == "l") {
    drawLocal = !drawLocal;
  }
  return (str != "q" && str != ".q");
}

bool askStep()
{
  return (!doAskSteps) ? 1 : ask();
}

int SplineDemo()
{

  const int nAxiliaryPoints = 10;

  using namespace o2::gpu;

  cout << "Test interpolation.." << endl;

  //TCanvas* canv = new TCanvas("cQA", "Spline1D  QA", 2000, 1000);

  gRandom->SetSeed(0);

  for (int seed = 13;; seed++) {

    //seed = gRandom->Integer(100000); // 605

    gRandom->SetSeed(seed);
    cout << "Random seed: " << seed << " " << gRandom->GetSeed() << endl;

    for (int i = 0; i < 2 * (Fdegree + 1); i++) {
      Fcoeff[i] = gRandom->Uniform(-1, 1);
    }

    o2::gpu::Spline1D spline(nKnots);

    o2::gpu::SplineHelper1D helper;
    helper.setSpline(spline, nAxiliaryPoints);

    std::unique_ptr<float[]> parameters = helper.constructParameters(1, F, 0., spline.getUmax());

    o2::gpu::Spline1D splineClassic(nKnots);
    helper.setSpline(splineClassic, nAxiliaryPoints);

    std::unique_ptr<float[]> parametersClassic = helper.constructParametersClassic(1, F, 0., splineClassic.getUmax());

    IrregularSpline1D splineLocal;
    int nKnotsLocal = 2 * nKnots - 1;
    splineLocal.constructRegular(nKnotsLocal);

    std::unique_ptr<float[]> parametersLocal(new float[nKnotsLocal]);
    for (int i = 0; i < nKnotsLocal; i++) {
      parametersLocal[i] = Flocal(splineLocal.getKnot(i).u);
    }
    splineLocal.correctEdges(parametersLocal.get());

    spline.print();
    splineLocal.print();

    canv->Draw();

    TH1F* qaX = new TH1F("qaX", "qaX [um]", 1000, -1000., 1000.);

    TNtuple* knots = new TNtuple("knots", "knots", "type:u:f");

    for (int i = 0; i < nKnots; i++) {
      double u = splineClassic.getKnot(i).u;
      double fs = splineClassic.interpolate1D(parametersClassic.get(), u);
      knots->Fill(1, u, fs);
    }

    for (int i = 0; i < nKnots; i++) {
      double u = spline.getKnot(i).u;
      double fs = spline.interpolate1D(parameters.get(), u);
      knots->Fill(2, u, fs);
      if (i < nKnots - 1) {
        double u1 = spline.getKnot(i + 1).u;
        int nax = nAxiliaryPoints;
        double du = (u1 - u) / (nax + 1);
        for (int j = 0; j < nax; j++) {
          double uu = u + du * (j + 1);
          double ff = spline.interpolate1D(parameters.get(), uu);
          knots->Fill(3, uu, ff);
        }
      }
    }

    for (int i = 0; i < splineLocal.getNumberOfKnots(); i++) {
      double u = splineLocal.getKnot(i).u;
      double fs = splineLocal.getSpline(parametersLocal.get(), u);
      knots->Fill(4, u * (nKnots - 1), fs);
    }

    TNtuple* nt = new TNtuple("nt", "nt", "u:f0:fComp:fClass:fLocal");

    float stepS = 1.e-4;
    int nSteps = (int)(1. / stepS + 1);

    double statDfComp = 0;
    double statDfClass = 0;
    double statDfLocal = 0;
    double drawMax = -1.e20;
    double drawMin = 1.e20;
    int statN = 0;
    for (float s = 0; s < 1. + stepS; s += stepS) {
      double u = s * (nKnots - 1);
      float f0;
      F(u, &f0);
      double fComp = spline.interpolate1D(parameters.get(), u);
      double fClass = splineClassic.interpolate1D(parametersClassic.get(), u);
      double fLocal = splineLocal.getSpline(parametersLocal.get(), s);

      statDfComp += (fComp - f0) * (fComp - f0);
      statDfClass += (fClass - f0) * (fClass - f0);
      statDfLocal += (fLocal - f0) * (fLocal - f0);
      statN++;
      qaX->Fill(1.e4 * (fComp - f0));
      nt->Fill(u, f0, fComp, fClass, fLocal);
      drawMax = std::max(drawMax, std::max(fComp, std::max(fClass, fLocal)));
      drawMin = std::min(drawMin, std::min(fComp, std::min(fClass, fLocal)));
    }

    cout << "\n"
         << std::endl;
    cout << "\nRandom seed: " << seed << " " << gRandom->GetSeed() << endl;
    cout << "std dev Classic : " << sqrt(statDfClass / statN) << std::endl;
    cout << "std dev Local     : " << sqrt(statDfLocal / statN) << std::endl;
    cout << "std dev Compact   : " << sqrt(statDfComp / statN) << std::endl;

    /*
      canv->cd(1);
      qaX->Draw();
      canv->cd(2);
    */

    //nt->SetMarkerColor(kBlack);
    //nt->Draw("f0:u","","");

    {
      TNtuple* ntRange = new TNtuple("ntRange", "nt", "u:f");
      drawMin -= 0.1 * (drawMax - drawMin);

      ntRange->Fill(0, drawMin);
      ntRange->Fill(0, drawMax);
      ntRange->Fill(nKnots - 1, drawMin);
      ntRange->Fill(nKnots - 1, drawMax);
      ntRange->SetMarkerColor(kWhite);
      ntRange->SetMarkerSize(0.1);
      ntRange->Draw("f:u", "", "");
      delete ntRange;
    }

    auto legend = new TLegend(0.1, 0.82, 0.3, 0.95);
    //legend->SetHeader("Splines of the same size:","C"); // option "C" allows to center the header

    nt->SetMarkerColor(kGray);
    nt->SetMarkerStyle(8);
    nt->SetMarkerSize(2.);
    nt->Draw("f0:u", "", "P,same");

    TH1* htemp = (TH1*)gPad->GetPrimitive("htemp");
    htemp->SetTitle("Splines of the same size");

    TLine* l0 = new TLine();
    l0->SetLineWidth(10);
    l0->SetLineColor(kGray);
    legend->AddEntry(l0, "input function", "L");
    legend->Draw();

    knots->SetMarkerStyle(8);
    knots->SetMarkerSize(1.5);

    if (!askStep()) {
      break;
    }

    nt->SetMarkerSize(.5);
    nt->SetMarkerColor(kGreen + 2);
    nt->Draw("fClass:u", "", "P,same");

    knots->SetMarkerColor(kGreen + 2);
    knots->SetMarkerSize(3.5);
    knots->Draw("f:u", "type==1", "same"); // Classic
    TLine* l1 = new TLine();
    l1->SetLineWidth(4);
    l1->SetLineColor(kGreen + 2);
    legend->AddEntry(l1, "Classic (N knots + N slopes)", "L");
    legend->Draw();

    if (!askStep()) {
      break;
    }

    if (drawLocal) {
      nt->SetMarkerColor(kBlue);
      nt->Draw("fLocal:u", "", "P,same");

      knots->SetMarkerSize(2.5);
      knots->SetMarkerColor(kBlue);
      knots->Draw("f:u", "type==4", "same"); // local
      TLine* l2 = new TLine(*l1);
      l2->SetLineColor(kBlue);
      legend->AddEntry(l2, "local (2N knots)", "L");
      legend->Draw();
      if (!askStep()) {
        break;
      }
    }

    nt->SetMarkerColor(kRed);
    nt->Draw("fComp:u", "", "P,same");

    knots->SetMarkerColor(kRed);
    knots->SetMarkerSize(2.5);
    knots->Draw("f:u", "type==2", "same"); // compact
    TLine* l3 = new TLine(*l1);
    l3->SetLineColor(kRed);
    legend->AddEntry(l3, "compact (N knots + N slopes)", "L");
    legend->Draw();

    if (!askStep()) {
      break;
    }

    knots->SetMarkerColor(kBlack);
    knots->SetMarkerSize(1.);
    knots->Draw("f:u", "type==3", "same"); // compact, axiliary points
    TMarker* l4 = new TMarker;
    l4->SetMarkerStyle(8);
    l4->SetMarkerSize(1.);
    l4->SetMarkerColor(kBlack);
    legend->AddEntry(l4, "construction points", "P");
    legend->Draw();

    if (!ask()) {
      break;
    }
    delete legend;
  }

  return 0;
}
