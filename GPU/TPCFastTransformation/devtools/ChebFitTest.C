/*
   root -l -q ChebFitTest.C
 */

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "GPU/ChebyshevFit1D.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TLegend.h"
#include "TLine.h"
#include "TMarker.h"
#include "TMath.h"
#include "TNtuple.h"
#include "TRandom.h"
#include "TSystem.h"

#endif

const int32_t Fdegree = 4;

static double Fcoeff[2 * (Fdegree + 1)];

double F(double x)
{
  double f = Fcoeff[0] / 2;
  for (int32_t i = 1; i <= Fdegree; i++) {
    f += Fcoeff[2 * i] * TMath::Cos(i * x) +
         Fcoeff[2 * i + 1] * TMath::Sin(i * x);
  }
  return f;
}

TCanvas* canv = new TCanvas("cQA", "ChebFitDemo", 1600, 800);

bool ask()
{
  canv->Update();
  std::cout << "type: 'q'-exit";
  std::cout << std::endl;
  std::string str;
  std::getline(std::cin, str);
  return (str != "q" && str != ".q");
}

int32_t ChebFitTest()
{
  const double xMin = 0.5;
  const double xMax = M_PI - 0.5;
  const int32_t nFitPoints = 10;
  const int32_t nCoeff = 10;

  using namespace o2::gpu;

  gRandom->SetSeed(0);

  for (int32_t seed = 1;; seed++) {
    gRandom->SetSeed(seed);
    std::cout << "Random seed: " << seed << " " << gRandom->GetSeed()
              << std::endl;

    for (int32_t i = 0; i < 2 * (Fdegree + 1); i++) {
      Fcoeff[i] = gRandom->Uniform(-1, 1);
    }

    TNtuple* nt = new TNtuple("points", "points", "type:x:f");

    o2::gpu::ChebyshevFit1D cheb2(nCoeff - 1, 0, M_PI);
    o2::gpu::ChebyshevFit1D cheb1(nCoeff - 1, xMin + 0.5, xMax - 0.5);

    double dx = (xMax - xMin) / (nFitPoints - 1);
    for (int32_t ip = 0; ip < nFitPoints; ip++) {
      double x = xMin + ip * dx;
      double f = F(x);
      cheb1.addMeasurement(x, f);
      cheb2.addMeasurement(x, f);
      nt->Fill(1, x, f);
    }

    cheb1.fit();
    cheb2.fit();

    double drawMin = 1.e10;
    double drawMax = -1.e10;

    dx = M_PI / 1000;
    for (double x = 0; x < M_PI; x += dx) {
      double f = F(x);
      double c1 = cheb1.eval(x);
      double c2 = cheb2.eval(x);
      nt->Fill(0, x, f);
      nt->Fill(2, x, c1);
      nt->Fill(3, x, c2);
      drawMin = (drawMin < f) ? drawMin : f;
      drawMax = (drawMax > f) ? drawMax : f;
      drawMin = (drawMin < c1) ? drawMin : c1;
      drawMax = (drawMax > c1) ? drawMax : c1;
      drawMin = (drawMin < c2) ? drawMin : c2;
      drawMax = (drawMax > c2) ? drawMax : c2;
    }

    canv->Draw();

    {
      TNtuple* ntRange = new TNtuple("ntRange", "ntRange", "x:f");
      double L = drawMax - drawMin;
      drawMin -= 0.0 * L;
      drawMax += 0.1 * L;

      ntRange->Fill(-0.001, drawMin);
      ntRange->Fill(-0.001, drawMax);
      ntRange->Fill(M_PI - 0.005, drawMin);
      ntRange->Fill(M_PI - 0.005, drawMax);
      ntRange->SetMarkerColor(kWhite);
      ntRange->SetMarkerSize(0.1);
      ntRange->Draw("f:x", "", "");
      delete ntRange;
    }

    TH1* htemp = (TH1*)gPad->GetPrimitive("htemp");
    htemp->SetTitle("Fit with Chebyshev polynoms");
    htemp->GetXaxis()->SetTitle("x");
    htemp->GetYaxis()->SetTitle("F");

    auto legend = new TLegend(0.1, 0.8, 0.3, 0.95);

    {
      nt->SetMarkerColor(kGray);
      nt->SetMarkerStyle(8);
      nt->SetMarkerSize(2.);
      nt->Draw("f:x", "type==0", "P,same");

      TLine* l0 = new TLine();
      l0->SetLineWidth(7);
      l0->SetLineColor(kGray);
      legend->AddEntry(l0, "Function to fit", "L");
      legend->Draw();
    }

    {
      nt->SetMarkerSize(1.);
      nt->SetMarkerColor(kBlue);
      nt->Draw("f:x", "type==2", "P,same");
      TLine* lCheb = new TLine();
      lCheb->SetLineWidth(2); // 4
      lCheb->SetLineColor(kBlue);
      legend->AddEntry(lCheb, Form("Chebyshev1 (%d coeff)", nCoeff), "L");
      legend->Draw();
    }
    {
      nt->SetMarkerSize(1.);
      nt->SetMarkerColor(kGreen + 2);
      nt->Draw("f:x", "type==3", "P,same");
      TLine* lCheb = new TLine();
      lCheb->SetLineWidth(2); // 4
      lCheb->SetLineColor(kGreen + 2);
      legend->AddEntry(lCheb, Form("Chebyshev2 (%d coeff)", nCoeff), "L");
      legend->Draw();
    }

    {
      nt->SetMarkerColor(kBlack);
      nt->SetMarkerSize(1.5);
      nt->SetMarkerStyle(8);

      nt->Draw("f:x", "type==1", "same");
      TMarker* l4 = new TMarker;
      l4->SetMarkerStyle(8);
      l4->SetMarkerSize(1.5);
      l4->SetMarkerColor(kBlack);
      legend->AddEntry(l4, "Construction points", "P");
      legend->Draw();
    }

    if (!ask()) {
      break;
    }

    delete legend;
  }

  return 0;
}
