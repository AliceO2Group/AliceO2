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
#include "GPU/Spline1DHelperOld.h"
#include "Math/Functor.h"
#include "Math/ChebyshevApprox.h"

#endif

const int Fdegree = 5;
int nKnots = 4;

static double Fcoeff[2 * (Fdegree + 1)];

void F(double u, double f[])
{
  double uu = u * TMath::Pi() / (nKnots - 1);
  f[0] = 0; // Fcoeff[0]/2;
  for (int i = 1; i <= Fdegree; i++) {
    f[0] += Fcoeff[2 * i] * TMath::Cos(i * uu) + Fcoeff[2 * i + 1] * TMath::Sin(i * uu);
  }
}

/*
void F(float u, float f[])
{
  double uu = u * 2 / (nKnots - 1)-1.;
  double t0=1;
  double t1 = uu;
  f[0] = 0;
  for (int i = 1; i <= Fdegree*2; i++) {
     double t = t = 2*uu*t1-t0;
     f[0] += Fcoeff[i]*t;
     t0 = t1;
     t1 = t;
  }
}
*/

double F1D(double u)
{
  double f = 0;
  F(u, &f);
  return f;
}

double Flocal(double u)
{
  double f = 0;
  F(u * (nKnots - 1), &f);
  return f;
}

TCanvas* canv = new TCanvas("cQA", "Spline Demo", 1600, 800);

bool doAskSteps = 1;
bool drawLocal = 0;
bool drawCheb = 0;
bool drawConstruction = 1;

bool ask()
{
  canv->Update();
  std::cout << "type: 'q'-exit";
  std::cout << ", 's'-individual steps";
  std::cout << ", 'l'-local splines";
  std::cout << ", 'c'-chebychev";
  std::cout << ", 'p'-construction points";

  std::cout << std::endl;

  std::string str;
  std::getline(std::cin, str);
  if (str == "s") {
    doAskSteps = !doAskSteps;
  } else if (str == "l") {
    drawLocal = !drawLocal;
  } else if (str == "p") {
    drawConstruction = !drawConstruction;
  } else if (str == "c") {
    drawCheb = !drawCheb;
  }

  return (str != "q" && str != ".q");
}

bool askStep()
{
  return (!doAskSteps) ? 1 : ask();
}

int SplineConstructionDemo()
{

  const int nAxiliaryPoints = 10;

  using namespace o2::gpu;

  std::cout << "Test interpolation.." << std::endl;

  // TCanvas* canv = new TCanvas("cQA", "Spline1D  QA", 2000, 1000);

  gRandom->SetSeed(0);

  TH1F* histDfBestFit = new TH1F("histDfBestFit", "Df BestFit", 100, -1., 1.);
  TH1F* histDfCheb = new TH1F("histDfCheb", "Df Chebyshev", 100, -1., 1.);
  TH1F* histMinMaxBestFit = new TH1F("histMinMaxBestFit", "MinMax BestFit", 100, -1., 1.);
  TH1F* histMinMaxCheb = new TH1F("histMinMaxCheb", "MinMax Chebyshev", 100, -1., 1.);

  for (int seed = 12;; seed++) {

    // seed = gRandom->Integer(100000); // 605

    gRandom->SetSeed(seed);
    std::cout << "Random seed: " << seed << " " << gRandom->GetSeed() << std::endl;

    for (int i = 0; i < 2 * (Fdegree + 1); i++) {
      Fcoeff[i] = gRandom->Uniform(-1, 1);
    }

    o2::gpu::Spline1D<float, 1> spline(nKnots);
    spline.approximateFunction(0, nKnots - 1, F, nAxiliaryPoints);

    o2::gpu::Spline1D<float, 1> splineClassic(nKnots);
    o2::gpu::Spline1DHelperOld<float> helper;
    {
      vector<double> vu, vy;
      for (int i = 0; i < nKnots; i += 1) {
        double u = spline.getKnot(i).u;
        if (i > 0) {
          vu.push_back(u);
          vy.push_back(F1D(u));
        }
        if (i >= 0) {

          vu.push_back(u + 0.5);
          vy.push_back(F1D(u + 0.5));

          /*
          vu.push_back(u + 0.6);
          vy.push_back(F1D(u + 0.6));
          */
        }
      }
      helper.approximateDataPoints(splineClassic, 0, nKnots - 1, &vu[0], &vy[0], vu.size());
    }

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

    ROOT::Math::Functor1D func1D(&F1D);
    ROOT::Math::ChebyshevApprox cheb(func1D, 0, nKnots - 1, nKnots + (nKnots - 1) * (nAxiliaryPoints)-1);

    canv->Draw();

    TNtuple* knots = new TNtuple("knots", "knots", "type:u:f");

    for (int i = 0; i < nKnots; i++) {
      double u = splineClassic.getKnot(i).u;
      double fs = splineClassic.interpolate(splineClassic.convUtoX(u));
      knots->Fill(1, u, fs);
    }

    helper.setSpline(spline, 1, nAxiliaryPoints);
    for (int j = 0; j < helper.getNumberOfDataPoints(); j++) {
      const typename Spline1DHelperOld<float>::DataPoint& p = helper.getDataPoint(j);
      double f0;
      F(p.u, &f0);
      double fs = spline.interpolate(spline.convUtoX(p.u));
      if (p.isKnot) {
        knots->Fill(2, p.u, fs);
      }
      knots->Fill(3, p.u, f0);

      double y = cos(M_PI * (j + 0.5) / (helper.getNumberOfDataPoints()));
      double uCheb = 0.5 * (y * (nKnots - 1) + nKnots - 1);
      double fCheb = cheb(uCheb, 2 * nKnots - 1);
      knots->Fill(5, uCheb, fCheb);
    }

    for (int i = 0; i < splineLocal.getNumberOfKnots(); i++) {
      double u = splineLocal.getKnot(i).u;
      double fs = splineLocal.getSpline(parametersLocal.get(), u);
      knots->Fill(4, u * (nKnots - 1), fs);
    }

    TNtuple* nt = new TNtuple("nt", "nt", "u:f0:fBestFit:fClass:fLocal:fCheb");

    float stepS = 1.e-4;
    int nSteps = (int)(1. / stepS + 1);

    double statDfBestFit = 0;
    double statDfClass = 0;
    double statDfLocal = 0;
    double statDfCheb = 0;

    double statMinMaxBestFit = 0;
    double statMinMaxClass = 0;
    double statMinMaxLocal = 0;
    double statMinMaxCheb = 0;

    double drawMax = -1.e20;
    double drawMin = 1.e20;
    int statN = 0;
    for (float s = 0; s < 1. + stepS; s += stepS) {
      double u = s * (nKnots - 1);
      double f0;
      F(u, &f0);
      double fBestFit = spline.interpolate(spline.convUtoX(u));
      double fClass = splineClassic.interpolate(splineClassic.convUtoX(u));
      double fLocal = splineLocal.getSpline(parametersLocal.get(), s);
      double fCheb = cheb(u, 2 * nKnots - 1);
      nt->Fill(u, f0, fBestFit, fClass, fLocal, fCheb);
      drawMax = std::max(drawMax, (double)f0);
      drawMin = std::min(drawMin, (double)f0);
      drawMax = std::max(drawMax, std::max(fBestFit, std::max(fClass, fLocal)));
      drawMin = std::min(drawMin, std::min(fBestFit, std::min(fClass, fLocal)));
      drawMax = std::max(drawMax, fCheb);
      drawMin = std::min(drawMin, fCheb);
      statDfBestFit += (fBestFit - f0) * (fBestFit - f0);
      statDfClass += (fClass - f0) * (fClass - f0);
      statDfLocal += (fLocal - f0) * (fLocal - f0);
      statDfCheb += (fCheb - f0) * (fCheb - f0);
      statN++;
      histDfBestFit->Fill(fBestFit - f0);
      histDfCheb->Fill(fCheb - f0);

      statMinMaxBestFit = std::max(statMinMaxBestFit, fabs(fBestFit - f0));
      statMinMaxClass = std::max(statMinMaxClass, fabs(fClass - f0));
      statMinMaxLocal = std::max(statMinMaxLocal, fabs(fLocal - f0));
      statMinMaxCheb = std::max(statMinMaxCheb, fabs(fCheb - f0));
    }

    histMinMaxBestFit->Fill(statMinMaxBestFit);
    histMinMaxCheb->Fill(statMinMaxCheb);

    std::cout << "\n"
              << std::endl;
    std::cout << "\nRandom seed: " << seed << " " << gRandom->GetSeed() << std::endl;
    std::cout << "Classical : std dev " << sqrt(statDfClass / statN) << " minmax " << statMinMaxClass << std::endl;
    std::cout << "Local     : std dev " << sqrt(statDfLocal / statN) << " minmax " << statMinMaxLocal << std::endl;
    std::cout << "Best-fit  : std dev " << sqrt(statDfBestFit / statN) << " minmax " << statMinMaxBestFit << std::endl;
    std::cout << "Chebyshev : std dev " << sqrt(statDfCheb / statN) << " minmax " << statMinMaxCheb << std::endl;

    /*
      canv->cd(1);
      qaX->Draw();
      canv->cd(2);
    */

    // nt->SetMarkerColor(kBlack);
    // nt->Draw("f0:u","","");

    {
      TNtuple* ntRange = new TNtuple("ntRange", "nt", "u:f");
      double L = drawMax - drawMin;
      drawMin -= 0.0 * L;
      drawMax += 0.1 * L;

      ntRange->Fill(-0.001, drawMin);
      ntRange->Fill(-0.001, drawMax);
      ntRange->Fill(nKnots - 1 - 0.005, drawMin);
      ntRange->Fill(nKnots - 1 - 0.005, drawMax);
      ntRange->SetMarkerColor(kWhite);
      ntRange->SetMarkerSize(0.1);
      ntRange->Draw("f:u", "", "");
      delete ntRange;
    }

    auto legend = new TLegend(0.1, 0.72, 0.4, 0.95);
    // legend->SetHeader("Splines of the same size:","C"); // option "C" allows to center the header

    nt->SetMarkerColor(kGray);
    nt->SetMarkerStyle(8);
    nt->SetMarkerSize(2.);
    nt->Draw("f0:u", "", "P,same");

    TH1* htemp = (TH1*)gPad->GetPrimitive("htemp");
    htemp->SetTitle("Splines of the same size");
    htemp->GetXaxis()->SetTitle("x");
    htemp->GetYaxis()->SetTitle("R");

    TLine* l0 = new TLine();
    l0->SetLineWidth(7);
    l0->SetLineColor(kGray);
    // legend->AddEntry(l0, "Input function", "L");
    legend->AddEntry(l0, "Function to approximate", "L");
    legend->Draw();

    knots->SetMarkerStyle(8);
    knots->SetMarkerSize(1.5);

    if (!askStep()) {
      break;
    }

    nt->SetMarkerSize(1.);
    nt->SetMarkerColor(kGreen + 2);
    nt->Draw("fClass:u", "", "P,same");

    knots->SetMarkerStyle(21);
    knots->SetMarkerColor(kGreen + 2);
    knots->SetMarkerSize(2.5);             // 5.
    knots->Draw("f:u", "type==1", "same"); // Classical
    TNtuple* l1 = new TNtuple();
    l1->SetMarkerStyle(21);
    l1->SetMarkerColor(kGreen + 2);
    l1->SetMarkerSize(1.5); // 3.5
    l1->SetLineColor(kGreen + 2);
    l1->SetLineWidth(2.); // 5.
    // legend->AddEntry(l1, Form("Interpolation spline (%d knots + %d slopes)", nKnots, nKnots), "LP");
    legend->AddEntry(l1, "Interpolation spline", "LP");
    legend->Draw();

    if (!askStep()) {
      break;
    }

    if (drawLocal) {
      nt->SetMarkerColor(kMagenta);
      nt->Draw("fLocal:u", "", "P,same");

      knots->SetMarkerSize(1.5); // 2.5
      knots->SetMarkerStyle(8);
      knots->SetMarkerColor(kMagenta);
      knots->Draw("f:u", "type==4", "same"); // local
      TLine* l2 = new TLine();
      l2->SetLineWidth(2); // 4
      l2->SetLineColor(kMagenta);
      legend->AddEntry(l2, Form("Local spline (%d knots)", 2 * nKnots - 1), "L");
      legend->Draw();
      if (!askStep()) {
        break;
      }
    }
    if (drawCheb) {
      nt->SetMarkerColor(kBlue);
      nt->Draw("fCheb:u", "", "P,same");
      TLine* lCheb = new TLine();
      lCheb->SetLineWidth(2); // 4
      lCheb->SetLineColor(kBlue);
      legend->AddEntry(lCheb, Form("Chebyshev (%d coeff)", 2 * nKnots), "L");
      legend->Draw();

      if (!askStep()) {
        break;
      }
    }

    nt->SetMarkerColor(kRed);
    nt->Draw("fBestFit:u", "", "P,same");

    knots->SetMarkerStyle(20);
    knots->SetMarkerColor(kRed);
    knots->SetMarkerSize(2.5);             // 5.
    knots->Draw("f:u", "type==2", "same"); // best-fit splines

    // TMarker * l3 = new TMarker();
    TNtuple* l3 = new TNtuple();
    l3->SetMarkerStyle(20);
    l3->SetMarkerColor(kRed);
    l3->SetMarkerSize(2.5); // 3.5
    l3->SetLineColor(kRed);
    l3->SetLineWidth(5.);
    // legend->AddEntry(l3, Form("Best-fit spline (%d knots + %d slopes)", nKnots, nKnots), "PL");
    legend->AddEntry(l3, "Best-fit spline", "PL");
    legend->Draw();

    knots->SetMarkerStyle(8);

    if (!askStep()) {
      break;
    }

    if (drawConstruction) {
      knots->SetMarkerColor(kBlack);
      knots->SetMarkerSize(1.5);
      knots->SetMarkerStyle(8);

      knots->Draw("f:u", "type==3", "same"); // best-fit data points
      // knots->Draw("f:u", "type==5", "same"); // chebyshev, data points
      TMarker* l4 = new TMarker;
      l4->SetMarkerStyle(8);
      l4->SetMarkerSize(1.5);
      l4->SetMarkerColor(kBlack);
      legend->AddEntry(l4, "Construction points", "P");
      legend->Draw();
      if (!askStep()) {
        break;
      }
    }

    if (!doAskSteps && !ask()) {
      break;
    }

    delete legend;
  }

  return 0;
}
