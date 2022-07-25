/*
   root -l -q SplineDemo.C

   The macro demonstrates how BestFitSpline constructor recovers regions with
   missing data points.
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
#include "GPU/Spline1DHelper.h"

#endif

const int Fdegree = 10;
int nKnots = 6;

static double Fcoeff[2 * (Fdegree + 1)];

void F(double u, double f[])
{
  double uu = u * TMath::Pi() / (nKnots - 1);
  f[0] = 0; // Fcoeff[0]/2;
  for (int i = 1; i <= Fdegree; i++) {
    f[0] += Fcoeff[2 * i] * TMath::Cos(i * uu) + Fcoeff[2 * i + 1] * TMath::Sin(i * uu);
  }
}

double F1D(double u)
{
  double f = 0;
  F(u, &f);
  return f;
}

TCanvas* canv = new TCanvas("cQA", "Spline Recovery Demo", 1600, 600);

bool doAskSteps = 1;
bool drawConstruction = 1;

bool ask()
{
  canv->Update();
  std::cout << "type: 'q'-exit";
  std::cout << ", 's'-individual steps";
  std::cout << ", 'p'-construction points";

  std::cout << std::endl;

  std::string str;
  std::getline(std::cin, str);
  if (str == "s") {
    doAskSteps = !doAskSteps;
  } else if (str == "p") {
    drawConstruction = !drawConstruction;
  }

  return (str != "q" && str != ".q");
}

bool doContinie = 1;

bool askStep()
{
  doContinie = (!doAskSteps) ? 1 : ask();
  return doContinie;
}

int SplineRecoveryDemo()
{

  const int nAxiliaryPoints = 10;

  using namespace o2::gpu;

  std::cout << "Test how the spline constructor fills gaps in the data.." << std::endl;

  // TCanvas* canv = new TCanvas("cQA", "Spline1D  QA", 2000, 1000);

  gRandom->SetSeed(0);

  TH1F* histDfBestFit = new TH1F("histDfBestFit", "Df BestFit", 100, -1., 1.);
  TH1F* histMinMaxBestFit = new TH1F("histMinMaxBestFit", "MinMax BestFit", 100, -1., 1.);

  for (int seed = 12; doContinie; seed++) {

    // seed = gRandom->Integer(100000); // 605

    gRandom->SetSeed(seed);
    std::cout << "Random seed: " << seed << " " << gRandom->GetSeed() << std::endl;

    for (int i = 0; i < 2 * (Fdegree + 1); i++) {
      Fcoeff[i] = gRandom->Uniform(-1, 1);
    }

    for (int deadRegion = -1; doContinie && (deadRegion < nKnots - 1); deadRegion++) {

      TNtuple* knots = new TNtuple("knots", "knots", "type:u:f");

      o2::gpu::Spline1D<float, 1> spline(nKnots);
      spline.approximateFunction(0, nKnots - 1, F, nAxiliaryPoints);

      for (int i = 0; i < nKnots; i++) {
        double u = spline.getKnot(i).u;
        double fs = spline.interpolate(spline.convUtoX(u));
        knots->Fill(2, u, fs);
      }

      o2::gpu::Spline1D<float, 1> splineRecovered(nKnots);
      o2::gpu::Spline1DHelper<float> helper;
      {
        vector<double> vu, vy;
        for (int i = 0; i < nKnots - 1; i++) {
          if (i < deadRegion || i > deadRegion + 1) {
            double u = spline.getKnot(i).u;
            double du = (spline.getKnot(i + 1).u - u) / (nAxiliaryPoints + 1);
            for (int iax = 0; iax < nAxiliaryPoints + 1; iax++, u += du) {
              double f = F1D(u);
              vu.push_back(u);
              vy.push_back(f);
              knots->Fill(3, u, f);
            }
          }
        }
        {
          int i = nKnots - 1;
          if (i < deadRegion || i > deadRegion + 2) {
            double u = spline.getKnot(i).u;
            double f = F1D(u);
            vu.push_back(u);
            vy.push_back(f);
            knots->Fill(3, u, f);
          }
        }
        helper.approximateDataPoints(splineRecovered, 0, nKnots - 1, &vu[0], &vy[0], vu.size());
      }

      spline.print();

      canv->Draw();

      for (int i = 0; i < nKnots; i++) {
        double u = splineRecovered.getKnot(i).u;
        double fs = splineRecovered.interpolate(splineRecovered.convUtoX(u));
        knots->Fill(1, u, fs);
      }

      TNtuple* nt = new TNtuple("nt", "nt", "u:f0:fBestFit:fRec");

      float stepS = 1.e-4;
      int nSteps = (int)(1. / stepS + 1);

      double statDfBestFit = 0;
      double statDfRec = 0;

      double statMinMaxBestFit = 0;
      double statMinMaxRec = 0;

      double drawMax = -1.e20;
      double drawMin = 1.e20;
      int statN = 0;
      for (float s = 0; s < 1. + stepS; s += stepS) {
        double u = s * (nKnots - 1);
        double f0;
        F(u, &f0);
        double fBestFit = spline.interpolate(spline.convUtoX(u));
        double fRec = splineRecovered.interpolate(splineRecovered.convUtoX(u));

        nt->Fill(u, f0, fBestFit, fRec);
        drawMax = std::max(drawMax, (double)f0);
        drawMin = std::min(drawMin, (double)f0);
        drawMax = std::max(drawMax, std::max(fBestFit, fRec));
        drawMin = std::min(drawMin, std::min(fBestFit, fRec));
        statDfBestFit += (fBestFit - f0) * (fBestFit - f0);
        statDfRec += (fRec - f0) * (fRec - f0);
        statN++;
        histDfBestFit->Fill(fBestFit - f0);

        statMinMaxBestFit = std::max(statMinMaxBestFit, fabs(fBestFit - f0));
        statMinMaxRec = std::max(statMinMaxRec, fabs(fRec - f0));
      }

      histMinMaxBestFit->Fill(statMinMaxBestFit);

      std::cout << "\n"
                << std::endl;
      std::cout << "\nRandom seed: " << seed << " " << gRandom->GetSeed() << std::endl;
      std::cout << "Best-fit  : std dev " << sqrt(statDfBestFit / statN) << " minmax " << statMinMaxBestFit << std::endl;
      std::cout << "Recovered : std dev " << sqrt(statDfRec / statN) << " minmax " << statMinMaxRec << std::endl;

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

      auto legend = new TLegend(0.1, 0.8, 0.4, 0.95);
      // legend->SetHeader("Splines of the same size:","C"); // option "C" allows to center the header

      nt->SetMarkerColor(kGray);
      nt->SetMarkerStyle(8);
      nt->SetMarkerSize(2.);
      nt->Draw("f0:u", "", "P,same");

      TH1* htemp = (TH1*)gPad->GetPrimitive("htemp");
      htemp->SetTitle("Best-Fit Spline with a recovered region");
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
      {
        int col = kGreen + 2;
        nt->SetMarkerColor(col);
        nt->SetMarkerStyle(8);
        nt->SetMarkerSize(1.);
        nt->Draw("fBestFit:u", "", "P,same");

        knots->SetMarkerStyle(20);
        knots->SetMarkerColor(col);
        knots->SetMarkerSize(2.5);             // 5.
        knots->Draw("f:u", "type==2", "same"); // best-fit splines

        // TMarker * l3 = new TMarker();
        TNtuple* l3 = new TNtuple();
        l3->SetMarkerStyle(20);
        l3->SetMarkerColor(col);
        l3->SetMarkerSize(2.5); // 3.5
        l3->SetLineColor(col);
        l3->SetLineWidth(5.);
        // legend->AddEntry(l3, Form("Best-fit spline (%d knots + %d slopes)", nKnots, nKnots), "PL");
        legend->AddEntry(l3, "Best-fit spline", "PL");
        legend->Draw();

        knots->SetMarkerStyle(8);

        if (!askStep()) {
          break;
        }
      }

      {
        int col = kRed;
        nt->SetMarkerSize(1.);
        nt->SetMarkerColor(col);
        nt->Draw("fRec:u", "", "P,same");

        knots->SetMarkerStyle(21);
        knots->SetMarkerColor(col);
        knots->SetMarkerSize(2.5);             // 5.
        knots->Draw("f:u", "type==1", "same"); // Recovered
        TNtuple* l1 = new TNtuple();
        l1->SetMarkerStyle(21);
        l1->SetMarkerColor(col);
        l1->SetMarkerSize(1.5); // 3.5
        l1->SetLineColor(col);
        l1->SetLineWidth(2.); // 5.
        // legend->AddEntry(l1, Form("Interpolation spline (%d knots + %d slopes)", nKnots, nKnots), "LP");
        legend->AddEntry(l1, "Recovered spline", "LP");
        legend->Draw();

        if (!askStep()) {
          break;
        }
      }

      if (drawConstruction) {
        knots->SetMarkerColor(kBlack);
        knots->SetMarkerSize(1.5);
        knots->SetMarkerStyle(8);

        knots->Draw("f:u", "type==3", "same"); // best-fit data points
        TMarker* l4 = new TMarker;
        l4->SetMarkerStyle(8);
        l4->SetMarkerSize(1.5);
        l4->SetMarkerColor(kBlack);
        legend->AddEntry(l4, "Construction points for recov. spline", "P");
        legend->Draw();
        if (!askStep()) {
          break;
        }
      }

      if (!doAskSteps && !ask()) {
        break;
      }

      delete legend;
    } // dead region
  }   // random F

  return 0;
}
