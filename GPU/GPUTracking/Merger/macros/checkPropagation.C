/*
 #include "Riostream.h"
 #include "TFile.h"
 #include "TMath.h"
 #include "TCanvas.h"
 #include "TH1F.h"
 #include "AliHLTTPCGeometry.h"
 */
const double kTwoPi = TMath::TwoPi(); // 2.*kPi;
const double kSliceDAngle = kTwoPi / 18.;
const double kSliceAngleOffset = kSliceDAngle / 2;

int GetSlice(double GlobalPhi)
{
  double phi = GlobalPhi;
  //  std::cout<<" GetSlice: phi = "<<phi<<std::endl;

  if (phi >= kTwoPi) {
    phi -= kTwoPi;
  }
  if (phi < 0) {
    phi += kTwoPi;
  }
  return (int)(phi / kSliceDAngle);
}

int GetDSlice(double LocalPhi) { return GetSlice(LocalPhi + kSliceAngleOffset); }

double GetSliceAngle(int iSlice) { return kSliceAngleOffset + iSlice * kSliceDAngle; }

int RecalculateSlice(GPUTPCGMPhysicalTrackModel& t, AliExternalTrackParam& t0, int& iSlice)
{
  double phi = atan2(t.GetY(), t.GetX());
  //  std::cout<<" recalculate: phi = "<<phi<<std::endl;
  int dSlice = GetDSlice(phi);

  if (dSlice == 0) {
    return 0; // nothing to do
  }
  //  std::cout<<" dSlice = "<<dSlice<<std::endl;
  double dAlpha = dSlice * kSliceDAngle;

  iSlice += dSlice;
  if (iSlice >= 18) {
    iSlice -= 18;
  }

  // rotate track on angle dAlpha
  t.Rotate(dAlpha);
  t0.Rotate(GetSliceAngle(iSlice));

  return 1;
}

int checkPropagation()
{
  // gSystem->Load("libAliHLTTPC.so");

  TH1F* hDiff[3] = {0, 0, 0};

  for (int i = 0; i < 3; i++) {
    char* s = i == 0 ? "X" : (i == 1 ? "Y" : "Z");
    char name[1024], title[1024];
    sprintf(name, "hDiff%s", s);
    sprintf(title, "Propagation Difference in %s", s);
    hDiff[i] = new TH1F(name, title, 1000, -20., 20.);
    hDiff[i]->GetXaxis()->SetTitle("Propagation difference [um]");
  }

  AliMagF* fld = new AliMagF("Fit", "Fit", 1., 1., AliMagF::k5kG);

  GPUTPCGMPolynomialField field;
  GPUTPCGMPolynomialFieldManager fieldCr;
  fieldCr.GetPolynomialField(field);

  GPUTPCGMPropagator prop;
  prop.SetPolynomialField(&field);
  prop.SetToyMCEventsFlag(kTRUE);

  const int nTracks = 1000;

  for (int itr = 0; itr < nTracks; itr++) {
    std::cout << "Track " << itr << ":" << std::endl;

    double dphi = kTwoPi / nTracks;
    double phi = kSliceAngleOffset + dphi * itr;
    double eta = gRandom->Uniform(-1.5, 1.5);
    double theta = 2 * TMath::ATan(1. / TMath::Exp(eta));
    double lambda = theta - TMath::Pi() / 2;
    // double theta = gRandom->Uniform(-60,60)*TMath::Pi()/180.;
    double pt = .1 * std::pow(10, gRandom->Uniform(0, 2.2));
    double q = 1.;
    int iSlice = GetSlice(phi);
    phi = phi - GetSliceAngle(iSlice);

    // std::cout<<"phi = "<<phi<<std::endl;

    double x0 = cos(phi);
    double y0 = sin(phi);
    double z0 = tan(lambda);
    double px = pt * x0;
    double py = pt * y0;
    double pz = pt * z0;
    GPUTPCGMPhysicalTrackModel t;
    t.Set(x0, y0, z0, px, py, pz, q);

    AliExternalTrackParam t0;
    {
      double alpha = GetSliceAngle(iSlice);
      double p[5] = {t.GetY(), t.GetZ(), t.GetSinPhi(), t.GetDzDs(), t.GetQPt()};
      double cv[15];
      for (int i = 0; i < 15; i++) {
        cv[i] = 0;
      }
      t0 = AliExternalTrackParam(x0, alpha, p, cv);
    }

    if (RecalculateSlice(t, t0, iSlice) != 0) {
      std::cout << "Initial slice wrong!!!" << std::endl;
      // exit(0);
    }
    AliHLTTPCGeometry geo;

    for (int iRow = 0; iRow < geo.GetNRows(); iRow++) {
      // if( iRow>=50 ) break; //SG!!!
      float xRow = geo.Row2X(iRow);
      // transport to row
      int err = 0;
      for (int itry = 0; itry < 1; itry++) {
        double alpha = GetSliceAngle(iSlice);
        float B[3];
        prop.GetBxByBz(alpha, t.GetX(), t.GetY(), t.GetZ(), B);
        // B[0]=0;
        // B[1]=0;
        float dLp = 0;
        err = t.PropagateToXBxByBz(xRow, B[0], B[1], B[2], dLp);

        double cs = TMath::Cos(alpha);
        double sn = TMath::Sin(alpha);
        const double kCLight = 0.000299792458;
        double b[3] = {(B[0] * cs - B[1] * sn) / kCLight, (B[0] * sn + B[1] * cs) / kCLight, B[2] / kCLight};
        err = err & !t0.PropagateToBxByBz(xRow, b);
        // err = err & !t0.PropagateTo( xRow, b[2] );
        if (err) {
          std::cout << "Can not propagate to x = " << xRow << std::endl;
          t.Print();
          break;
        }
        if (fabsf(t.GetZ()) >= 250.) {
          std::cout << "Can not propagate to x = " << xRow << ": Z outside the volume" << std::endl;
          t.Print();
          err = -1;
          break;
        }
        // rotate track coordinate system to current sector
        int isNewSlice = RecalculateSlice(t, t0, iSlice);
        if (!isNewSlice) {
          break;
        } else {
          std::cout << "track " << itr << ": new slice " << iSlice << " at row " << iRow << std::endl;
        }
      }
      if (err) {
        break;
      }
      // std::cout<<" track "<<itr<<": Slice "<<iSlice<<" row "<<iRow<<" params :"<<std::endl;
      // t.Print();
      // track at row iRow, slice iSlice
      t.UpdateValues();

      double dx = 1.e4 * (t.GetX() - t0.GetX());
      double dy = 1.e4 * (t.GetY() - t0.GetY());
      double dz = 1.e4 * (t.GetZ() - t0.GetZ());
      hDiff[0]->Fill(dx);
      hDiff[1]->Fill(dy);
      hDiff[2]->Fill(dz);
      // cout<<" x "<<xRow<<" dx "<<dx<<" dy "<<dy<<" dz "<<dz<<endl;
    } // iRow
  }   // itr

  // finish

  TFile* tout = new TFile("propagate.root", "RECREATE");
  TCanvas* c = new TCanvas("PropagatorErrors", "Propagator Errors", 0, 0, 700, 700. * 2. / 3.);
  c->Divide(3);
  int ipad = 1;
  for (int i = 0; i < 3; i++) {
    c->cd(ipad++);

    if (tout) {
      hDiff[i]->Write();
    }
    gPad->SetLogy();
    hDiff[i]->Draw();
  }
  c->Print("propagatorErrors.pdf");
  delete c;
  if (tout) {
    tout->Close();
    delete tout;
  }

  return 0;
}
