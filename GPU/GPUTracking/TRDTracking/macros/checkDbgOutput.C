// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file checkDbgOutput.C
/// \brief Macro for analyzing TRD tracker debug output

/// \author Ole Schmidt

/*
   Usage:
   .L checkDbgOutput.C+
   InitAnalysis();
   FitAngularResolution();
 */

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TROOT.h"
#include "TStyle.h"
#include "TFile.h"
#include "TTree.h"
#include "TVectorF.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TStatToolkit.h"
#include "TString.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TLegend.h"
#endif

TFile* f = 0x0;    // input file (TRDhlt.root)
TTree* tree = 0x0; // input tree (tracksFinal)
TFile* fOut = 0x0; // output file with results

static const Float_t xDrift = 3.f; // drift length in TRD

Float_t mRadiusTRD[540] = {
  // values obtained from TRD geometry with misalignment from Run 2 data
  301.1638, 313.6753, 326.2809, 338.9940, 351.6026, 364.0537, 300.1665, 312.7854, 325.2572, 338.0002,
  350.5855, 363.2468, 299.9399, 312.6280, 325.1296, 337.8112, 350.4096, 362.9868, 300.0480, 312.6387,
  325.2549, 337.8503, 350.4702, 363.1088, 300.6183, 313.1226, 325.6801, 336.4037, 351.1088, 363.7209,
  301.0214, 313.5769, 325.9181, 338.8168, 351.3764, 364.0374, 300.4273, 313.0337, 325.5900, 338.2114,
  350.3062, 363.4128, 300.4677, 312.4737, 325.6458, 338.2561, 350.8812, 363.4771, 300.6205, 313.2002,
  325.1768, 338.4358, 351.0317, 363.6595, 301.0286, 313.5724, 326.1777, 338.8574, 351.4896, 364.1179,
  300.9745, 313.4982, 326.1171, 338.7638, 351.3966, 363.9760, 300.3126, 312.8979, 325.4928, 338.0868,
  350.6675, 363.3101, 300.0797, 312.6622, 325.2723, 337.8378, 350.4357, 363.0448, 300.1662, 312.7673,
  325.3743, 337.9777, 350.5836, 363.2047, 300.5636, 313.1315, 325.7466, 338.3991, 351.0360, 363.6499,
  300.0736, 312.5803, 325.2037, 337.8509, 350.4709, 363.0919, 299.4032, 311.9627, 324.5612, 337.1712,
  349.7448, 362.3622, 299.0729, 311.6657, 324.2644, 336.8565, 349.4228, 362.0316, 299.3591, 311.9435,
  324.5541, 337.1744, 349.7820, 362.3092, 299.7018, 312.2701, 324.9075, 337.5642, 350.1906, 362.8041,
  300.2838, 312.7707, 325.4074, 338.0439, 350.6742, 363.2568, 299.4787, 312.0710, 324.6480, 337.2245,
  349.8300, 362.4292, 299.1346, 311.7354, 324.3476, 336.9353, 349.5186, 362.1122, 299.3472, 311.9391,
  324.5588, 337.1538, 349.7728, 362.3968, 299.7233, 312.2910, 324.9137, 337.5317, 350.1523, 362.7660,
  300.8478, 313.3206, 325.9622, 338.6137, 351.2384, 363.7973, 300.4421, 313.0483, 325.6368, 338.2621,
  350.8496, 363.5216, 300.2939, 312.8903, 325.4760, 338.0599, 350.6551, 363.2704, 300.2710, 312.8796,
  325.4750, 338.0652, 350.6739, 363.2903, 300.6549, 313.2075, 325.8105, 338.4437, 351.0597, 363.6997,
  300.3321, 312.8056, 325.4342, 338.0782, 350.7090, 363.2995, 299.7911, 312.3856, 324.9652, 337.5777,
  350.1493, 362.7679, 299.6359, 312.2484, 324.8446, 337.4233, 350.0143, 362.6087, 299.7957, 312.3872,
  324.9664, 337.5905, 350.1895, 362.8127, 300.1930, 312.7411, 325.3527, 337.9474, 350.5567, 363.1714,
  299.8708, 312.4178, 325.0097, 337.6199, 350.2367, 362.8197, 299.1785, 311.7572, 324.3414, 337.3875,
  349.5200, 362.1488, 299.4955, 312.1514, 324.7430, 337.3484, 349.4956, 362.4435, 300.2071, 313.4209,
  325.7401, 338.3114, 351.2445, 363.8318, 301.1368, 313.7283, 323.1139, 339.0000, 348.2148, 364.2878,
  300.2202, 311.5885, 325.3409, 337.9732, 350.6152, 363.1797, 299.4919, 312.0841, 324.6882, 337.2600,
  349.8945, 362.5386, 299.8091, 312.4409, 325.0175, 337.6343, 350.2683, 362.9218, 300.8080, 313.4294,
  326.0444, 338.6830, 351.3325, 364.0310, 302.0378, 314.4691, 327.2754, 339.9633, 352.7125, 365.4345,
  299.9575, 312.4977, 325.0936, 337.7063, 350.2953, 362.9128, 299.5390, 311.1592, 324.7299, 337.3160,
  349.9368, 362.5510, 299.5108, 312.1093, 324.7206, 337.2922, 349.9214, 362.4301, 299.8687, 312.4665,
  325.1017, 337.6802, 350.2963, 362.8917, 300.9908, 313.4976, 326.1486, 338.7900, 351.4818, 363.9900,
  300.6823, 313.2546, 325.9451, 338.5287, 351.1094, 363.7102, 300.0946, 312.6830, 324.9392, 337.8933,
  350.4915, 363.1127, 299.8324, 312.4263, 325.0230, 337.6105, 350.2232, 362.8239, 299.3513, 311.9513,
  324.7056, 337.9153, 350.5267, 363.1600, 300.4924, 313.0683, 325.7024, 338.3462, 350.9743, 363.6421,
  300.7480, 313.3162, 325.9303, 338.5727, 351.1999, 361.9792, 300.3176, 312.9042, 325.4741, 338.1341,
  350.7175, 363.3588, 300.1988, 312.8076, 325.3764, 337.9723, 350.5598, 363.1935, 300.2925, 312.8776,
  325.4606, 338.0796, 350.6691, 363.2837, 300.7752, 313.2974, 325.9370, 338.5706, 351.1929, 363.7629,
  301.4623, 314.0134, 326.6686, 339.3200, 351.9519, 364.5737, 300.6925, 313.2856, 325.9172, 338.5039,
  351.1359, 363.7670, 300.2042, 312.8122, 325.4005, 337.9800, 350.5984, 363.2198, 300.2045, 312.8112,
  325.3964, 337.9856, 350.5974, 363.1866, 300.5374, 313.1093, 325.7511, 338.3422, 350.9572, 363.0667,
  301.8664, 314.4035, 327.0365, 339.7027, 352.3793, 364.9678, 301.2704, 313.8376, 326.4926, 339.1151,
  351.7640, 364.3762, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 300.4171, 313.0230,
  325.6084, 338.2077, 350.8230, 363.4106, 300.7245, 313.2963, 325.9065, 338.5176, 351.1591, 363.7482,
  301.1933, 313.7285, 326.4055, 339.0260, 351.7000, 364.3218, 300.6251, 313.2307, 325.8524, 338.4811,
  351.0968, 363.7139, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 300.0714, 312.6378,
  325.2482, 337.8352, 350.4462, 363.0263, 300.4317, 312.9689, 325.5656, 338.2149, 350.8201, 363.4381,
  301.1367, 313.6926, 326.3340, 338.9482, 351.6006, 364.2099, 300.6340, 313.2314, 325.8670, 338.4652,
  351.1076, 363.7123, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 299.9679, 312.5547,
  325.1368, 337.7163, 350.3134, 362.9132, 300.3560, 312.9071, 325.4938, 338.1319, 350.7675, 363.3560,
  301.6351, 314.1871, 326.8156, 339.4562, 352.0907, 364.7228, 300.8649, 313.4675, 326.0716, 338.7047,
  350.1673, 363.9594, 300.3626, 312.4153, 325.4608, 338.1805, 350.7680, 363.3876, 300.2386, 312.8433,
  325.4392, 338.0548, 350.6596, 363.2844, 300.7667, 313.3035, 325.9411, 338.6026, 351.2442, 363.8458,
  301.1233, 313.6592, 326.2866, 338.9925, 351.5959, 364.2216, 300.4396, 313.0188, 325.6150, 338.2195,
  350.8347, 363.4567, 300.2179, 312.8220, 325.4206, 337.9880, 350.5974, 363.2124, 300.2803, 312.8911,
  325.4789, 338.1055, 350.7065, 363.3173, 300.9001, 313.4750, 326.0733, 338.7448, 0.0000, 363.9888};

// branches
Int_t event = 0x0;
Int_t nTPCtracks = 0x0;
Int_t nTracklets = 0x0;
Int_t nTrackletsOffline = 0x0;
Int_t nLayers = 0x0;
Float_t chi2Total = 0x0;
Float_t trackPtTPC = 0x0;
Int_t nRelated = 0x0;
Int_t nRelatedOffline = 0x0;
Int_t nMatching = 0x0;
Int_t nMatchingOffline = 0x0;
Int_t nFake = 0x0;
Int_t nFakeOffline = 0x0;
Int_t trackID = 0x0;
Int_t trackIDref = 0x0;
Int_t trackPID = 0x0;
TVectorF* update = 0x0;
TVectorF* trackX = 0x0;
TVectorF* trackY = 0x0;
TVectorF* trackYerr = 0x0;
TVectorF* trackPhi = 0x0;
TVectorF* trackLambda = 0x0;
TVectorF* trackQPt = 0x0;
TVectorF* trackPt = 0x0;
TVectorF* trackZ = 0x0;
TVectorF* trackZerr = 0x0;
TVectorF* trackSec = 0x0;
TVectorF* trackXreal = 0x0;
TVectorF* trackYreal = 0x0;
TVectorF* trackZreal = 0x0;
TVectorF* trackSecReal = 0x0;
TVectorF* trackNoUpX = 0x0;
TVectorF* trackNoUpY = 0x0;
TVectorF* trackNoUpZ = 0x0;
TVectorF* trackNoUpYerr = 0x0;
TVectorF* trackNoUpZerr = 0x0;
TVectorF* trackNoUpPhi = 0x0;
TVectorF* trackNoUpSec = 0x0;
TVectorF* roadY = 0x0;
TVectorF* roadZ = 0x0;
TVectorF* findable = 0x0;
TVectorF* findableMC = 0x0;
TVectorF* nMatchingTracklets = 0x0;
TVectorF* trackletX = 0x0;
TVectorF* trackletY = 0x0;
TVectorF* trackletZ = 0x0;
TVectorF* trackletYRaw = 0x0;
TVectorF* trackletZRaw = 0x0;
TVectorF* trackletDy = 0x0;
TVectorF* trackletDet = 0x0;
TVectorF* trackletXreal = 0x0;
TVectorF* trackletYreal = 0x0;
TVectorF* trackletZreal = 0x0;
TVectorF* trackletYRawreal = 0x0;
TVectorF* trackletZRawreal = 0x0;
TVectorF* trackletSecReal = 0x0;
TVectorF* trackletDetReal = 0x0;
TVectorF* chi2Update = 0x0;
TVectorF* chi2Real = 0x0;
Float_t XvMC = 0x0;
Float_t YvMC = 0x0;
Float_t ZvMC = 0x0;
//

static const Int_t nCandidatesSweep = 3;
static const Int_t maxMissingLayersSweep = 1;
static const Int_t chi2PenaltySweep = 1;

static const Int_t nCandidates[nCandidatesSweep] = {1, 2, 3};
static const Int_t maxMissingLayers[maxMissingLayersSweep] = {4};
static const Int_t chi2Penalty[chi2PenaltySweep] = {16};

// functions
Bool_t InitAnalysis(const char* filename = "TRDhlt.root", Bool_t isMC = kTRUE);
void GetEfficiency(TH1F* hEfficiency);
void GetFakeRate(TH1F* hFakeRate);
void Reset();

TF1* fitRMSAngle = new TF1("fitRMSAngle", "sqrt([0]**2+[2]**2*(x-[1])**2)", -1, 1);       // RMS of angular residuals as function of track phi
TF1* fitRMSDelta012 = new TF1("fitRMSdelta012", "sqrt([0]**2+[2]**2*(x-[1])**2)", -1, 1); // RMS of r-phi residuals (using interpolation from surrounding layers) as function of track phi
TF1* fitTrackletDy = new TF1("fitTrackletDy", "pol2");                                    // tracklet deflection as function of track phi
TF1* fitDAngleCorr = new TF1("fitDAngleCorr", "pol1");                                    //

Bool_t InitTree(const char* filename)
{
  f = new TFile(filename, "open");
  if (!f) {
    printf("File could not be opened\n");
    return kFALSE;
  }
  tree = (TTree*)f->Get("tracksFinal");
  if (!tree) {
    printf("Tree could not be opened\n");
    return kFALSE;
  }
  return kTRUE;
}

void InitBranches()
{
  if (!tree) {
    return;
  }

  // for explanations of the variables see GPUTRDTrackerDebug.h
  tree->SetBranchAddress("trackID", &trackID);
  tree->SetBranchAddress("labelRef", &trackIDref);
  tree->SetBranchAddress("pdgCode", &trackPID);
  tree->SetBranchAddress("update.", &update);
  tree->SetBranchAddress("trackX.", &trackX);
  tree->SetBranchAddress("trackY.", &trackY);
  tree->SetBranchAddress("trackYerr.", &trackYerr);
  tree->SetBranchAddress("trackPhi.", &trackPhi);
  tree->SetBranchAddress("trackLambda.", &trackLambda);
  tree->SetBranchAddress("trackQPt.", &trackQPt);
  tree->SetBranchAddress("trackPt.", &trackPt);
  tree->SetBranchAddress("trackZ.", &trackZ);
  tree->SetBranchAddress("trackZerr.", &trackZerr);
  tree->SetBranchAddress("trackSec.", &trackSec);
  tree->SetBranchAddress("trackXReal.", &trackXreal);
  tree->SetBranchAddress("trackYReal.", &trackYreal);
  tree->SetBranchAddress("trackZReal.", &trackZreal);
  tree->SetBranchAddress("trackSecReal.", &trackSecReal);
  tree->SetBranchAddress("trackNoUpX.", &trackNoUpX);
  tree->SetBranchAddress("trackNoUpY.", &trackNoUpY);
  tree->SetBranchAddress("trackNoUpZ.", &trackNoUpZ);
  tree->SetBranchAddress("trackNoUpYerr.", &trackNoUpYerr);
  tree->SetBranchAddress("trackNoUpZerr.", &trackNoUpZerr);
  tree->SetBranchAddress("trackNoUpPhi.", &trackNoUpPhi);
  tree->SetBranchAddress("trackNoUpSec.", &trackNoUpSec);
  tree->SetBranchAddress("roadY.", &roadY);
  tree->SetBranchAddress("roadZ.", &roadZ);
  tree->SetBranchAddress("findable.", &findable);
  tree->SetBranchAddress("findableMC.", &findableMC);
  tree->SetBranchAddress("nMatchingTracklets.", &nMatchingTracklets);
  tree->SetBranchAddress("trackletX.", &trackletX);
  tree->SetBranchAddress("trackletY.", &trackletY);
  tree->SetBranchAddress("trackletDy.", &trackletDy);
  tree->SetBranchAddress("trackletZ.", &trackletZ);
  tree->SetBranchAddress("trackletYRaw.", &trackletYRaw);
  tree->SetBranchAddress("trackletZRaw.", &trackletZRaw);
  tree->SetBranchAddress("trackletDet.", &trackletDet);
  tree->SetBranchAddress("trackletXReal.", &trackletXreal);
  tree->SetBranchAddress("trackletYReal.", &trackletYreal);
  tree->SetBranchAddress("trackletZReal.", &trackletZreal);
  tree->SetBranchAddress("trackletYRawReal.", &trackletYRawreal);
  tree->SetBranchAddress("trackletZRawReal.", &trackletZRawreal);
  tree->SetBranchAddress("trackletSecReal.", &trackletSecReal);
  tree->SetBranchAddress("trackletDetReal.", &trackletDetReal);
  tree->SetBranchAddress("chi2Update.", &chi2Update);
  tree->SetBranchAddress("chi2Real.", &chi2Real);
  tree->SetBranchAddress("XvMC", &XvMC);
  tree->SetBranchAddress("YvMC", &YvMC);
  tree->SetBranchAddress("ZvMC", &ZvMC);
  tree->SetBranchAddress("event", &event);
  tree->SetBranchAddress("nTPCtracks", &nTPCtracks);
  tree->SetBranchAddress("nTracklets", &nTracklets);
  tree->SetBranchAddress("nTrackletsOffline", &nTrackletsOffline);
  tree->SetBranchAddress("nLayers", &nLayers);
  tree->SetBranchAddress("chi2Total", &chi2Total);
  tree->SetBranchAddress("trackPtTPC", &trackPtTPC);
  tree->SetBranchAddress("nRelated", &nRelated);
  tree->SetBranchAddress("nMatching", &nMatching);
  tree->SetBranchAddress("nFake", &nFake);
  tree->SetBranchAddress("nTrackletsOfflineRelated", &nRelatedOffline);
  tree->SetBranchAddress("nTrackletsOfflineMatch", &nMatchingOffline);
  tree->SetBranchAddress("nTrackletsOfflineFake", &nFakeOffline);
}

void InitCalib()
{
  fitTrackletDy->SetParameters(0.14, 2.35, -0.635);
  fitRMSAngle->SetParameters(0.0473, 0.163, 0.427);
  fitRMSDelta012->SetParameters(0.08, 0.08, 0.34);
}

void SetAlias(TTree* tree, Bool_t isMC)
{
  if (!tree) {
    return;
  }

  tree->SetAlias("layer", "Iteration$");
  tree->SetAlias("isPresent", "update.fElements>0");
  tree->SetAlias("matchAvail", "nMatchingTracklets.fElements>0");
  tree->SetAlias("geoFindable", "findable.fElements>0");
  if (isMC) {
    tree->SetAlias("isGold", "nMatching==6");
    tree->SetAlias("isMatch", "isPresent&&update.fElements<3.5");
    tree->SetAlias("isRelated", "update.fElements>3.5&&update.fElements<8");
    tree->SetAlias("isGood", "isMatch||isRelated");
    tree->SetAlias("isFake", "isPresent&&update.fElements>8");
  } else {
    // for data we can only distinguish between there is a tracklet or there is none
    tree->SetAlias("isGold", "nTracklets==6");
    tree->SetAlias("isMatch", "isPresent");
    tree->SetAlias("isRelated", "isPresent");
    tree->SetAlias("isGood", "isPresent");
    tree->SetAlias("isFake", "isPresent");
  }
  tree->SetAlias("lowMult", "nTPCtracks<200");

  tree->SetAlias("roadY", "roadY.fElements");
  tree->SetAlias("roadZ", "roadZ.fElements");
  tree->SetAlias("inYroad", "(abs(GetDeltaYmatch(layer))-roadY)<0");
  tree->SetAlias("inZroad", "(abs(GetDeltaZmatch(layer))-roadZ)<0");
  tree->SetAlias("trkPhi", "trackPhi.fElements");
  tree->SetAlias("trkltDy", "trackletDy.fElements");
  tree->SetAlias("resY", "(trackY.fElements-trackletY.fElements)");
  tree->SetAlias("resZ", "(trackZ.fElements-trackletZ.fElements)");
  tree->SetAlias("pullY", "resY/sqrt(trackYerr.fElements+trackletYerr.fElements)");
  tree->SetAlias("pullZ", "resZ/sqrt(trackZerr.fElements+trackletZerr.fElements)");
  tree->SetAlias("isIdealAngle", "abs(trackPhi.fElements-0.12)<0.05");
}

Int_t LoadBranches(Int_t entry)
{
  static Int_t lastEntry = -1;
  if (lastEntry != entry) {
    tree->GetEntry(entry);
    lastEntry = entry;
  }
  return 1;
}

Int_t GetStack(Float_t det)
{
  Int_t detInt = TMath::Nint(det);
  return (detInt % 30) / 6;
}

// convert azimuthal track angle (sin(phi) = param[3]) to dy of tracklet
Double_t AngleToDy(Double_t tanPhi) { return fitTrackletDy->Eval(tanPhi); }
// convert dy of tracklet to track azimuthal angle sin(phi)
Double_t DyToSnp(Double_t dy) { return ((dy / xDrift) / TMath::Sqrt(1 + TMath::Power((dy / xDrift), 2))); }
// convert sin(phi) to dy
Double_t SnpToDy(Double_t snp) { return (xDrift * snp / TMath::Sqrt(1 - TMath::Power(snp, 2))); }
// TRD angular resolution at given angle (best resolution at lorentz angle)
Double_t GetAngularResolution(Double_t tanPhi) { return fitRMSAngle->Eval(tanPhi); }

// angular pull for tracklet in given layer
Double_t GetPullAngle(Int_t layer)
{
  if (layer < 0 || layer > 5) {
    return -999;
  }
  if ((*update)[layer] < 1) {
    return -666;
  }
  Double_t trackletDyLayer = (*trackletDy)[layer];
  Double_t trackletDyFit = AngleToDy((*trackPhi)[layer]);
  Double_t dYresolution = GetAngularResolution((*trackPhi)[layer]);
  if (dYresolution > 0) {
    return (trackletDyLayer - trackletDyFit) / dYresolution;
  }
  return -999;
}

// angular residuals in given layer
Double_t GetDeltaAngle(Int_t layer)
{
  if (layer < 0 || layer > 5) {
    return -999;
  }
  if ((*update)[layer] < 1) {
    return -666;
  }
  Double_t trackletDyLayer = (*trackletDy)[layer];
  Double_t trackletDyF = AngleToDy((*trackPhi)[layer]);
  return (trackletDyLayer - trackletDyF);
}

Double_t GetEta(Float_t tgl)
{
  Double_t theta = TMath::Pi() / 2. - TMath::ATan(tgl);
  return -TMath::Log(TMath::Tan(theta * .5));
}

Double_t GetDeltaR(Int_t layer)
{
  if (layer < 0 || layer > 5 || (*trackletXreal)[layer] < 10) {
    return -999;
  }
  if (TMath::Abs(GetEta((*trackLambda)[layer])) > 0.05) {
    return 999;
  }
  return (*trackletXreal)[layer] - mRadiusTRD[TMath::Nint((*trackletDetReal)[layer])];
}

// tracklet residual in rphi, either based on interpolated tracklet position from the surrounding layers
// or only on the tracklet position in given layer
Double_t GetDeltaRPhi(Int_t layer, Bool_t interpolation = kFALSE, Bool_t rawTrkltPosition = kFALSE)
{
  if ((layer < 0) || (layer > 5) || ((*update)[layer] < 0.5)) {
    return -999;
  }
  if (interpolation && (layer < 1 || layer > 4)) {
    return -999;
  }
  Double_t tilt = TMath::Tan(TMath::Pi() / 90.);
  Double_t trkltYpos = 0;
  if (rawTrkltPosition) {
    trkltYpos = (*trackletYRaw)[layer] + ((*trackletZRaw)[layer] - (*trackNoUpZ)[layer]) * tilt * (-1 + 2 * ((layer % 2) == 0));
  } else {
    trkltYpos = (*trackletY)[layer];
  }
  if (!interpolation) {
    return (trkltYpos - (*trackNoUpY)[layer]);
  }
  Int_t counter = 0;
  Double_t deltaY[3];
  for (Int_t iLy = layer - 1; iLy <= layer + 1; ++iLy) {
    if ((*update)[iLy] < 1) {
      continue;
    }
    if (rawTrkltPosition) {
      deltaY[counter] = (*trackletYRaw)[iLy] + ((*trackletZRaw)[iLy] - (*trackNoUpZ)[iLy]) * tilt * (-1 + 2 * ((iLy % 2) == 0)) - (*trackNoUpY)[iLy];
    } else {
      deltaY[counter] = (*trackletY)[iLy] - (*trackNoUpY)[iLy];
    }
    ++counter;
  }
  if (counter < 3) {
    return -999;
  }
  return (deltaY[1] - 0.5 * (deltaY[0] + deltaY[2]));
}

void FitAngularResolution()
{
  if (!tree) {
    return;
  }
  fOut = new TFile("results.root", "update");
  TGraph* gr = TStatToolkit::MakeGraphErrors(tree, "trackletDy.fElements:trackPhi.fElements", "isGold&&LoadBranches(Entry$)", 25, 1, 1, 0, 100000);
  gr->Fit("pol2", "rob=0.9", "", -0.3, 0.3);
  tree->SetAlias("trackletDyFit", TString::Format("(%f)+(%f)*trackPhi.fElements+(%f)*trackPhi.fElements^2", gr->GetFunction("pol2")->GetParameter(0), gr->GetFunction("pol2")->GetParameter(1), gr->GetFunction("pol2")->GetParameter(2)));
  gr->Draw("ap");
  fitTrackletDy->SetParameters(gr->GetFunction("pol2")->GetParameter(0), gr->GetFunction("pol2")->GetParameter(1), gr->GetFunction("pol2")->GetParameter(2));
  tree->Draw("trackletDy.fElements-trackletDyFit:trackPhi.fElements>>hisDyPhi(20, -0.25, 0.25, 50, -0.3, 0.3)", "isGold", "colz");
  TH2* hisDyPhi = (TH2*)tree->GetHistogram();
  hisDyPhi->FitSlicesY();
  TH1* hAngularResolution = (TH1*)gROOT->FindObject("hisDyPhi_2");
  fitRMSAngle->SetParameters(0.05, 0.05, 0.1);
  hAngularResolution->Fit(fitRMSAngle);
  hAngularResolution->SetTitle("");
  hAngularResolution->GetXaxis()->SetTitle("track #phi");
  hAngularResolution->GetYaxis()->SetTitle("RMS (tracklet defl. - fit tracklet defl.)");
  gStyle->SetOptStat(0);
  hAngularResolution->Draw();
  fitRMSAngle->Draw("same");
  hAngularResolution->Write();
  delete hAngularResolution;
  fitRMSAngle->Write();
  delete fitRMSAngle;
  fOut->Close();
  delete fOut;
}

void FitPositionResolution()
{
  if (!tree) {
    return;
  }
  fOut = new TFile("results.root", "update");
  tree->Draw("GetDeltaRPhi(layer, 1, 0):trackNoUpPhi.fElements>>hisDeltaRPhi(15, -0.4, 0.4, 50, -0.3, 0.3)", "isGold&&lowMult&&LoadBranches(Entry$)", "colz");
  ((TH2*)tree->GetHistogram())->FitSlicesY();
  ((TH1*)gROOT->FindObject("hisDeltaRPhi_2"))->Fit(fitRMSDelta012);
  gPad->SaveAs("FitPositionResolution.png");
  TH1* hPositionResolution = (TH1*)gROOT->FindObject("hisDeltaRPhi_2");
  hPositionResolution->Draw();
  hPositionResolution->Write();
  delete hPositionResolution;
  fitRMSDelta012->Write();
  delete fitRMSDelta012;
  fOut->Close();
  delete fOut;
}

void FitRPhiVsChamber(Bool_t rawTrkltPosition = kTRUE)
{
  if (!tree) {
    return;
  }
  if (rawTrkltPosition) {
    tree->Draw("GetDeltaRPhi(layer, 1, 1):trackletDet.fElements>>hisRPhiDet(540, -0.5, 539.5, 50, -0.5, 0.5)", "isGold&&lowMult&&LoadBranches(Entry$)", "colz");
    ((TH2*)tree->GetHistogram())->FitSlicesY();
    ((TH1*)gROOT->FindObject("hisRPhiDet_1"))->Draw();
  } else {
    tree->Draw("GetDeltaRPhi(layer, 1, 0):trackletDet.fElements>>hisRPhiDet(540, -0.5, 539.5, 50, -0.5, 0.5)", "isGold&&lowMult&&LoadBranches(Entry$)", "colz");
    ((TH2*)tree->GetHistogram())->FitSlicesY();
    ((TH1*)gROOT->FindObject("hisRPhiDet_1"))->Draw();
  }
}

void CheckRadialOffset()
{
  TH2F* hRadialOffset = new TH2F("radialOffset", "", 540, 0, 540, 100, -3, 3);
  for (Int_t i = 0; i < tree->GetEntriesFast(); ++i) {
    tree->GetEntry(i);
    if (trackID < 0) {
      continue;
    }
    if (nMatching != 6) {
      continue;
    }
    //if ((*trackPhi)[0] < 0.15) {
    //  continue;
    //}
    for (Int_t iLy = 0; iLy < 6; iLy++) {
      Double_t phi = (*trackPhi)[iLy];
      Double_t deltaRPhi = (*trackletY)[iLy] - (*trackNoUpY)[iLy];
      Int_t det = (*trackletDet)[iLy];
      Double_t deltaR = deltaRPhi / TMath::Tan(TMath::ASin(phi));
      hRadialOffset->Fill(det, deltaR);
    }
  }
  hRadialOffset->Draw("colz");
}

void FitRadialSpreadTrklts()
{
  if (!tree) {
    return;
  }
  tree->Draw("GetDeltaR(layer):trackletDetReal.fElements>>hisRDet(540, -0.5, 539.5, 50, -5, 5)", "LoadBranches(Entry$)", "colz");
  ((TH2*)tree->GetHistogram())->FitSlicesY();
  ((TH1*)gROOT->FindObject("hisRDet_2"))->Draw();
}

void PrintEfficiency(Float_t ptCut = 1.5)
{
  if (!tree) {
    printf("Input tree not found\n");
    return;
  }
  TString cutMatchAvailable = TString::Format("matchAvail&&trackPt.fElements>%f", ptCut);
  TString cutMatchAvailableUpdate = TString::Format("isPresent&&matchAvail&&trackPt.fElements>%f", ptCut);
  TString cutPresent = TString::Format("isPresent&&trackPt.fElements>%f", ptCut);
  TString cutReal = TString::Format("isMatch&&trackPt.fElements>%f", ptCut);
  TString cutRelated = TString::Format("isRelated&&trackPt.fElements>%f", ptCut);
  TString cutFake = TString::Format("isFake&&trackPt.fElements>%f", ptCut);
  Double_t nUpdatesAvail = tree->Draw("update.fElements", cutMatchAvailable.Data());
  Double_t nUpdatesAvailUpdate = tree->Draw("update.fElements", cutMatchAvailableUpdate.Data());
  Double_t nUpdatesTotal = tree->Draw("update.fElements", cutPresent.Data());
  Double_t nUpdatesMatch = tree->Draw("update.fElements", cutReal.Data());
  Double_t nUpdatesRelated = tree->Draw("update.fElements", cutRelated.Data());
  Double_t nUpdatesFake = tree->Draw("update.fElements", cutFake.Data());
  printf("--- without deflection cut (pT > %.2f GeV) ---\n", ptCut);
  printf("Total number of updates: %.0f\n", nUpdatesTotal);
  printf("Fraction of matches: %.3f\n", nUpdatesMatch / nUpdatesTotal);
  printf("Fraction of relatives: %.3f\n", nUpdatesRelated / nUpdatesTotal);
  printf("Fraction of fakes: %.3f\n", nUpdatesFake / nUpdatesTotal);
  printf("---\n");
  printf("Available matches %.0f, fraction with updates %.3f\n", nUpdatesAvail, nUpdatesAvailUpdate / nUpdatesAvail);
  gPad->Close();
}

void Sweep()
{
  // check efficiency and fake rate for different tracker parameters
  fOut = new TFile("sweep-results.root", "recreate");
  fOut->Close();

  for (Int_t iCandidates = 0; iCandidates < nCandidatesSweep; ++iCandidates) {
    Int_t nCan = nCandidates[iCandidates];
    for (Int_t iMaxMissingLayers = 0; iMaxMissingLayers < maxMissingLayersSweep; ++iMaxMissingLayers) {
      Int_t nMaxLy = maxMissingLayers[iMaxMissingLayers];
      for (Int_t iChi2Penalty = 0; iChi2Penalty < chi2PenaltySweep; ++iChi2Penalty) {
        Int_t chi2 = chi2Penalty[iChi2Penalty];
        TString filename = TString::Format("%i/%i/%i/TRDhlt.root", nCan, nMaxLy, chi2);
        if (!InitAnalysis(filename.Data(), 1)) {
          printf("Could not open %s. Skipping it...\n", filename.Data());
          continue;
        }
        TFile fOut("sweep-results.root", "update");
        TH1F* hEfficiency = new TH1F(TString::Format("eff%iCands%iMaxLy%iChi2", nCan, nMaxLy, chi2), ";p_{T} (GeV/#it{c});efficiency", 10, 0, 10);
        printf("\n---\n Checking nCandidates=%i, maxMissingLayers=%i, chi2Penalty=%i \n---\n", nCan, nMaxLy, chi2);
        PrintEfficiency(.8);
        GetEfficiency(hEfficiency);
        TH1F* hFakeRate = new TH1F(TString::Format("fakes%iCands%iMaxLy%iChi2", nCan, nMaxLy, chi2), ";p_{T} (GeV/#it{c});fraction of matches", 10, 0, 10);
        GetFakeRate(hFakeRate);
        hEfficiency->Write();
        delete hEfficiency;
        hFakeRate->Write();
        delete hFakeRate;
        fOut.Close();
        Reset();
      }
    }
  }
  delete fOut;
  fOut = 0x0;
}

void ParameterSweepResults()
{
  TFile fIn("sweep-results.root", "read");
  if (!fIn.IsOpen()) {
    return;
  }

  TCanvas* c1 = new TCanvas("c1", "c1");
  gStyle->SetErrorX(0);

  TLegend* legend = new TLegend(0.2, 0.15, 0.8, 0.45);
  legend->SetNColumns(3);

  TH1F* hEfficiency[nCandidatesSweep * maxMissingLayersSweep * chi2PenaltySweep] = {0x0};
  TH1F* hFakeRate[nCandidatesSweep * maxMissingLayersSweep * chi2PenaltySweep] = {0x0};

  std::vector<Int_t> missingFiles;

  for (Int_t iCandidates = 0; iCandidates < nCandidatesSweep; ++iCandidates) {
    Int_t nCan = nCandidates[iCandidates];
    for (Int_t iMaxMissingLayers = 0; iMaxMissingLayers < maxMissingLayersSweep; ++iMaxMissingLayers) {
      Int_t nMaxLy = maxMissingLayers[iMaxMissingLayers];
      for (Int_t iChi2Penalty = 0; iChi2Penalty < chi2PenaltySweep; ++iChi2Penalty) {
        Int_t chi2 = chi2Penalty[iChi2Penalty];
        Int_t index = iCandidates * chi2PenaltySweep * maxMissingLayersSweep + iMaxMissingLayers * chi2PenaltySweep + iChi2Penalty;
        TString histName1 = TString::Format("eff%iCands%iMaxLy%iChi2", nCan, nMaxLy, chi2);
        TString histName2 = TString::Format("fakes%iCands%iMaxLy%iChi2", nCan, nMaxLy, chi2);
        TString legName = TString::Format("nCand(%i) max missing ly(%i) chi2(%i)", nCan, nMaxLy, chi2);
        hEfficiency[index] = (TH1F*)fIn.Get(histName1.Data());
        hFakeRate[index] = (TH1F*)fIn.Get(histName2.Data());
        if (!hEfficiency[index] || !hFakeRate[index]) {
          printf("Could not get histogram(s) %i\n", index);
          missingFiles.push_back(index);
          continue;
        }
        hFakeRate[index]->SetStats(0);
        hFakeRate[index]->SetMarkerStyle(22);
        hFakeRate[index]->SetMarkerColor(1 + index);
        hFakeRate[index]->SetLineColor(1 + index);
        hFakeRate[index]->GetYaxis()->SetRangeUser(0.9, 1.05);
        hEfficiency[index]->SetStats(0);
        hEfficiency[index]->SetMarkerStyle(22);
        hEfficiency[index]->SetMarkerColor(1 + index);
        hEfficiency[index]->SetLineColor(1 + index);
        hEfficiency[index]->GetYaxis()->SetRangeUser(0.5, 1.00);
        hEfficiency[index]->Draw("same e0");
        legend->AddEntry(hEfficiency[index], legName.Data(), "lp");
      }
    }
  }
  legend->Draw();
  c1->SetGridx();
  c1->SetGridy();
  c1->SaveAs("efficiency.pdf");
  c1->SaveAs("efficiency.png");
  gPad->Close();
  TCanvas* c2 = new TCanvas("c2", "c2");
  for (Int_t idx = 0; idx < nCandidatesSweep * maxMissingLayersSweep * chi2PenaltySweep; ++idx) {
    if (std::find(missingFiles.begin(), missingFiles.end(), idx) == missingFiles.end()) {
      hFakeRate[idx]->Draw("same e0");
    }
  }
  legend->Draw();
  c2->SetGridx();
  c2->SetGridy();
  c2->SaveAs("purity.pdf");
  c2->SaveAs("purity.png");
  gPad->Close();
}

void GetEfficiency(TH1F* hEfficiency)
{
  TAxis* ax = hEfficiency->GetXaxis();
  TH1F* hAll = new TH1F("all", "all", ax->GetNbins(), ax->GetBinUpEdge(0), ax->GetBinUpEdge(ax->GetNbins()));
  TH1F* hFrac = new TH1F("frac", "frac", ax->GetNbins(), ax->GetBinUpEdge(0), ax->GetBinUpEdge(ax->GetNbins()));
  hAll->Sumw2();
  hFrac->Sumw2();
  for (Int_t i = 0; i < tree->GetEntriesFast(); ++i) {
    tree->GetEntry(i);
    if (trackID < 0) {
      continue;
    }
    Double_t trkPt = trackPtTPC;
    for (Int_t iLy = 0; iLy < 6; iLy++) {
      if (((*nMatchingTracklets)[iLy] > 0) && ((*findableMC)[iLy] > 0)) {
        hAll->Fill(trkPt);
        if ((*update)[iLy] > 0) {
          hFrac->Fill(trkPt);
        }
      }
    }
  }
  hEfficiency->Divide(hFrac, hAll, 1, 1, "B");
  delete hAll;
  delete hFrac;
}

void GetFakeRate(TH1F* hFakeRate)
{
  TAxis* ax = hFakeRate->GetXaxis();
  TH1F* hAll = new TH1F("all", "all", ax->GetNbins(), ax->GetBinUpEdge(0), ax->GetBinUpEdge(ax->GetNbins()));
  TH1F* hFrac = new TH1F("frac", "frac", ax->GetNbins(), ax->GetBinUpEdge(0), ax->GetBinUpEdge(ax->GetNbins()));
  hAll->Sumw2();
  hFrac->Sumw2();
  for (Int_t i = 0; i < tree->GetEntriesFast(); ++i) {
    tree->GetEntry(i);
    if (trackID < 0) {
      continue;
    }
    Double_t trkPt = trackPtTPC;
    for (Int_t iLy = 0; iLy < 6; iLy++) {
      if ((*update)[iLy] > 0) {
        hAll->Fill(trkPt);
        if ((*update)[iLy] < 8) {
          hFrac->Fill(trkPt);
        }
      }
    }
  }
  hFakeRate->Divide(hFrac, hAll, 1, 1, "B");
  delete hAll;
  delete hFrac;
}

void MakeLogScale(TH1* hist, Double_t xMin = -1, Double_t xMax = -1)
{
  Int_t nBins = hist->GetXaxis()->GetNbins();
  if (xMin < 0) {
    Double_t xMin = hist->GetXaxis()->GetBinCenter(1);
  }
  if (xMin <= 0) {
    printf("No log scale possible (xMin = %f)\n", xMin);
    return;
  }
  if (xMax < 0) {
    Double_t xMax = hist->GetXaxis()->GetBinCenter(nBins);
  }
  Double_t xBins[nBins + 1];
  Double_t logXmin = TMath::Log10(xMin);
  Double_t logXmax = TMath::Log10(xMax);
  Double_t binWidth = (logXmax - logXmin) / nBins;
  for (int i = 0; i <= nBins; i++) {
    xBins[i] = xMin + TMath::Power(10, logXmin + i * binWidth);
  }
  hist->GetXaxis()->Set(nBins, xBins);
}

void TwoTrackletEfficiency(Int_t nEntries = -1)
{
  fOut = new TFile("results.root", "update");
  TH1F* hAll = new TH1F("all", "p_{T} spectrum for all tracks;track pT (GeV); counts", 10, 0, 10);
  TH1F* h2Trklts = new TH1F("h2trklts", "p_{T} spectrum for all tracks w/ N_{trklts} >= 2;track pT (GeV); counts", 10, 0, 10);
  TH1F* h2TrkltsNoFakes = new TH1F("h2trkltsNoFakes", "p_{T} spectrum for all tracks w/ N_{trklts} >= 2 and zero fakes;track pT (GeV); counts", 10, 0, 10);
  TH1F* h2TrkltsRef = new TH1F("h2trkltsRef", "p_{T} spectrum for all tracks w/ N_{trklts} >= 2 (ref);track pT (GeV); counts", 10, 0, 10);
  TH1F* h2TrkltsNoFakesRef = new TH1F("h2trkltsNoFakesRef", "p_{T} spectrum for all tracks w/ N_{trklts} >= 2 and zero fakes (ref);track pT (GeV); counts", 10, 0, 10);
  TH1F* h2TrkltsEff = new TH1F("h2TrkltsEff", "fraction with at least two tracklets;track pT (GeV); counts", 10, 0, 10);
  TH1F* h2TrkltsNoFakesEff = new TH1F("h2TrkltsNoFakesEff", "fraction of two tracklet tracks w/o fakes;track pT (GeV); counts", 10, 0, 10);
  TH1F* h2TrkltsEffRef = new TH1F("h2TrkltsEffRef", "fraction with at least two tracklets (ref);track pT (GeV); counts", 10, 0, 10);
  TH1F* h2TrkltsNoFakesEffRef = new TH1F("h2TrkltsNoFakesEffRef", "fraction of two tracklet tracks w/o fakes(ref);track pT (GeV); counts", 10, 0, 10);

  MakeLogScale(hAll, .2, 12.);
  MakeLogScale(h2Trklts, .2, 12.);
  MakeLogScale(h2TrkltsNoFakes, .2, 12.);
  MakeLogScale(h2TrkltsRef, .2, 12.);
  MakeLogScale(h2TrkltsNoFakesRef, .2, 12.);
  MakeLogScale(h2TrkltsEff, .2, 12.);
  MakeLogScale(h2TrkltsNoFakesEff, .2, 12.);
  MakeLogScale(h2TrkltsEffRef, .2, 12.);
  MakeLogScale(h2TrkltsNoFakesEffRef, .2, 12.);

  hAll->Sumw2();
  h2Trklts->Sumw2();
  h2TrkltsNoFakes->Sumw2();
  h2TrkltsRef->Sumw2();
  h2TrkltsNoFakesRef->Sumw2();

  if (nEntries < 0) {
    nEntries = tree->GetEntriesFast();
  }
  for (Int_t iEntry = 0; iEntry < nEntries; iEntry++) {
    tree->GetEntry(iEntry);
    if (trackID < 0) {
      continue;
    }
    if (findable->Sum() < 2) {
      continue;
    }
    Double_t pt = trackPtTPC;
    hAll->Fill(pt);
    //if (nTracklets >= 2) {
    if (nMatching + nRelated >= 2) {
      h2Trklts->Fill(pt);
      if (nFake == 0) {
        h2TrkltsNoFakes->Fill(pt);
      }
    }
    //if (nTrackletsOffline >= 2) {
    if (nMatchingOffline + nRelatedOffline >= 2) {
      h2TrkltsRef->Fill(pt);
      if (nFakeOffline == 0) {
        h2TrkltsNoFakesRef->Fill(pt);
      }
    }
  }
  gStyle->SetOptStat(0);
  h2TrkltsEff->Divide(h2Trklts, hAll, 1, 1, "B");
  h2TrkltsNoFakesEff->Divide(h2TrkltsNoFakes, h2Trklts, 1, 1, "B");
  h2TrkltsEffRef->Divide(h2TrkltsRef, hAll, 1, 1, "B");
  h2TrkltsNoFakesEffRef->Divide(h2TrkltsNoFakesRef, h2TrkltsRef, 1, 1, "B");

  fOut->cd();
  h2TrkltsEff->Write();
  delete h2TrkltsEff;
  h2TrkltsNoFakesEff->Write();
  delete h2TrkltsNoFakesEff;
  h2TrkltsEffRef->Write();
  delete h2TrkltsEffRef;
  h2TrkltsNoFakesEffRef->Write();
  delete h2TrkltsNoFakesEffRef;
  fOut->Close();
  delete fOut;
  fOut = 0x0;
}



void PlotTRDEfficiency(Int_t nEntries = -1, Bool_t writeToFile = kTRUE)
{
  // plot fraction of tracks with at least 4/5/6 online/offline tracklets
  // independent of TRD online tracker

  if (!tree) {
    printf("Initialize first!\n");
    return;
  }

  if (writeToFile) {
    fOut = new TFile("TRDefficiency.root", "recreate");
  }

  if (nEntries < 0) {
    nEntries = tree->GetEntriesFast();
  }
  TH1F* hAll = new TH1F("all", "tmp histogram;pT (GeV);counts", 6, 0, 5);
  TH1F* hFracOffline[6];
  TH1F* hFracOnline[6];
  TH1F* hEffOffline[6];
  TH1F* hEffOnline[6];
  for (Int_t i = 0; i < 6; ++i) {
    TString hNameFracOff = TString::Format("fracOff%i", i + 1);
    TString hNameFracOn = TString::Format("fracOn%i", i + 1);
    TString hNameEffOff = TString::Format("effOff%i", i + 1);
    TString hNameEffOn = TString::Format("effOn%i", i + 1);
    hFracOffline[i] = new TH1F(hNameFracOff.Data(), "", 6, 0, 5);
    hFracOnline[i] = new TH1F(hNameFracOn.Data(), "", 6, 0, 5);
    hEffOffline[i] = new TH1F(hNameEffOff.Data(), ";p_{T} (GeV/#it{c});efficiency", 6, 0, 5);
    hEffOnline[i] = new TH1F(hNameEffOn.Data(), ";p_{T} (GeV/#it{c});efficiency", 6, 0, 5);
    hFracOffline[i]->Sumw2();
    hFracOnline[i]->Sumw2();
    MakeLogScale(hFracOffline[i], 0.2, 12.);
    MakeLogScale(hFracOnline[i], 0.2, 12.);
    MakeLogScale(hEffOffline[i], 0.2, 12.);
    MakeLogScale(hEffOnline[i], 0.2, 12.);
  }
  hAll->Sumw2();
  MakeLogScale(hAll, 0.2, 12.);
  for (Int_t iEntry = 0; iEntry < nEntries; iEntry++) {
    tree->GetEntry(iEntry);
    if (trackID < 0) {
      continue;
    }
    Double_t trkPt = (*trackPt)[0];
    hAll->Fill(trkPt);
    Int_t nTrackletsRef = nMatchingOffline + nRelatedOffline;
    Int_t nTrackletsOnline = 0;
    for (Int_t iLy = 0; iLy < 6; ++iLy) {
      if ((*nMatchingTracklets)[iLy] > 0) {
        ++nTrackletsOnline;
      }
    }
    for (Int_t nTrkltsMin = 1; nTrkltsMin <= 6; ++nTrkltsMin) {
      if (nTrackletsRef >= nTrkltsMin) {
        hFracOffline[nTrkltsMin - 1]->Fill(trkPt);
      }
      if (nTrackletsOnline >= nTrkltsMin) {
        hFracOnline[nTrkltsMin - 1]->Fill(trkPt);
      }
    }
  }
  gStyle->SetOptStat(0);
  gStyle->SetErrorX(0);
  TCanvas* c1 = 0x0;
  if (!writeToFile) {
    c1 = new TCanvas("c1", "c1");
    c1->SetGridy();
    c1->SetGridx();
    c1->SetLogx();
  }
  Bool_t hasDrawnHist = kFALSE;

  hEffOffline[5]->SetMarkerStyle(24);
  hEffOffline[4]->SetMarkerStyle(25);
  hEffOffline[3]->SetMarkerStyle(26);
  hEffOffline[2]->SetMarkerStyle(32);
  hEffOffline[1]->SetMarkerStyle(27);
  hEffOffline[0]->SetMarkerStyle(28);

  hEffOnline[5]->SetMarkerStyle(20);
  hEffOnline[4]->SetMarkerStyle(21);
  hEffOnline[3]->SetMarkerStyle(22);
  hEffOnline[2]->SetMarkerStyle(23);
  hEffOnline[1]->SetMarkerStyle(33);
  hEffOnline[0]->SetMarkerStyle(34);

  for (Int_t j = 0; j < 6; ++j) {
    hEffOffline[j]->Divide(hFracOffline[j], hAll, 1, 1, "B");
    hEffOnline[j]->Divide(hFracOnline[j], hAll, 1, 1, "B");
    hEffOffline[j]->GetYaxis()->SetRangeUser(0., 1.);
    hEffOffline[j]->SetLineWidth(2);
    hEffOffline[j]->SetLineColor(kBlue);
    hEffOffline[j]->SetMarkerSize(1.5);
    hEffOffline[j]->SetMarkerColor(kBlue);
    hEffOnline[j]->SetLineWidth(2);
    hEffOnline[j]->SetLineColor(kBlue);
    hEffOnline[j]->SetMarkerSize(1.5);
    hEffOnline[j]->SetMarkerColor(kBlue);
    hEffOnline[j]->GetYaxis()->SetRangeUser(0., 1.);
    if (!writeToFile) {
      if (!hasDrawnHist) {
        hEffOffline[j]->Draw("ep");
        hasDrawnHist = kTRUE;
      } else {
        hEffOffline[j]->Draw("ep same");
      }
      hEffOnline[j]->Draw("ep same");
    } else {
      hEffOffline[j]->Write();
      hEffOnline[j]->Write();
      delete hEffOffline[j];
      delete hEffOnline[j];
    }
  }
  if (writeToFile) {
    fOut->Close();
    delete fOut;
    fOut = 0x0;
  }
}

void PlotOnlineTrackletEfficiency(Int_t nEntries = -1)
{

  // plot online tracklet efficiency independent of HLT tracking
  // efficiency: fraction of online tracklets with correct track label
  //             if track produced hits within the TRD layer

  fOut = new TFile("results.root", "update");
  if (nEntries < 0) {
    nEntries = tree->GetEntriesFast();
  }
  TH1F* hAllPos = new TH1F("allPos", "tmp histogram;pT (GeV);counts", 10, 0, 5);
  TH1F* hFracPos = new TH1F("fracPos", "tmp histogram;pT (GeV);counts", 10, 0, 5);
  TH1F* hEffPos = new TH1F("effPos", "online tracklet efficiency;p_{T} (GeV/#it{c});efficiency", 10, 0, 5);
  hAllPos->Sumw2();
  hFracPos->Sumw2();
  MakeLogScale(hAllPos, 0.2, 12);
  MakeLogScale(hFracPos, 0.2, 12);
  MakeLogScale(hEffPos, 0.2, 12);
  TH1F* hAllNeg = new TH1F("allNeg", "tmp histogram;pT (GeV);counts", 10, 0, 5);
  TH1F* hFracNeg = new TH1F("fracNeg", "tmp histogram;pT (GeV);counts", 10, 0, 5);
  TH1F* hEffNeg = new TH1F("effNeg", "online tracklet efficiency;p_{T} (GeV/#it{c});efficiency", 10, 0, 5);
  hAllNeg->Sumw2();
  hFracNeg->Sumw2();
  MakeLogScale(hAllNeg, 0.2, 12);
  MakeLogScale(hFracNeg, 0.2, 12);
  MakeLogScale(hEffNeg, 0.2, 12);
  for (Int_t iEntry = 0; iEntry < nEntries; iEntry++) {
    tree->GetEntry(iEntry);
    if (trackID < 0 || TMath::Abs(trackPID) != 211) {
      continue;
    }
    for (Int_t iLy = 1; iLy < 6; iLy++) {
      if ((*findable)[iLy] < 1 || (*findableMC)[iLy] < 1) {
        continue;
      }
      Double_t trkPt = (*trackPt)[iLy];
      //(*trackQPt)[iLy] > 0 ? hAllPos->Fill(trkPt) : hAllNeg->Fill(trkPt);
      trackPID == 211 ? hAllPos->Fill(trkPt) : hAllNeg->Fill(trkPt);
      if ((*nMatchingTracklets)[iLy] > 0) {
        //(*trackQPt)[iLy] > 0 ? hFracPos->Fill(trkPt) : hFracNeg->Fill(trkPt);
        trackPID == 211 ? hFracPos->Fill(trkPt) : hFracNeg->Fill(trkPt);
      }
    }
  }
  gStyle->SetOptStat(0);
  gStyle->SetErrorX(0);
  hEffPos->Divide(hFracPos, hAllPos, 1, 1, "B");
  hEffNeg->Divide(hFracNeg, hAllNeg, 1, 1, "B");
  //TCanvas* c1 = new TCanvas("c1", "c1");
  //c1->SetGridy();
  //c1->SetGridx();
  //c1->SetLogx();
  hEffPos->SetLineWidth(2);
  hEffPos->SetLineColor(kRed);
  hEffPos->SetMarkerStyle(22);
  hEffPos->SetMarkerColor(kRed);
  hEffPos->GetYaxis()->SetRangeUser(0.2, 1.);
  hEffNeg->SetLineWidth(2);
  hEffNeg->SetLineColor(kBlack);
  hEffNeg->SetMarkerStyle(22);
  hEffNeg->SetMarkerColor(kBlack);
  //hEff->Draw("ep");
  hEffPos->Write();
  delete hEffPos;
  hEffNeg->Write();
  delete hEffNeg;
  fOut->Close();
}

void MultipleHypothesisCheck(Float_t minPt = 0.7, Int_t nEntries = -1)
{
  if (!tree) {
    printf("Initialize first!\n");
    return;
  }
  if (nEntries < 0) {
    nEntries = tree->GetEntriesFast();
  }
  TH1F* hChi2 = new TH1F("chi2", "chi2 for correct update;chi2;counts", 100, 0, 100);
  Int_t nPossibleSaves = 0;
  Int_t nTracksTotal = 0;
  for (Int_t iEntry = 0; iEntry < nEntries; iEntry++) {
    tree->GetEntry(iEntry);
    if (trackID < 0) {
      continue;
    }
    if (trackPtTPC < minPt) {
      continue;
    }
    if (nTracklets < 0) {
      continue;
    }
    ++nTracksTotal;
    for (Int_t iLy = 0; iLy < 5; ++iLy) {
      if ((*update)[iLy] > 6) {
        // fake tracklet attached in this layer
        if ((*update)[iLy + 1] > 6 || (*update)[iLy + 1] < 1) {
          // no tracklet or fake tracklet in the next
          if ((*nMatchingTracklets)[iLy + 1] > 0) {
            // but match would have been available
            ++nPossibleSaves;
            hChi2->Fill((*chi2Real)[iLy + 1]);
            //hChi2->Fill((*chi2Update)[iLy+1]-(*chi2Real)[iLy+1]);
            break;
          }
        }
      }
    }
  }
  printf("Out of %i tracks, %i additional updates could have possibly been done\n", nTracksTotal, nPossibleSaves);
  TCanvas* c1 = new TCanvas("c1", "c1");
  hChi2->SetLineWidth(2);
  hChi2->Draw();
}

void LoopFakes(Int_t nEntries = -1)
{
  if (nEntries < 0) {
    nEntries = tree->GetEntriesFast();
  }
  TH1F* hFakes = new TH1F("fakes", "all fake updates;pT (GeV);counts", 20, 0, 5);
  TH1F* hFakesAv = new TH1F("fakesAv", "all fake updates for which match is available;pT (GeV);counts", 20, 0, 5);
  TH1F* hFakesAvWindow = new TH1F("fakesAvWindow", "all fake updates for which match is available in window;pT (GeV);counts", 20, 0, 5);
  TH1F* hChi2 = new TH1F("chi2", "preferred fake over available match;#chi^{2}_{real}-#chi^{2}_{fake};counts", 100, 0, 10);
  TH1F* hChi2Real = new TH1F("chi2Real", "Matching tracklet in road, but none chosen;#chi^{2}_{real};counts", 100, 0, 10);
  TH1F* hChi2Fakes = new TH1F("chi2Fakes", "All fake updates;#chi^{2}_{fake};counts", 100, 0, 10);
  TH1F* hChi2AvFakes = new TH1F("chi2AvFakes", "Fake over match was chosen;#chi^{2}_{fake};counts", 100, 0, 10);
  MakeLogScale(hChi2, 1e-5, 1e3);
  MakeLogScale(hChi2Real, 1, 1e3);
  MakeLogScale(hChi2Fakes, 1e-2, 15);
  MakeLogScale(hChi2AvFakes, 1e-2, 15);
  hFakes->Sumw2();
  hFakesAv->Sumw2();
  hFakesAvWindow->Sumw2();
  TH1F* hFakesAvEff = new TH1F("fakesAvEff", "fake attachement w/ available match;p_{T} (GeV);fraction", 20, 0, 5);
  TH1F* hFakesAvWindowEff = new TH1F("fakesAvWindowEff", "available matching tracklet in search road for fake attachements;p_{T} (GeV);fraction", 20, 0, 5);
  MakeLogScale(hFakesAv, 0.1, 5);
  MakeLogScale(hFakesAvWindow, 0.1, 5);
  MakeLogScale(hFakesAvWindowEff, 0.1, 5);

  MakeLogScale(hFakesAvEff, 0.1, 5);
  MakeLogScale(hFakes, 0.1, 5);

  TH1F* hAllMissing = new TH1F("allMiss", "tmp histogram;pT (GeV);counts", 20, 0, 5);
  TH1F* hFracAvailable = new TH1F("fracAvail", "tmp histogram;pT (GeV);counts", 20, 0, 5);
  TH1F* hEff = new TH1F("eff", "Matching tracklet exists, but none was chosen;p_{T} (GeV);fraction", 20, 0, 5);
  hAllMissing->Sumw2();
  hFracAvailable->Sumw2();
  MakeLogScale(hAllMissing, 0.1, 5);
  MakeLogScale(hFracAvailable, 0.1, 5);
  MakeLogScale(hEff, 0.1, 5);

  for (Int_t iEntry = 0; iEntry < nEntries; iEntry++) {
    tree->GetEntry(iEntry);
    if (trackID < 0) {
      continue;
    }
    if (trackPtTPC < 1.) {
      continue;
    }
    for (Int_t iLy = 1; iLy < 6; iLy++) {
      Double_t trkPt = (*trackPt)[iLy];

      if ((*findable)[iLy] < 1 || (*findableMC)[iLy] < 1) {
        continue;
      }
      if ((*update)[iLy] < 1) {
        // no tracklet was found
        hAllMissing->Fill(trkPt);
        if ((*nMatchingTracklets)[iLy] > 0) {
          // matching tracklet would have been available
          hFracAvailable->Fill(trkPt);
          int trackSector = (*trackSec)[iLy];
          int trackletSector = (*trackletSecReal)[iLy];
          if ((*trackSec)[iLy] == (*trackletSecReal)[iLy] && TMath::Abs((*trackYreal)[iLy] - (*trackletYreal)[iLy]) < (*roadY)[iLy] && TMath::Abs((*trackZreal)[iLy] - (*trackletZreal)[iLy]) < (*roadZ)[iLy]) {
            // matching tracklet available and in search road
            hChi2Real->Fill((*chi2Real)[iLy]);
          } else {
            if (trackSector != trackletSector) {
              printf("trackSec(%i) != trackletSec(%i)\n", trackSector, trackletSector);
              printf("or tracklet outside road y(%f), z(%f)\n", (*roadY)[iLy], (*roadZ)[iLy]);
              printf("trackY (%f), trackletY(%f)\n", (*trackYreal)[iLy], (*trackletYreal)[iLy]);
              printf("trackZ (%f), trackletZ(%f)\n", (*trackZreal)[iLy], (*trackletZreal)[iLy]);
            }
          }
        }
      }
      if ((*update)[iLy] > 8) {
        // fake update was attached instead
        hFakes->Fill(trkPt);
        hChi2Fakes->Fill((*chi2Update)[iLy]);
        if ((*nMatchingTracklets)[iLy] > 0) {
          // matching tracklet would have been available
          hFakesAv->Fill(trkPt);
          hChi2AvFakes->Fill((*chi2Update)[iLy]);
          if ((*trackSec)[iLy] == (*trackletSecReal)[iLy] && TMath::Abs((*trackYreal)[iLy] - (*trackletYreal)[iLy]) < (*roadY)[iLy] && TMath::Abs((*trackZreal)[iLy] - (*trackletZreal)[iLy]) < (*roadZ)[iLy]) {
            // matching tracklet available and in search road
            hFakesAvWindow->Fill(trkPt);
            hChi2->Fill((*chi2Real)[iLy] - (*chi2Update)[iLy]);
          }
          // else {
          //  if ((*trackSec)[iLy] != (*trackletSecReal)[iLy]) {
          //    printf("---\n");
          //    printf("Matching tracklet in different sector (%.0f) than track (%.0f)\n", (*trackletSecReal)[iLy], (*trackSec)[iLy]);
          //    printf("event(%i), trackID(%i), layer(%i), pT(%f), trackletY(%f), trackY(%f), trackletYreal(%f), trackYreal(%f), trackSigmaY(%f)\n",
          //            event, trackID, iLy, trkPt, (*trackletY)[iLy], (*trackY)[iLy], (*trackletYreal)[iLy], (*trackYreal)[iLy], TMath::Sqrt((*trackYerr)[iLy]));
          //  }
          //}
        }
      }
      //
    }
  }
  // gStyle->SetOptStat(0);

  hFakesAvEff->Divide(hFakesAv, hFakes, 1, 1, "B");
  hFakesAvWindowEff->Divide(hFakesAvWindow, hFakes, 1, 1, "B");

  TCanvas* cAv = new TCanvas("cAv", "cAv");
  cAv->SetGridx();
  cAv->SetGridy();
  hFakesAvEff->GetYaxis()->SetRangeUser(0., 1.);
  hFakesAvEff->SetLineWidth(2);
  hFakesAvEff->Draw();

  TCanvas* cAvWindow = new TCanvas("cAvWindow", "cAvWindow");
  hFakesAvWindowEff->GetYaxis()->SetRangeUser(0., 1.);
  hFakesAvWindowEff->SetLineWidth(2);
  cAvWindow->SetGridx();
  cAvWindow->SetGridy();
  hFakesAvWindowEff->Draw();

  TCanvas* cChi2 = new TCanvas("cChi2", "cChi2");
  hChi2->SetLineWidth(2);
  hChi2->Draw();
  cChi2->SetLogx();

  TCanvas* cChi2Fakes = new TCanvas("cChi2Fakes", "cChi2Fakes");
  cChi2Fakes->SetLogx();
  hChi2Fakes->SetLineWidth(2);
  hChi2Fakes->Draw();

  TCanvas* cChi2AvFakes = new TCanvas("cChi2AvFakes", "cChi2AvFakes");
  cChi2AvFakes->SetLogx();
  hChi2AvFakes->Draw();

  TCanvas* cMissing = new TCanvas("cMissing", "cMissing");
  hEff->Divide(hFracAvailable, hAllMissing, 1, 1, "B");
  hEff->GetYaxis()->SetRangeUser(0., 1.);
  hEff->SetLineWidth(2);
  hEff->Draw();
  cMissing->SetGridx();
  cMissing->SetGridy();

  TCanvas* cTmpChi2 = new TCanvas("cTmpChi2", "cTmpChi2");
  hChi2Real->SetLineWidth(2);
  hChi2Real->Draw();
  cTmpChi2->SetLogx();
}

void LoopUpdates(Int_t nEntries = -1)
{
  TH1F* hUpdateAv = new TH1F("updateAv", ";track p_{T} (GeV);counts", 20, 0, 5);
  TH1F* hUpdateAvInWindow = new TH1F("updateAvInWindow", ";track p_{T} (GeV);counts", 20, 0, 5);
  TH1F* hUpdateAvInWindowChi2 = new TH1F("updateAvInWindowChi2", ";track p_{T} (GeV);counts", 20, 0, 5);
  TH1F* hUpdateAvAll = new TH1F("updateAvAll", ";track p_{T} (GeV);counts", 20, 0, 5);
  TH1F* hUpdateAvReal = new TH1F("updateAvReal", ";track p_{T} (GeV);counts", 20, 0, 5);
  TH1F* hUpdateAvFake = new TH1F("updateAvFake", ";track p_{T} (GeV);counts", 20, 0, 5);
  TH1F* hUpdateNoAv = new TH1F("updateNoAv", ";track p_{T} (GeV);counts", 20, 0, 5);
  TH1F* hUpdateNoAvAll = new TH1F("updateNoAvAll", ";track p_{T} (GeV);counts", 20, 0, 5);
  TH1F* hUpdateNoAvReal = new TH1F("updateNoAvReal", ";track p_{T} (GeV);counts", 20, 0, 5);
  TH1F* hUpdateNoAvFake = new TH1F("updateNoAvFake", ";track p_{T} (GeV);counts", 20, 0, 5);
  TH1F* hUpdateAll = new TH1F("updateAll", ";track p_{T} (GeV);counts", 20, 0, 5);
  TH1F* hUpdateReal = new TH1F("updateReal", ";track p_{T} (GeV);counts", 20, 0, 5);
  TH1F* hUpdateFake = new TH1F("updateFake", ";track p_{T} (GeV);counts", 20, 0, 5);
  TH1F* hUpdateFakeButAv = new TH1F("updateFakeButAv", ";track p_{T} (GeV);counts", 20, 0, 5);

  hUpdateAv->Sumw2();
  hUpdateAvInWindow->Sumw2();
  hUpdateAvInWindowChi2->Sumw2();
  hUpdateAvAll->Sumw2();
  hUpdateAvReal->Sumw2();
  hUpdateAvFake->Sumw2();
  hUpdateNoAv->Sumw2();
  hUpdateNoAvAll->Sumw2();
  hUpdateNoAvReal->Sumw2();
  hUpdateNoAvFake->Sumw2();
  hUpdateAll->Sumw2();
  hUpdateReal->Sumw2();
  hUpdateFake->Sumw2();
  hUpdateFakeButAv->Sumw2();

  TH1F* hUpdateAvEff = new TH1F("updateAvEff", "fraction of updates if match available;track p_{T} (GeV);fraction", 20, 0, 5);
  TH1F* hUpdateAvInWindowEff = new TH1F("updateAvInWindowEff", "fraction of matches in window if match available but no update;track p_{T} (GeV);fraction", 20, 0, 5);
  TH1F* hUpdateAvInWindowChi2Eff = new TH1F("updateAvInWindowChi2Eff", "fraction of matches in window meating chi2 cut if match available but no update;track p_{T} (GeV);fraction", 20, 0, 5);
  TH1F* hUpdateAvRealEff = new TH1F("updateAvRealEff", "fraction of matches in case match available and update performed;track p_{T} (GeV);fraction", 20, 0, 5);
  TH1F* hUpdateAvFakeEff = new TH1F("updateAvFakeEff", "fraction of fakes in case match available and update performed;track p_{T} (GeV);fraction", 20, 0, 5);
  TH1F* hUpdateRealEff = new TH1F("updateRealEff", "fraction of matches (all update);track p_{T} (GeV);fraction", 20, 0, 5);
  TH1F* hUpdateFakeEff = new TH1F("updateFakeEff", "fraction of fakes (all update);track p_{T} (GeV);fraction", 20, 0, 5);
  TH1F* hFakesButAv = new TH1F("fakesButAv", "fraction of fakes for which match would be there;track p_{T} (GeV);fraction", 20, 0, 5);

  if (nEntries < 0) {
    nEntries = tree->GetEntriesFast();
  }
  for (Int_t iEntry = 0; iEntry < nEntries; iEntry++) {
    tree->GetEntry(iEntry);
    if (trackID < 0) {
      continue;
    }
    for (Int_t iLy = 1; iLy < 6; iLy++) {
      if ((*findable)[iLy] < 1 || (*findableMC)[iLy] < 1) {
        continue;
      }

      // fraction of fakes for all updates
      if ((*update)[iLy] > 0) {
        hUpdateAll->Fill((*trackPt)[iLy]);
        if ((*update)[iLy] < 8) {
          hUpdateReal->Fill((*trackPt)[iLy]);
        } else {
          hUpdateFake->Fill((*trackPt)[iLy]);
          if ((*nMatchingTracklets)[iLy] > 0) {
            hUpdateFakeButAv->Fill((*trackPt)[iLy]);
          }
        }
      }

      // efficiency for available tracklets
      if ((*nMatchingTracklets)[iLy] > 0) {
        hUpdateAv->Fill((*trackPt)[iLy]);
        if ((*update)[iLy] > 0) {
          hUpdateAvAll->Fill((*trackPt)[iLy]);
          if ((*update)[iLy] < 8) {
            hUpdateAvReal->Fill((*trackPt)[iLy]);
          } else {
            hUpdateAvFake->Fill((*trackPt)[iLy]);
          }
        } else {
          // no update although match was available
          if (TMath::Abs((*trackYreal)[iLy] - (*trackletYreal)[iLy]) < (*roadY)[iLy] && TMath::Abs((*trackZreal)[iLy] - (*trackletZreal)[iLy]) < (*roadZ)[iLy]) {
            hUpdateAvInWindow->Fill((*trackPt)[iLy]);
            if ((*chi2Real)[iLy] < 10) {
              hUpdateAvInWindowChi2->Fill((*trackPt)[iLy]);
            }
          }
        }
      }

      // efficiency for not available tracklets
      else if ((*nMatchingTracklets)[iLy] < 1) {
        hUpdateNoAv->Fill((*trackPt)[iLy]);
        if ((*update)[iLy] > 0) {
          hUpdateNoAvAll->Fill((*trackPt)[iLy]);
          if ((*update)[iLy] < 8) {
            hUpdateNoAvReal->Fill((*trackPt)[iLy]);
          } else {
            hUpdateNoAvFake->Fill((*trackPt)[iLy]);
          }
        }
      }
    }
  }
  gStyle->SetOptStat(0);

  hUpdateAvEff->Divide(hUpdateAvAll, hUpdateAv, 1, 1, "B");
  hUpdateAvRealEff->Divide(hUpdateAvReal, hUpdateAvAll, 1, 1, "B");
  hUpdateAvFakeEff->Divide(hUpdateAvFake, hUpdateAvAll, 1, 1, "B");
  hUpdateRealEff->Divide(hUpdateReal, hUpdateAll, 1, 1, "B");
  hUpdateFakeEff->Divide(hUpdateFake, hUpdateAll, 1, 1, "B");
  hUpdateAvInWindowEff->Divide(hUpdateAvInWindow, hUpdateAv, 1, 1, "B");
  hUpdateAvInWindowChi2Eff->Divide(hUpdateAvInWindowChi2, hUpdateAvInWindow, 1, 1, "B");
  hFakesButAv->Divide(hUpdateFakeButAv, hUpdateFake, 1, 1, "B");

  TCanvas* cA = new TCanvas("cA", "cA");
  hUpdateAvEff->Draw();
  TCanvas* cB = new TCanvas("cB", "cB");
  // hUpdateAvRealEff->Draw();
  hUpdateAvInWindowEff->Draw();
  TCanvas* cC = new TCanvas("cC", "cC");
  hUpdateAvFakeEff->Draw();
  TCanvas* cD = new TCanvas("cD", "cD");
  // hUpdateRealEff->Draw();
  hUpdateAvInWindowChi2Eff->Draw();
  TCanvas* cE = new TCanvas("cE", "cE");
  hUpdateFakeEff->Draw();
  TCanvas* cF = new TCanvas("cF", "cF");
  hFakesButAv->Draw();

  /*
        TCanvas *c1 = new TCanvas("c1", "c1");
        hUpdateAv->Draw();
        TCanvas *c2 = new TCanvas("c2", "c2");
        hUpdateAvAll->Draw();
        TCanvas *c3 = new TCanvas("c3", "c3");
        hUpdateAvReal->Draw();
        TCanvas *c4 = new TCanvas("c4", "c4");
        hUpdateAvFake->Draw();
        TCanvas *c5 = new TCanvas("c5", "c5");
        hUpdateNoAv->Draw();
        TCanvas *c6 = new TCanvas("c6", "c6");
        hUpdateNoAvAll->Draw();
        TCanvas *c7 = new TCanvas("c7", "c7");
        hUpdateNoAvReal->Draw();
        TCanvas *c8 = new TCanvas("c8", "c8");
        hUpdateNoAvFake->Draw();
        TCanvas *c9 = new TCanvas("c9", "c9");
        hUpdateAll->Draw();
        TCanvas *c10 = new TCanvas("c10", "c10");
        hUpdateReal->Draw();
        TCanvas *c11 = new TCanvas("c11", "c11");
        hUpdateFake->Draw();
   */
}

void LoopCheckHitsMC(Int_t nEntries = -1)
{
  // only makes sense if related tracklets were not
  // counted as matches
  // Loops over all tracks and looks for layers where a matching tracklet was available for the track, but
  // no hits are availabel at the entrance / exit of the TRD chamber
  if (nEntries < 0) {
    nEntries = tree->GetEntriesFast();
  }
  Double_t nTracks = 0;
  Double_t nTracksErr = 0;
  for (Int_t iEntry = 0; iEntry < nEntries; iEntry++) {
    tree->GetEntry(iEntry);
    if (trackID < 0) {
      continue;
    }
    nTracks += 1;
    for (Int_t iLy = 1; iLy < 6; iLy++) {
      if ((*findable)[iLy] > 0 && (*nMatchingTracklets)[iLy] > 0 && (*findableMC)[iLy] < 1) {
        nTracksErr += 1;
        printf("Error found for entry %i\n", iEntry);
        printf("--- update ---\n");
        (*update).Print();
        printf("--- findable ---\n");
        (*findable).Print();
        printf("--- findableMC ---\n");
        (*findableMC).Print();
        printf("--- nMatchingTracklets ---\n");
        (*nMatchingTracklets).Print();
        printf("--- trackY ---\n");
        (*trackY).Print();
        printf("--- trackZ ---\n");
        (*trackZ).Print();
        printf("--- trackletY ---\n");
        (*trackletY).Print();
        printf("--- trackletZ ---\n");
        (*trackletZ).Print();
        printf("track vertex: x=%f, y=%f, z=%f, R=%f\n", XvMC, YvMC, ZvMC, TMath::Sqrt(XvMC * XvMC + YvMC * YvMC));
        continue;
      }
    }
  }
  printf("Out of %.0f tracks, %.0f had errors\n", nTracks, nTracksErr);
}

void chi2Distribution(Int_t nEntries = -1)
{
  // compare chi2 for matching tracklets with chi2 for fake tracklets
  // scaled w.r.t. total number of updates
  TH1F* hChi2Match = new TH1F("chi2Match", ";#chi^{2};normalized counts", 50, 0, 15);
  TH1F* hChi2Fake = new TH1F("chi2Fake", ";#chi^{2};normalized counts", 50, 0, 15);

  MakeLogScale(hChi2Match, 0.05, 25);
  MakeLogScale(hChi2Fake, 0.05, 25);

  Double_t nUpdates = 0;

  if (nEntries < 0) {
    nEntries = tree->GetEntries();
  }
  for (Int_t i = 0; i < nEntries; i++) {
    // loop over all tracks
    tree->GetEntry(i);
    if (trackID < 0) {
      continue;
    }
    for (Int_t iLy = 0; iLy < 6; ++iLy) {
      if ((*update)[iLy] < 1) {
        continue;
      }
      nUpdates += 1;
      if ((*update)[iLy] < 8) {
        hChi2Match->Fill((*chi2Update)[iLy]);
      } else if ((*update)[iLy] > 8) {
        hChi2Fake->Fill((*chi2Update)[iLy]);
      }
    }
  }
  TCanvas* c1 = new TCanvas("c1", "c1");
  c1->SetLogx();
  hChi2Match->SetLineWidth(2);
  hChi2Fake->SetLineWidth(2);
  hChi2Fake->SetLineColor(kRed);
  hChi2Match->Scale(1. / nUpdates);
  hChi2Fake->Scale(1. / nUpdates);
  TLegend* leg = new TLegend(0.15, 0.6, 0.35, 0.8);
  leg->AddEntry(hChi2Match, "matches", "l");
  leg->AddEntry(hChi2Fake, "fakes", "l");
  hChi2Match->Draw();
  hChi2Fake->Draw("same");
  leg->Draw();
}

void OnlineTrackletEfficiency(Int_t nEntries = -1)
{

  // only possible if full track information was stored

  const Int_t nBins = 20;
  const Double_t binStartPhi = -0.8;
  const Double_t binEndPhi = 0.8;
  const Double_t binStartPt = 0.1;
  const Double_t binEndPt = 5.;

  const Int_t nBins2D = 50;
  TH2F* hIsFindableQptY = new TH2F("isFindableQptY", ";#it{y} (cm);#it{q}/#it{p}_{T} (#it{c}/GeV)", nBins2D, -60, 60, nBins2D, -2, 2);
  TH2F* hIsFoundQptY = new TH2F("isFoundQptY", ";#it{y} (cm);#it{q}/#it{p}_{T} (#it{c}/GeV)", nBins2D, -60, 60, nBins2D, -2, 2);
  TH2F* hEfficiencyQptY = new TH2F("isEfficiencyQptY", ";#it{y} (cm);#it{q}/#it{p}_{T} (#it{c}/GeV)", nBins2D, -60, 60, nBins2D, -2, 2);

  TH1F* hIsFindablePhi = new TH1F("isFindablePhi", ";track #varphi;counts", nBins, binStartPhi, binEndPhi);
  hIsFindablePhi->Sumw2();
  TH1F* hIsFoundPhi = new TH1F("isFoundPhi", ";track #varphi;counts", nBins, binStartPhi, binEndPhi);
  hIsFoundPhi->Sumw2();
  TH1F* hEfficiencyPhi = new TH1F("efficiencyPhi", "TRD online tracklet efficiency;track #varphi;eff.", nBins, binStartPhi, binEndPhi);

  TH1F* hIsFindablePt = new TH1F("isFindablePt", ";track p_{T} (GeV/#it{c});counts", nBins, binStartPt, binEndPt);
  hIsFindablePt->Sumw2();
  TH1F* hIsFoundPt = new TH1F("isFoundPt", ";track p_{T} (GeV/#it{c});counts", nBins, binStartPt, binEndPt);
  hIsFoundPt->Sumw2();
  TH1F* hEfficiencyPt = new TH1F("efficiencyPt", "TRD online tracklet efficiency;track p_{T} (GeV/#it{c});eff.", nBins, binStartPt, binEndPt);

  MakeLogScale(hIsFindablePt, 0.1, 5);
  MakeLogScale(hIsFoundPt, 0.1, 5);
  MakeLogScale(hEfficiencyPt, 0.1, 5);

  if (nEntries < 0) {
    nEntries = tree->GetEntries();
  }
  for (Int_t i = 0; i < nEntries; i++) {
    // loop over all tracks
    tree->GetEntry(i);
    if (trackID < 0) {
      continue;
    }
    if (TMath::Sqrt(XvMC * XvMC + YvMC * YvMC) > 1) {
      // only consider tracks originating from close to the vertex
      continue;
    }
    for (Int_t iLy = 0; iLy < 6; ++iLy) {
      if ((*findable)[iLy] < 1 || (*findableMC)[iLy] < 1) {
        continue;
      }
      // track is in acceptance and MC hits are available
      hIsFindablePhi->Fill((*trackPhi)[iLy]);
      hIsFindablePt->Fill((*trackPt)[iLy]);
      Double_t trkQpt = (*trackQPt)[iLy];
      Double_t trkY = (*trackY)[iLy];
      hIsFindableQptY->Fill(trkY, trkQpt);
      if ((*nMatchingTracklets)[iLy] > 0) {
        hIsFoundPhi->Fill((*trackPhi)[iLy]);
        hIsFoundPt->Fill((*trackPt)[iLy]);
        hIsFoundQptY->Fill(trkY, trkQpt);
      }
    }
  }
  gStyle->SetOptStat(0);

  hEfficiencyPhi->Divide(hIsFoundPhi, hIsFindablePhi, 1, 1, "B");
  hEfficiencyPhi->SetLineWidth(2);
  hEfficiencyPhi->GetYaxis()->SetRangeUser(0., 1.);
  hEfficiencyPt->Divide(hIsFoundPt, hIsFindablePt, 1, 1, "B");
  hEfficiencyPt->SetLineWidth(2);
  hEfficiencyPt->GetYaxis()->SetRangeUser(0., 1.);

  for (Int_t iBinX = 0; iBinX < nBins2D; iBinX++) {
    for (Int_t iBinY = 0; iBinY < nBins2D; iBinY++) {
      Int_t iBin = hIsFindableQptY->GetBin(iBinX, iBinY);
      if (hIsFindableQptY->GetBinContent(iBin) <= 0) {
        hEfficiencyQptY->SetBinContent(iBin, 0);
        continue;
      }
      hEfficiencyQptY->SetBinContent(iBin, hIsFoundQptY->GetBinContent(iBin) / hIsFindableQptY->GetBinContent(iBin));
    }
  }

  TCanvas* cEffPhi = new TCanvas("effPhi", "effPhi");
  gPad->SetGridx();
  gPad->SetGridy();
  hEfficiencyPhi->Draw("ep");

  TCanvas* cEffPt = new TCanvas("effPt", "effPt");
  gPad->SetGridx();
  gPad->SetGridy();
  hEfficiencyPt->Draw("ep");
  cEffPt->SetLogx();

  TCanvas* cFindablePhi = new TCanvas("findablePhi", "findablePhi");
  hIsFindablePhi->SetLineWidth(2);
  hIsFindablePhi->SetMarkerStyle(22);
  hIsFindablePhi->SetMarkerSize(.95);
  hIsFindablePhi->SetMarkerColor(kRed);
  hIsFoundPhi->SetLineWidth(2);
  hIsFoundPhi->SetMarkerStyle(23);
  hIsFoundPhi->SetMarkerSize(.95);
  hIsFoundPhi->SetMarkerColor(kBlue);
  hIsFindablePhi->SetLineColor(kRed);
  hIsFindablePhi->Draw("ep");
  hIsFoundPhi->Draw("ep same");
  TLegend* legFindable = new TLegend(0.7, 0.7, 0.9, 0.9);
  legFindable->AddEntry(hIsFindablePhi, "track ref exists", "lp");
  legFindable->AddEntry(hIsFoundPhi, "online trklt exists", "lp");
  legFindable->Draw();

  TCanvas* cFindablePt = new TCanvas("findablePt", "findablePt");
  hIsFindablePt->SetLineWidth(2);
  hIsFindablePt->SetMarkerStyle(22);
  hIsFindablePt->SetMarkerSize(.95);
  hIsFindablePt->SetMarkerColor(kRed);
  hIsFoundPt->SetLineWidth(2);
  hIsFoundPt->SetMarkerStyle(23);
  hIsFoundPt->SetMarkerSize(.95);
  hIsFoundPt->SetMarkerColor(kBlue);
  hIsFindablePt->SetLineColor(kRed);
  hIsFindablePt->Draw("ep");
  hIsFoundPt->Draw("ep same");
  legFindable->Draw();

  TCanvas* cEfficiencyQptY = new TCanvas("efficiencyQptY", "efficiencyQptY");
  hEfficiencyQptY->GetXaxis()->SetRangeUser(-45.0, 45.);
  hEfficiencyQptY->GetZaxis()->SetRangeUser(0.0, 1);
  hEfficiencyQptY->Draw("colz");
}

Bool_t InitAnalysis(const char* filename, Bool_t isMC)
{
  Reset();
  if (!InitTree(filename)) {
    return kFALSE;
  }
  //
  InitBranches();
  //
  InitCalib();
  //
  SetAlias(tree, isMC);
  //
  return kTRUE;
}

void Reset()
{
  if (f) {
    delete f;
    f = 0x0;
  }
  tree = 0x0;
}

void checkDbgOutput()
{
  printf("Basic usage:\n");
  printf("InitAnalysis(\"TRDhlt.root\", 1);\n");
}

/*
   currently not needed


   //TCanvas *cTest = new TCanvas("cTest", "cTest");
   //tree->Draw("GetDeltaYmatch(layer):trkPhi>>his(10, -.3, .3, 100, -5, 5)", "isFake&&matchAvail&&LoadBranches(Entry$)", "colz");
   //tree->Draw("GetDeltaY(layer)-GetDeltaYmatch(layer)>>his(100, -5., 5.)", "isFake&&matchAvail&&LoadBranches(Entry$)", "colz");
   //tree->Draw("roadY", "isFake&&matchAvail&&LoadBranches(Entry$)", "colz");
   //tree->Draw("abs(GetDeltaYmatch(layer))-abs(GetDeltaY(layer))>>hist(100, -2, 2)", "isFake&&LoadBranches(Entry$)&&matchAvail&&inYroad&&inZroad&&abs(GetDeltaZmatch(layer))<abs(GetDeltaZ(layer))");
   // obviously a bug -> if distance of real matching tracklet is smaller than chosen one, why was the real one not chosen?
   // what about tracklet errors? in z different for different pad widths, in y constant. should not make a difference, or?
   //tree->Draw("chi2Real.fElements-chi2Update.fElements", "isFake&&LoadBranches(Entry$)&&matchAvail&&inYroad&&inZroad&&abs(GetDeltaZmatch(layer))<abs(GetDeltaZ(layer))&&abs(GetDeltaYmatch(layer))<abs(GetDeltaY(layer))" );
   //tree->Draw("update.fElements", "isPresent&&LoadBranches(Entry$)" );

 */
