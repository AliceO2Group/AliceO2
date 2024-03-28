// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//-----------------------------------------------------------------------------
/// \file Aligner
/// Aligner class for the ALICE DiMuon spectrometer
///
/// MUON specific alignment class which interface to Millepede.
/// For each track ProcessTrack calculates the local and global derivatives
/// at each cluster and fill the corresponding local equations. Provide methods
/// for fixing or constraining detection elements for best results.
///
/// \author Javier Castillo Castellanos
//-----------------------------------------------------------------------------
#include <iostream>
#include <ctime>

#include "DataFormatsMCH/Cluster.h"
#include "DetectorsCommonDataFormats/AlignParam.h"
#include "ForwardAlign/MillePede2.h"
#include "ForwardAlign/MillePedeRecord.h"
#include "Framework/Logger.h"
#include "MCHAlign/Aligner.h"
#include "MathUtils/Cartesian.h"
#include "MCHTracking/Track.h"
#include "MCHTracking/TrackParam.h"
#include "MCHGeometryTransformer/Transformations.h"

#include <TClonesArray.h>
#include <TGeoManager.h>
#include <TGeoGlobalMagField.h>
#include <TGraphErrors.h>
#include <TMath.h>
#include <TMatrixD.h>
#include <TMatrixDSym.h>
#include <TObject.h>
#include <TString.h>

namespace o2
{
namespace mch
{

using namespace std;

//_____________________________________________________________________
// static variables
const int Aligner::fgNDetElemCh[Aligner::fgNCh] = {4, 4, 4, 4, 18, 18, 26, 26, 26, 26};
const int Aligner::fgSNDetElemCh[Aligner::fgNCh + 1] = {0, 4, 8, 12, 16, 34, 52, 78, 104, 130, 156};

// number of detector elements in each half-chamber
const int Aligner::fgNDetElemHalfCh[Aligner::fgNHalfCh] = {2, 2, 2, 2, 2, 2, 2, 2, 9, 9, 9, 9, 13, 13, 13, 13, 13, 13, 13, 13};

// list of detector elements for each half chamber
const int Aligner::fgDetElemHalfCh[Aligner::fgNHalfCh][Aligner::fgNDetHalfChMax] =
  {
    {100, 103, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {101, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},

    {200, 203, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {201, 202, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},

    {300, 303, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {301, 302, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},

    {400, 403, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {401, 402, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},

    {500, 501, 502, 503, 504, 514, 515, 516, 517, 0, 0, 0, 0},
    {505, 506, 507, 508, 509, 510, 511, 512, 513, 0, 0, 0, 0},

    {600, 601, 602, 603, 604, 614, 615, 616, 617, 0, 0, 0, 0},
    {605, 606, 607, 608, 609, 610, 611, 612, 613, 0, 0, 0, 0},

    {700, 701, 702, 703, 704, 705, 706, 720, 721, 722, 723, 724, 725},
    {707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719},

    {800, 801, 802, 803, 804, 805, 806, 820, 821, 822, 823, 824, 825},
    {807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819},

    {900, 901, 902, 903, 904, 905, 906, 920, 921, 922, 923, 924, 925},
    {907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919},

    {1000, 1001, 1002, 1003, 1004, 1005, 1006, 1020, 1021, 1022, 1023, 1024, 1025},
    {1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019}

};

//_____________________________________________________________________
/// self initialized array, used for adding constraints
class Array
{

 public:
  /// contructor
  Array()
  {
    for (int i = 0; i < Aligner::fNGlobal; ++i) {
      values[i] = 0;
    }
  }

  /// array
  double values[Aligner::fNGlobal];

 private:
  /// Not implemented
  Array(const Array&);

  /// Not implemented
  Array& operator=(const Array&);
};

//________________________________________________________________________
double Square(double x) { return x * x; }

//_____________________________________________________________________
Aligner::Aligner()
  : TObject(),
    fInitialized(false),
    fRunNumber(0),
    fBFieldOn(false),
    fRefitStraightTracks(false),
    fStartFac(65536),
    fResCutInitial(1000),
    fResCut(100),
    fMillepede(nullptr),
    fNStdDev(3),
    fDetElemNumber(0),
    fGlobalParameterStatus(std::vector<int>(fNGlobal)),
    fGlobalDerivatives(std::vector<double>(fNGlobal)),
    fLocalDerivatives(std::vector<double>(fNLocal)),
    fTrackRecord(),
    mNEntriesAutoSave(10000),
    mRecordWriter(new o2::fwdalign::MilleRecordWriter()),
    mWithConstraintsRecWriter(false),
    mConstraintsRecWriter(nullptr),
    mRecordReader(new o2::fwdalign::MilleRecordReader()),
    mWithConstraintsRecReader(false),
    mConstraintsRecReader(nullptr),
    fTransformCreator(),
    fDoEvaluation(false),
    fDisableRecordWriter(false),
    mRead(false),
    fTrkClRes(nullptr),
    fTFile(nullptr),
    fTTree(nullptr)
{
  /// constructor
  fSigma[0] = 1.5e-1;
  fSigma[1] = 1.0e-2;

  // default allowed variations
  fAllowVar[0] = 0.5;  // x
  fAllowVar[1] = 0.5;  // y
  fAllowVar[2] = 0.01; // phi_z
  fAllowVar[3] = 5;    // z

  // initialize millepede
  fMillepede = new o2::fwdalign::MillePede2();

  // initialize degrees of freedom
  // by default all parameters are free
  for (int iPar = 0; iPar < fNGlobal; ++iPar) {
    fGlobalParameterStatus[iPar] = kFreeParId;
  }

  // initialize local equations
  for (int i = 0; i < fNLocal; ++i) {
    fLocalDerivatives[i] = 0.0;
  }

  for (int i = 0; i < fNGlobal; ++i) {
    fGlobalDerivatives[i] = 0.0;
  }
}

//________________________________________________________________________
Aligner::~Aligner()
{
  delete mRecordWriter;
  delete mRecordReader;
  delete fMillepede;
}

//_____________________________________________________________________
void Aligner::init(TString DataRecFName, TString ConsRecFName)
{

  /// initialize
  /**
  initialize millipede
  must be called after necessary detectors have been fixed,
  but before constrains are added and before global parameters initial value are set
  */
  if (fInitialized) {
    LOG(fatal) << "Millepede already initialized";
  }

  if (!mRead) {
    if (!fDisableRecordWriter) {
      mRecordWriter->setCyclicAutoSave(mNEntriesAutoSave);
      mRecordWriter->setDataFileName(DataRecFName);
      fMillepede->SetRecordWriter(mRecordWriter);

      if (mWithConstraintsRecWriter) {
        mConstraintsRecWriter->setCyclicAutoSave(mNEntriesAutoSave);
        mConstraintsRecWriter->setDataFileName(ConsRecFName);
        fMillepede->SetConstraintsRecWriter(mConstraintsRecWriter);
      }
    } else {
      fMillepede->SetRecord(&fTrackRecord);
    }

  } else {

    TChain* ch = new TChain(mRecordReader->getDataTreeName());
    if (DataRecFName.EndsWith(".root")) {
      ch->AddFile(DataRecFName);
    }
    int nent = ch->GetEntries();

    if (nent < 1) {
      LOG(fatal) << "Obtained chain is empty, please check your record ROOT file.";
    }

    mRecordReader->connectToChain(ch);
    fMillepede->SetRecordReader(mRecordReader);

    if (mConstraintsRecReader) {

      TChain* ch_cons = new TChain(mConstraintsRecReader->getDataTreeName());
      if (ConsRecFName.EndsWith(".root")) {
        ch_cons->AddFile(ConsRecFName);
      }
      int nent_cons = ch_cons->GetEntries();

      if (nent_cons < 1) {
        LOG(fatal) << "Obtained chain is empty, please check your record ROOT file.";
      }

      mConstraintsRecReader->connectToChain(ch_cons);
      fMillepede->SetConstraintsRecReader(mConstraintsRecReader);
    }
  }

  // assign proper groupID to free parameters
  int nGlobal = 0;
  for (int iPar = 0; iPar < fNGlobal; ++iPar) {

    if (fGlobalParameterStatus[iPar] == kFixedParId) {
      // fixed parameters are left unchanged
      continue;

    } else if (fGlobalParameterStatus[iPar] == kFreeParId || fGlobalParameterStatus[iPar] == kGroupBaseId) {

      // free parameters or first element of group are assigned a new group id
      fGlobalParameterStatus[iPar] = nGlobal++;
      continue;

    } else if (fGlobalParameterStatus[iPar] < kGroupBaseId) {

      // get detector element id from status, get chamber parameter id
      const int iDeBase(kGroupBaseId - 1 - fGlobalParameterStatus[iPar]);
      const int iParBase = iPar % fgNParCh;

      // check
      if (iDeBase < 0 || iDeBase >= iPar / fgNParCh) {
        LOG(fatal) << "Group for parameter index " << iPar << " has wrong base detector element: " << iDeBase;
      }

      // assign identical group id to current
      fGlobalParameterStatus[iPar] = fGlobalParameterStatus[iDeBase * fgNParCh + iParBase];
      LOG(info) << "Parameter " << iPar << " grouped to detector " << iDeBase << " (" << GetParameterMaskString(1 << iParBase).Data() << ")";

    } else {
      LOG(fatal) << "Unrecognized parameter status for index " << iPar << ": " << fGlobalParameterStatus[iPar];
    }
  }

  LOG(info) << "Free Parameters: " << nGlobal << " out of " << fNGlobal;

  // initialize millepedes
  fMillepede->InitMille(fNGlobal, fNLocal, fNStdDev, fResCut, fResCutInitial, fGlobalParameterStatus);

  if (!mRead) {
    if (!fDisableRecordWriter) {
      mRecordWriter->init();
    } else {
      fMillepede->DisableRecordWriter();
    }
  }

  fInitialized = true;

  // some debug output
  for (int iPar = 0; iPar < fgNParCh; ++iPar) {
    LOG(info) << "fAllowVar[" << iPar << "]= " << fAllowVar[iPar];
  }

  // set allowed variations for all parameters
  for (int iDet = 0; iDet < fgNDetElem; ++iDet) {
    for (int iPar = 0; iPar < fgNParCh; ++iPar) {
      fMillepede->SetParSigma(iDet * fgNParCh + iPar, fAllowVar[iPar]);
    }
  }

  // Set iterations
  if (fStartFac > 1) {
    fMillepede->SetIterations(fStartFac);
  }
  // setup monitoring TFile
  if (fDoEvaluation) {
    // if (fDoEvaluation && fRefitStraightTracks) {
    string Path_file = Form("%s%s", "Residual", ".root");

    fTFile = new TFile(Path_file.c_str(), "RECREATE");
    fTTree = new TTree("TreeE", "Evaluation");

    const int kSplitlevel = 98;
    const int kBufsize = 32000;

    fTrkClRes = new o2::mch::LocalTrackClusterResidual();
    fTTree->Branch("fClDetElem", &(fTrkClRes->fClDetElem), "fClDetElem/I");
    fTTree->Branch("fClDetElemNumber", &(fTrkClRes->fClDetElemNumber), "fClDetElemNumber/I");
    fTTree->Branch("fClusterX", &(fTrkClRes->fClusterX), "fClusterX/F");
    fTTree->Branch("fClusterY", &(fTrkClRes->fClusterY), "fClusterY/F");
    fTTree->Branch("fTrackX", &(fTrkClRes->fTrackX), "fTrackX/F");
    fTTree->Branch("fTrackY", &(fTrkClRes->fTrackY), "fTrackY/F");
    fTTree->Branch("fClusterXloc", &(fTrkClRes->fClusterXloc), "fClusterXloc/F");
    fTTree->Branch("fClusterYloc", &(fTrkClRes->fClusterYloc), "fClusterYloc/F");
    fTTree->Branch("fTrackXloc", &(fTrkClRes->fTrackXloc), "fTrackXloc/F");
    fTTree->Branch("fTrackYloc", &(fTrkClRes->fTrackYloc), "fTrackYloc/F");
    fTTree->Branch("fTrackSlopeX", &(fTrkClRes->fTrackSlopeX), "fTrackSlopeX/F");
    fTTree->Branch("fTrackSlopeY", &(fTrkClRes->fTrackSlopeY), "fTrackSlopeY/F");
    fTTree->Branch("fBendingMomentum", &(fTrkClRes->fBendingMomentum), "fBendingMomentum/F");
    fTTree->Branch("fResiduXGlobal", &(fTrkClRes->fResiduXGlobal), "fResiduXGlobal/F");
    fTTree->Branch("fResiduYGlobal", &(fTrkClRes->fResiduYGlobal), "fResiduYGlobal/F");
    fTTree->Branch("fResiduXLocal", &(fTrkClRes->fResiduXLocal), "fResiduXLocal/F");
    fTTree->Branch("fResiduYLocal", &(fTrkClRes->fResiduYLocal), "fResiduYLocal/F");
    fTTree->Branch("fCharge", &(fTrkClRes->fCharge), "fCharge/F");
    fTTree->Branch("fClusterZ", &(fTrkClRes->fClusterZ), "fClusterZ/F");
    fTTree->Branch("fTrackZ", &(fTrkClRes->fTrackZ), "fTrackZ/F");
    fTTree->Branch("fBx", &(fTrkClRes->fBx), "fBx/F");
    fTTree->Branch("fBy", &(fTrkClRes->fBy), "fBy/F");
    fTTree->Branch("fBz", &(fTrkClRes->fBz), "fBz/F");
  }
}

//_____________________________________________________
void Aligner::terminate()
{
  fInitialized = kFALSE;
  LOG(info) << "Closing Evaluation TFile";
  if (fDoEvaluation) {
    if (fTFile && fTTree) {
      fTFile->cd();
      fTTree->Write();
      fTFile->Close();
    }
  }
}

//_____________________________________________________
void Aligner::ProcessTrack(Track& track, const o2::mch::geo::TransformationCreator& transformation, bool doAlignment, double weight)
{

  /// process track for alignment minimization
  // reset track records

  if (fMillepede->GetRecord()) {
    fMillepede->GetRecord()->Reset();
  }

  // loop over clusters to get starting values
  bool first(true);

  auto itTrackParam = track.begin();
  for (; itTrackParam != track.end(); ++itTrackParam) {

    // get cluster
    const Cluster* cluster = itTrackParam->getClusterPtr();
    if (!cluster) {
      continue;
    }
    //  for first valid cluster, save track position as "starting" values
    if (first) {

      first = false;
      FillTrackParamData(&*itTrackParam);
      fTrackPos0[0] = fTrackPos[0];
      fTrackPos0[1] = fTrackPos[1];
      fTrackPos0[2] = fTrackPos[2];
      fTrackSlope0[0] = fTrackSlope[0];
      fTrackSlope0[1] = fTrackSlope[1];

      break;
    }
  }

  // redo straight track fit
  if (fRefitStraightTracks) {

    // refit straight track
    const LocalTrackParam trackParam(RefitStraightTrack(track, fTrackPos0[2]));

    /*
    copy new parameters to stored ones for derivatives calculation
    this is done only if BFieldOn is false, for which these parameters are used
    */
    if (!fBFieldOn) {
      fTrackPos0[0] = trackParam.fTrackX;
      fTrackPos0[1] = trackParam.fTrackY;
      fTrackPos0[2] = trackParam.fTrackZ;
      fTrackSlope0[0] = trackParam.fTrackSlopeX;
      fTrackSlope0[1] = trackParam.fTrackSlopeY;
    }
  }

  // second loop to perform alignment
  itTrackParam = track.begin();
  for (; itTrackParam != track.end(); ++itTrackParam) {

    // get cluster
    const Cluster* cluster = itTrackParam->getClusterPtr();
    if (!cluster) {
      continue;
    }

    // fill local variables for this position --> one measurement

    FillDetElemData(cluster); // function to get the transformation matrix
    FillRecPointData(cluster);
    FillTrackParamData(&*itTrackParam);

    // 'inverse' (GlobalToLocal) rotation matrix
    // const double* r(fGeoCombiTransInverse.GetRotationMatrix());
    o2::math_utils::Transform3D trans = transformation(cluster->getDEId());
    // LOG(info) << Form("cluster ID: %i", cluster->getDEId());
    TMatrixD transMat(3, 4);
    trans.GetTransformMatrix(transMat);
    // transMat.Print();
    double r[12];
    r[0] = transMat(0, 0);
    r[1] = transMat(0, 1);
    r[2] = transMat(0, 2);
    r[3] = transMat(1, 0);
    r[4] = transMat(1, 1);
    r[5] = transMat(1, 2);
    r[6] = transMat(2, 0);
    r[7] = transMat(2, 1);
    r[8] = transMat(2, 2);
    r[9] = transMat(0, 3);
    r[10] = transMat(1, 3);
    r[11] = transMat(2, 3);
    // calculate measurements
    if (fBFieldOn) {

      // use residuals (cluster - track) for measurement
      fMeas[0] = r[0] * (fClustPos[0] - fTrackPos[0]) + r[1] * (fClustPos[1] - fTrackPos[1]);
      fMeas[1] = r[3] * (fClustPos[0] - fTrackPos[0]) + r[4] * (fClustPos[1] - fTrackPos[1]);
    } else {

      // use cluster position for measurement
      fMeas[0] = (r[0] * fClustPos[0] + r[1] * fClustPos[1]);
      fMeas[1] = (r[3] * fClustPos[0] + r[4] * fClustPos[1]);
    }
    // printf("DE %d, X: %f %f ; Y: %f %f ; Z: %f\n", cluster->getDEId(), fClustPos[0], fTrackPos[0], fClustPos[1], fTrackPos[1], fClustPos[2]);

    if (fDoEvaluation) {

      const float InvBendingMom = itTrackParam->getInverseBendingMomentum();
      const float TrackCharge = itTrackParam->getCharge();

      double B[3] = {0.0, 0.0, 0.0};
      double x[3] = {fTrackPos[0], fTrackPos[1], fTrackPos[2]};
      TGeoGlobalMagField::Instance()->Field(x, B);
      const float Bx = B[0];
      const float By = B[1];
      const float Bz = B[2];

      fTrkClRes->fClDetElem = cluster->getDEId();
      fTrkClRes->fClDetElemNumber = GetDetElemNumber(cluster->getDEId());
      fTrkClRes->fClusterX = fClustPos[0];
      fTrkClRes->fClusterY = fClustPos[1];
      fTrkClRes->fClusterZ = fClustPos[2];

      // fTrkClRes->fTrackX = fTrackPos0[0] + fTrackSlope0[0] * (fTrackPos[2] - fTrackPos0[2]); // fTrackPos[0];
      // fTrkClRes->fTrackY = fTrackPos0[1] + fTrackSlope0[1] * (fTrackPos[2] - fTrackPos0[2]); // fTrackPos[1];
      // fTrkClRes->fTrackSlopeX = fTrackSlope0[0];
      // fTrkClRes->fTrackSlopeY = fTrackSlope0[1];

      fTrkClRes->fTrackX = fTrackPos[0];
      fTrkClRes->fTrackY = fTrackPos[1];
      fTrkClRes->fTrackZ = fTrackPos[2];

      fTrkClRes->fClusterXloc = r[0] * fClustPos[0] + r[1] * fClustPos[1];
      fTrkClRes->fClusterYloc = r[3] * fClustPos[0] + r[4] * fClustPos[1];

      fTrkClRes->fTrackXloc = r[0] * fTrackPos[0] + r[1] * fTrackPos[1];
      fTrkClRes->fTrackYloc = r[3] * fTrackPos[0] + r[4] * fTrackPos[1];

      fTrkClRes->fTrackSlopeX = fTrackSlope[0];
      fTrkClRes->fTrackSlopeY = fTrackSlope[1];

      fTrkClRes->fBendingMomentum = TrackCharge / InvBendingMom;

      fTrkClRes->fResiduXGlobal = fClustPos[0] - fTrackPos[0];
      fTrkClRes->fResiduYGlobal = fClustPos[1] - fTrackPos[1];
      fTrkClRes->fResiduXLocal = r[0] * (fClustPos[0] - fTrackPos[0]) + r[1] * (fClustPos[1] - fTrackPos[1]);
      fTrkClRes->fResiduYLocal = r[3] * (fClustPos[0] - fTrackPos[0]) + r[4] * (fClustPos[1] - fTrackPos[1]);

      fTrkClRes->fCharge = TrackCharge;

      fTrkClRes->fBx = Bx;
      fTrkClRes->fBy = By;
      fTrkClRes->fBz = Bz;

      if (fTTree) {
        fTTree->Fill();
      }
    }
    // Set local equations
    LocalEquationX(r);
    LocalEquationY(r);
  }

  // copy track record
  if (!fDisableRecordWriter) {
    mRecordWriter->setRecordRun(fRunNumber);
    mRecordWriter->setRecordWeight(weight);
  }

  // save record data
  if (doAlignment) {
    if (!fDisableRecordWriter) {
      mRecordWriter->fillRecordTree();
    }
  }
}

//_____________________________________________________________________
void Aligner::FixAll(unsigned int mask)
{
  /// fix parameters matching mask, for all chambers
  LOG(info) << "Fixing " << GetParameterMaskString(mask).Data() << " for all detector elements";

  // fix all stations
  for (int i = 0; i < fgNDetElem; ++i) {
    if (mask & ParX) {
      FixParameter(i, 0);
    }
    if (mask & ParY) {
      FixParameter(i, 1);
    }
    if (mask & ParTZ) {
      FixParameter(i, 2);
    }
    if (mask & ParZ) {
      FixParameter(i, 3);
    }
  }
}

//_____________________________________________________________________
void Aligner::FixChamber(int iCh, unsigned int mask)
{
  /// fix parameters matching mask, for all detector elements in a given chamber, counting from 1

  // check boundaries
  if (iCh < 1 || iCh > 10) {
    LOG(fatal) << "Invalid chamber index " << iCh;
  }

  // get first and last element
  const int iDetElemFirst = fgSNDetElemCh[iCh - 1];
  const int iDetElemLast = fgSNDetElemCh[iCh];
  for (int i = iDetElemFirst; i < iDetElemLast; ++i) {

    LOG(info) << "Fixing " << GetParameterMaskString(mask).Data() << " for detector element " << i;

    if (mask & ParX) {
      FixParameter(i, 0);
    }
    if (mask & ParY) {
      FixParameter(i, 1);
    }
    if (mask & ParTZ) {
      FixParameter(i, 2);
    }
    if (mask & ParZ) {
      FixParameter(i, 3);
    }
  }
}

//_____________________________________________________________________
void Aligner::FixDetElem(int iDetElemId, unsigned int mask)
{
  /// fix parameters matching mask, for a given detector element, counting from 0
  const int iDet(GetDetElemNumber(iDetElemId));
  if (mask & ParX) {
    FixParameter(iDet, 0);
  }
  if (mask & ParY) {
    FixParameter(iDet, 1);
  }
  if (mask & ParTZ) {
    FixParameter(iDet, 2);
  }
  if (mask & ParZ) {
    FixParameter(iDet, 3);
  }
}

//_____________________________________________________________________
void Aligner::FixHalfSpectrometer(const bool* lChOnOff, unsigned int sidesMask, unsigned int mask)
{

  /// Fix parameters matching mask for all detectors in selected chambers and selected sides of the spectrometer
  for (int i = 0; i < fgNDetElem; ++i) {

    // get chamber matching detector
    const int iCh(GetChamberId(i));
    if (!lChOnOff[iCh - 1]) {
      continue;
    }

    // get detector element in chamber
    int lDetElemNumber = i - fgSNDetElemCh[iCh - 1];

    // skip detector if its side is off
    // stations 1 and 2
    if (iCh >= 1 && iCh <= 4) {
      if (lDetElemNumber == 0 && !(sidesMask & SideTopRight)) {
        continue;
      }
      if (lDetElemNumber == 1 && !(sidesMask & SideTopLeft)) {
        continue;
      }
      if (lDetElemNumber == 2 && !(sidesMask & SideBottomLeft)) {
        continue;
      }
      if (lDetElemNumber == 3 && !(sidesMask & SideBottomRight)) {
        continue;
      }
    }

    // station 3
    if (iCh >= 5 && iCh <= 6) {
      if (lDetElemNumber >= 0 && lDetElemNumber <= 4 && !(sidesMask & SideTopRight)) {
        continue;
      }
      if (lDetElemNumber >= 5 && lDetElemNumber <= 10 && !(sidesMask & SideTopLeft)) {
        continue;
      }
      if (lDetElemNumber >= 11 && lDetElemNumber <= 13 && !(sidesMask & SideBottomLeft)) {
        continue;
      }
      if (lDetElemNumber >= 14 && lDetElemNumber <= 17 && !(sidesMask & SideBottomRight)) {
        continue;
      }
    }

    // stations 4 and 5
    if (iCh >= 7 && iCh <= 10) {
      if (lDetElemNumber >= 0 && lDetElemNumber <= 6 && !(sidesMask & SideTopRight)) {
        continue;
      }
      if (lDetElemNumber >= 7 && lDetElemNumber <= 13 && !(sidesMask & SideTopLeft)) {
        continue;
      }
      if (lDetElemNumber >= 14 && lDetElemNumber <= 19 && !(sidesMask & SideBottomLeft)) {
        continue;
      }
      if (lDetElemNumber >= 20 && lDetElemNumber <= 25 && !(sidesMask & SideBottomRight)) {
        continue;
      }
    }

    // detector is accepted, fix it
    FixDetElem(i, mask);
  }
}

//______________________________________________________________________
void Aligner::FixParameter(int iPar)
{

  /// fix a given parameter, counting from 0
  if (fInitialized) {
    LOG(fatal) << "Millepede already initialized";
  }

  fGlobalParameterStatus[iPar] = kFixedParId;
}

//_____________________________________________________________________
void Aligner::ReleaseChamber(int iCh, unsigned int mask)
{
  /// release parameters matching mask, for all detector elements in a given chamber, counting from 1

  // check boundaries
  if (iCh < 1 || iCh > 10) {
    LOG(fatal) << "Invalid chamber index " << iCh;
  }

  // get first and last element
  const int iDetElemFirst = fgSNDetElemCh[iCh - 1];
  const int iDetElemLast = fgSNDetElemCh[iCh];
  for (int i = iDetElemFirst; i < iDetElemLast; ++i) {

    LOG(info) << "Releasing " << GetParameterMaskString(mask).Data() << " for detector element " << i;

    if (mask & ParX) {
      ReleaseParameter(i, 0);
    }
    if (mask & ParY) {
      ReleaseParameter(i, 1);
    }
    if (mask & ParTZ) {
      ReleaseParameter(i, 2);
    }
    if (mask & ParZ) {
      ReleaseParameter(i, 3);
    }
  }
}

//_____________________________________________________________________
void Aligner::ReleaseDetElem(int iDetElemId, unsigned int mask)
{
  /// release parameters matching mask, for a given detector element, counting from 0
  const int iDet(GetDetElemNumber(iDetElemId));
  if (mask & ParX) {
    ReleaseParameter(iDet, 0);
  }
  if (mask & ParY) {
    ReleaseParameter(iDet, 1);
  }
  if (mask & ParTZ) {
    ReleaseParameter(iDet, 2);
  }
  if (mask & ParZ) {
    ReleaseParameter(iDet, 3);
  }
}

//______________________________________________________________________
void Aligner::ReleaseParameter(int iPar)
{

  /// release a given parameter, counting from 0
  if (fInitialized) {
    LOG(fatal) << "Millepede already initialized";
  }

  fGlobalParameterStatus[iPar] = kFreeParId;
}

//_____________________________________________________________________
void Aligner::GroupChamber(int iCh, unsigned int mask)
{
  /// group parameters matching mask for all detector elements in a given chamber, counting from 1
  if (iCh < 1 || iCh > fgNCh) {
    LOG(fatal) << "Invalid chamber index " << iCh;
  }

  const int detElemMin = 100 * iCh;
  const int detElemMax = 100 * iCh + fgNDetElemCh[iCh] - 1;
  GroupDetElems(detElemMin, detElemMax, mask);
}

//_____________________________________________________________________
void Aligner::GroupHalfChamber(int iCh, int iHalf, unsigned int mask)
{
  /// group parameters matching mask for all detector elements in a given tracking module (= half chamber), counting from 0
  if (iCh < 1 || iCh > fgNCh) {
    LOG(fatal) << "Invalid chamber index " << iCh;
  }

  if (iHalf < 0 || iHalf > 1) {
    LOG(fatal) << "Invalid half chamber index " << iHalf;
  }

  const int iHalfCh = 2 * (iCh - 1) + iHalf;
  GroupDetElems(&fgDetElemHalfCh[iHalfCh][0], fgNDetElemHalfCh[iHalfCh], mask);
}

//_____________________________________________________________________
void Aligner::GroupDetElems(int detElemMin, int detElemMax, unsigned int mask)
{
  /// group parameters matching mask for all detector elements between min and max
  // check number of detector elements
  const int nDetElem = detElemMax - detElemMin + 1;
  if (nDetElem < 2) {
    LOG(fatal) << "Requested group of DEs " << detElemMin << "-" << detElemMax << " contains less than 2 DE's";
  }

  // create list
  int* detElemList = new int[nDetElem];
  for (int i = 0; i < nDetElem; ++i) {
    detElemList[i] = detElemMin + i;
  }

  // group
  GroupDetElems(detElemList, nDetElem, mask);
  delete[] detElemList;
}

//_____________________________________________________________________
void Aligner::GroupDetElems(const int* detElemList, int nDetElem, unsigned int mask)
{
  /// group parameters matching mask for all detector elements in list
  if (fInitialized) {
    LOG(fatal) << "Millepede already initialized";
  }

  const int iDeBase(GetDetElemNumber(detElemList[0]));
  for (int i = 0; i < nDetElem; ++i) {
    const int iDeCurrent(GetDetElemNumber(detElemList[i]));
    if (mask & ParX) {
      fGlobalParameterStatus[iDeCurrent * fgNParCh + 0] = (i == 0) ? kGroupBaseId : (kGroupBaseId - iDeBase - 1);
    }
    if (mask & ParY) {
      fGlobalParameterStatus[iDeCurrent * fgNParCh + 1] = (i == 0) ? kGroupBaseId : (kGroupBaseId - iDeBase - 1);
    }
    if (mask & ParTZ) {
      fGlobalParameterStatus[iDeCurrent * fgNParCh + 2] = (i == 0) ? kGroupBaseId : (kGroupBaseId - iDeBase - 1);
    }
    if (mask & ParZ) {
      fGlobalParameterStatus[iDeCurrent * fgNParCh + 3] = (i == 0) ? kGroupBaseId : (kGroupBaseId - iDeBase - 1);
    }

    if (i == 0) {
      LOG(info) << "Creating new group for detector " << detElemList[i] << " and variable " << GetParameterMaskString(mask).Data();
    } else {
      LOG(info) << "Adding detector element " << detElemList[i] << " to current group";
    }
  }
}

//______________________________________________________________________
void Aligner::SetChamberNonLinear(int iCh, unsigned int mask)
{
  /// Set parameters matching mask as non linear, for all detector elements in a given chamber, counting from 1
  const int iDetElemFirst = fgSNDetElemCh[iCh - 1];
  const int iDetElemLast = fgSNDetElemCh[iCh];
  for (int i = iDetElemFirst; i < iDetElemLast; ++i) {

    if (mask & ParX) {
      SetParameterNonLinear(i, 0);
    }
    if (mask & ParY) {
      SetParameterNonLinear(i, 1);
    }
    if (mask & ParTZ) {
      SetParameterNonLinear(i, 2);
    }
    if (mask & ParZ) {
      SetParameterNonLinear(i, 3);
    }
  }
}

//_____________________________________________________________________
void Aligner::SetDetElemNonLinear(int iDetElemId, unsigned int mask)
{
  /// Set parameters matching mask as non linear, for a given detector element, counting from 0
  const int iDet(GetDetElemNumber(iDetElemId));
  if (mask & ParX) {
    SetParameterNonLinear(iDet, 0);
  }
  if (mask & ParY) {
    SetParameterNonLinear(iDet, 1);
  }
  if (mask & ParTZ) {
    SetParameterNonLinear(iDet, 2);
  }
  if (mask & ParZ) {
    SetParameterNonLinear(iDet, 3);
  }
}

//______________________________________________________________________
void Aligner::SetParameterNonLinear(int iPar)
{
  /// Set nonlinear flag for parameter iPar
  if (!fInitialized) {
    LOG(fatal) << "Millepede not initialized";
  }

  fMillepede->SetNonLinear(iPar);
  LOG(info) << "Parameter " << iPar << " set to non linear ";
}

//______________________________________________________________________
void Aligner::AddConstraints(const bool* lChOnOff, unsigned int mask)
{
  /// Add constraint equations for selected chambers and degrees of freedom

  Array fConstraintX;
  Array fConstraintY;
  Array fConstraintTZ;
  Array fConstraintZ;

  for (int i = 0; i < fgNDetElem; ++i) {

    // get chamber matching detector
    const int iCh(GetChamberId(i));
    if (lChOnOff[iCh - 1]) {

      if (mask & ParX) {
        fConstraintX.values[i * fgNParCh + 0] = 1.0;
      }
      if (mask & ParY) {
        fConstraintY.values[i * fgNParCh + 1] = 1.0;
      }
      if (mask & ParTZ) {
        fConstraintTZ.values[i * fgNParCh + 2] = 1.0;
      }
      if (mask & ParZ) {
        fConstraintTZ.values[i * fgNParCh + 3] = 1.0;
      }
    }
  }

  if (mask & ParX) {
    AddConstraint(fConstraintX.values, 0.0);
  }
  if (mask & ParY) {
    AddConstraint(fConstraintY.values, 0.0);
  }
  if (mask & ParTZ) {
    AddConstraint(fConstraintTZ.values, 0.0);
  }
  if (mask & ParZ) {
    AddConstraint(fConstraintZ.values, 0.0);
  }
}

//______________________________________________________________________
void Aligner::AddConstraints(const bool* lChOnOff, const bool* lVarXYT, unsigned int sidesMask)
{
  /*
  questions:
  - is there not redundancy/inconsistency between lDetTLBR and lSpecLROnOff ? shouldn't we use only lDetTLBR ?
  - why is weight ignored for ConstrainT and ConstrainB
  - why is there no constrain on z
  */

  /// Add constraint equations for selected chambers, degrees of freedom and detector half
  double lMeanY = 0.;
  double lSigmaY = 0.;
  double lMeanZ = 0.;
  double lSigmaZ = 0.;
  int lNDetElem = 0;

  for (int i = 0; i < fgNDetElem; ++i) {

    // get chamber matching detector
    const int iCh(GetChamberId(i));

    // skip detector if chamber is off
    if (lChOnOff[iCh - 1]) {
      continue;
    }

    // get detector element id from detector element number
    const int lDetElemNumber = i - fgSNDetElemCh[iCh - 1];
    const int lDetElemId = iCh * 100 + lDetElemNumber;

    // skip detector if its side is off
    // stations 1 and 2
    if (iCh >= 1 && iCh <= 4) {
      if (lDetElemNumber == 0 && !(sidesMask & SideTopRight)) {
        continue;
      }
      if (lDetElemNumber == 1 && !(sidesMask & SideTopLeft)) {
        continue;
      }
      if (lDetElemNumber == 2 && !(sidesMask & SideBottomLeft)) {
        continue;
      }
      if (lDetElemNumber == 3 && !(sidesMask & SideBottomRight)) {
        continue;
      }
    }

    // station 3
    if (iCh >= 5 && iCh <= 6) {
      if (lDetElemNumber >= 0 && lDetElemNumber <= 4 && !(sidesMask & SideTopRight)) {
        continue;
      }
      if (lDetElemNumber >= 5 && lDetElemNumber <= 10 && !(sidesMask & SideTopLeft)) {
        continue;
      }
      if (lDetElemNumber >= 11 && lDetElemNumber <= 13 && !(sidesMask & SideBottomLeft)) {
        continue;
      }
      if (lDetElemNumber >= 14 && lDetElemNumber <= 17 && !(sidesMask & SideBottomRight)) {
        continue;
      }
    }

    // stations 4 and 5
    if (iCh >= 7 && iCh <= 10) {
      if (lDetElemNumber >= 0 && lDetElemNumber <= 6 && !(sidesMask & SideTopRight)) {
        continue;
      }
      if (lDetElemNumber >= 7 && lDetElemNumber <= 13 && !(sidesMask & SideTopLeft)) {
        continue;
      }
      if (lDetElemNumber >= 14 && lDetElemNumber <= 19 && !(sidesMask & SideBottomLeft)) {
        continue;
      }
      if (lDetElemNumber >= 20 && lDetElemNumber <= 25 && !(sidesMask & SideBottomRight)) {
        continue;
      }
    }

    // get global x, y and z position
    double lDetElemGloX = 0.;
    double lDetElemGloY = 0.;
    double lDetElemGloZ = 0.;

    auto fTransform = fTransformCreator(lDetElemId);
    o2::math_utils::Point3D<double> SlatPos{0.0, 0.0, 0.0};
    o2::math_utils::Point3D<double> GlobalPos;

    fTransform.LocalToMaster(SlatPos, GlobalPos);
    lDetElemGloX = GlobalPos.x();
    lDetElemGloY = GlobalPos.y();
    lDetElemGloZ = GlobalPos.z();
    // fTransform->Local2Global(lDetElemId, 0, 0, 0, lDetElemGloX, lDetElemGloY, lDetElemGloZ);

    // increment mean Y, mean Z, sigmas and number of accepted detectors
    lMeanY += lDetElemGloY;
    lSigmaY += lDetElemGloY * lDetElemGloY;
    lMeanZ += lDetElemGloZ;
    lSigmaZ += lDetElemGloZ * lDetElemGloZ;
    lNDetElem++;
  }

  // calculate mean values
  lMeanY /= lNDetElem;
  lSigmaY /= lNDetElem;
  lSigmaY = TMath::Sqrt(lSigmaY - lMeanY * lMeanY);
  lMeanZ /= lNDetElem;
  lSigmaZ /= lNDetElem;
  lSigmaZ = TMath::Sqrt(lSigmaZ - lMeanZ * lMeanZ);
  LOG(info) << "Used " << lNDetElem << " DetElem, MeanZ= " << lMeanZ << ", SigmaZ= " << lSigmaZ;

  // create all possible arrays
  Array fConstraintX[4];  // Array for constraint equation X
  Array fConstraintY[4];  // Array for constraint equation Y
  Array fConstraintP[4];  // Array for constraint equation P
  Array fConstraintXZ[4]; // Array for constraint equation X vs Z
  Array fConstraintYZ[4]; // Array for constraint equation Y vs Z
  Array fConstraintPZ[4]; // Array for constraint equation P vs Z

  // do we really need these ?
  Array fConstraintXY[4]; // Array for constraint equation X vs Y
  Array fConstraintYY[4]; // Array for constraint equation Y vs Y
  Array fConstraintPY[4]; // Array for constraint equation P vs Y

  // fill bool sides array based on masks, for convenience
  bool lDetTLBR[4];
  lDetTLBR[0] = sidesMask & SideTop;
  lDetTLBR[1] = sidesMask & SideLeft;
  lDetTLBR[2] = sidesMask & SideBottom;
  lDetTLBR[3] = sidesMask & SideRight;

  for (int i = 0; i < fgNDetElem; ++i) {

    // get chamber matching detector
    const int iCh(GetChamberId(i));

    // skip detector if chamber is off
    if (!lChOnOff[iCh - 1]) {
      continue;
    }

    // get detector element id from detector element number
    const int lDetElemNumber = i - fgSNDetElemCh[iCh - 1];
    const int lDetElemId = iCh * 100 + lDetElemNumber;

    // get global x, y and z position
    double lDetElemGloX = 0.;
    double lDetElemGloY = 0.;
    double lDetElemGloZ = 0.;

    auto fTransform = fTransformCreator(lDetElemId);
    o2::math_utils::Point3D<double> SlatPos{0.0, 0.0, 0.0};
    o2::math_utils::Point3D<double> GlobalPos;

    fTransform.LocalToMaster(SlatPos, GlobalPos);
    lDetElemGloX = GlobalPos.x();
    lDetElemGloY = GlobalPos.y();
    lDetElemGloZ = GlobalPos.z();
    // fTransform->Local2Global(lDetElemId, 0, 0, 0, lDetElemGloX, lDetElemGloY, lDetElemGloZ);

    // loop over sides
    for (int iSide = 0; iSide < 4; iSide++) {

      // skip if side is not selected
      if (!lDetTLBR[iSide]) {
        continue;
      }

      // skip detector if it is not in the selected side
      // stations 1 and 2
      if (iCh >= 1 && iCh <= 4) {
        if (lDetElemNumber == 0 && !(iSide == 0 || iSide == 3)) {
          continue; // top-right
        }
        if (lDetElemNumber == 1 && !(iSide == 0 || iSide == 1)) {
          continue; // top-left
        }
        if (lDetElemNumber == 2 && !(iSide == 2 || iSide == 1)) {
          continue; // bottom-left
        }
        if (lDetElemNumber == 3 && !(iSide == 2 || iSide == 3)) {
          continue; // bottom-right
        }
      }

      // station 3
      if (iCh >= 5 && iCh <= 6) {
        if (lDetElemNumber >= 0 && lDetElemNumber <= 4 && !(iSide == 0 || iSide == 3)) {
          continue; // top-right
        }
        if (lDetElemNumber >= 5 && lDetElemNumber <= 9 && !(iSide == 0 || iSide == 1)) {
          continue; // top-left
        }
        if (lDetElemNumber >= 10 && lDetElemNumber <= 13 && !(iSide == 2 || iSide == 1)) {
          continue; // bottom-left
        }
        if (lDetElemNumber >= 14 && lDetElemNumber <= 17 && !(iSide == 2 || iSide == 3)) {
          continue; // bottom-right
        }
      }

      // stations 4 and 5
      if (iCh >= 7 && iCh <= 10) {
        if (lDetElemNumber >= 0 && lDetElemNumber <= 6 && !(iSide == 0 || iSide == 3)) {
          continue; // top-right
        }
        if (lDetElemNumber >= 7 && lDetElemNumber <= 13 && !(iSide == 0 || iSide == 1)) {
          continue; // top-left
        }
        if (lDetElemNumber >= 14 && lDetElemNumber <= 19 && !(iSide == 2 || iSide == 1)) {
          continue; // bottom-left
        }
        if (lDetElemNumber >= 20 && lDetElemNumber <= 25 && !(iSide == 2 || iSide == 3)) {
          continue; // bottom-right
        }
      }

      // constrain x
      if (lVarXYT[0]) {
        fConstraintX[iSide].values[i * fgNParCh + 0] = 1;
      }

      // constrain y
      if (lVarXYT[1]) {
        fConstraintY[iSide].values[i * fgNParCh + 1] = 1;
      }

      // constrain phi (rotation around z)
      if (lVarXYT[2]) {
        fConstraintP[iSide].values[i * fgNParCh + 2] = 1;
      }

      // x-z shearing
      if (lVarXYT[3]) {
        fConstraintXZ[iSide].values[i * fgNParCh + 0] = (lDetElemGloZ - lMeanZ) / lSigmaZ;
      }

      // y-z shearing
      if (lVarXYT[4]) {
        fConstraintYZ[iSide].values[i * fgNParCh + 1] = (lDetElemGloZ - lMeanZ) / lSigmaZ;
      }

      // phi-z shearing
      if (lVarXYT[5]) {
        fConstraintPZ[iSide].values[i * fgNParCh + 2] = (lDetElemGloZ - lMeanZ) / lSigmaZ;
      }

      // x-y shearing
      if (lVarXYT[6]) {
        fConstraintXY[iSide].values[i * fgNParCh + 0] = (lDetElemGloY - lMeanY) / lSigmaY;
      }

      // y-y shearing
      if (lVarXYT[7]) {
        fConstraintYY[iSide].values[i * fgNParCh + 1] = (lDetElemGloY - lMeanY) / lSigmaY;
      }

      // phi-y shearing
      if (lVarXYT[8]) {
        fConstraintPY[iSide].values[i * fgNParCh + 2] = (lDetElemGloY - lMeanY) / lSigmaY;
      }
    }
  }

  // pass constraints to millepede
  for (int iSide = 0; iSide < 4; iSide++) {
    // skip if side is not selected
    if (!lDetTLBR[iSide]) {
      continue;
    }

    if (lVarXYT[0]) {
      AddConstraint(fConstraintX[iSide].values, 0.0);
    }
    if (lVarXYT[1]) {
      AddConstraint(fConstraintY[iSide].values, 0.0);
    }
    if (lVarXYT[2]) {
      AddConstraint(fConstraintP[iSide].values, 0.0);
    }
    if (lVarXYT[3]) {
      AddConstraint(fConstraintXZ[iSide].values, 0.0);
    }
    if (lVarXYT[4]) {
      AddConstraint(fConstraintYZ[iSide].values, 0.0);
    }
    if (lVarXYT[5]) {
      AddConstraint(fConstraintPZ[iSide].values, 0.0);
    }
    if (lVarXYT[6]) {
      AddConstraint(fConstraintXY[iSide].values, 0.0);
    }
    if (lVarXYT[7]) {
      AddConstraint(fConstraintYY[iSide].values, 0.0);
    }
    if (lVarXYT[8]) {
      AddConstraint(fConstraintPY[iSide].values, 0.0);
    }
  }
}

//______________________________________________________________________
void Aligner::InitGlobalParameters(double* par)
{
  /// Initialize global parameters with par array
  if (!fInitialized) {
    LOG(fatal) << "Millepede is not initialized";
  }

  fMillepede->SetGlobalParameters(par);
}

//______________________________________________________________________
void Aligner::SetAllowedVariation(int iPar, double value)
{
  /// "Encouraged" variation for degrees of freedom
  // check initialization
  if (fInitialized) {
    LOG(fatal) << "Millepede already initialized";
  }

  // check initialization
  if (!(iPar >= 0 && iPar < fgNParCh)) {
    LOG(fatal) << "Invalid index: " << iPar;
  }

  fAllowVar[iPar] = value;
}

//______________________________________________________________________
void Aligner::SetSigmaXY(double sigmaX, double sigmaY)
{

  /// Set expected measurement resolution
  fSigma[0] = sigmaX;
  fSigma[1] = sigmaY;

  // print
  for (int i = 0; i < 2; ++i) {
    LOG(info) << "fSigma[" << i << "] =" << fSigma[i];
  }
}

//_____________________________________________________
void Aligner::GlobalFit(std::vector<double>& parameters, std::vector<double>& errors, std::vector<double>& pulls)
{
  /// Call global fit; Global parameters are stored in parameters
  fMillepede->GlobalFit(parameters, errors, pulls);

  LOG(info) << "Done fitting global parameters";
  for (int iDet = 0; iDet < fgNDetElem; ++iDet) {
    LOG(info) << iDet << " " << parameters[iDet * fgNParCh + 0] << " " << parameters[iDet * fgNParCh + 1] << " " << parameters[iDet * fgNParCh + 3] << " " << parameters[iDet * fgNParCh + 2];
  }
}

//_____________________________________________________
void Aligner::PrintGlobalParameters() const
{
  fMillepede->PrintGlobalParameters();
}

//_____________________________________________________
double Aligner::GetParError(int iPar) const
{
  return fMillepede->GetParError(iPar);
}

//______________________________________________________________________
void Aligner::ReAlign(
  std::vector<o2::detectors::AlignParam>& params,
  std::vector<double>& misAlignments)
{

  /// Returns a new AliMUONGeometryTransformer with the found misalignments
  /// applied.

  // Takes the internal geometry module transformers, copies them
  // and gets the Detection Elements from them.
  // Takes misalignment parameters and applies these
  // to the local transform of the Detection Element
  // Obtains the global transform by multiplying the module transformer
  // transformation with the local transformation
  // Applies the global transform to a new detection element
  // Adds the new detection element to a new module transformer
  // Adds the new module transformer to a new geometry transformer
  // Returns the new geometry transformer

  double lModuleMisAlignment[fgNParCh] = {0};
  double lDetElemMisAlignment[fgNParCh] = {0};

  o2::detectors::AlignParam lAP;
  for (int hc = 0; hc < 20; hc++) {

    TGeoCombiTrans localDeltaTransform;
    localDeltaTransform = DeltaTransform(lModuleMisAlignment);

    std::string sname = fmt::format("MCH/HC{}", hc);
    lAP.setSymName(sname.c_str());

    double lPsi, lTheta, lPhi = 0.;
    if (!isMatrixConvertedToAngles(localDeltaTransform.GetRotationMatrix(),
                                   lPsi, lTheta, lPhi)) {
      LOG(error) << "Problem extracting angles!";
    }

    lAP.setGlobalParams(localDeltaTransform);

    // lAP.print();
    lAP.applyToGeometry();
    params.emplace_back(lAP);
    for (int de = 0; de < fgNDetElemHalfCh[hc]; de++) {

      // store detector element id and number
      const int iDetElemId = fgDetElemHalfCh[hc][de];
      if (DetElemIsValid(iDetElemId)) {

        const int iDetElemNumber(GetDetElemNumber(iDetElemId));

        for (int i = 0; i < fgNParCh; ++i) {
          lDetElemMisAlignment[i] = 0.0;
          if (hc < fgNHalfCh) {
            lDetElemMisAlignment[i] = misAlignments[iDetElemNumber * fgNParCh + i];
          }
        }

        sname = fmt::format("MCH/HC{}/DE{}", hc, fgDetElemHalfCh[hc][de]);
        lAP.setSymName(sname.c_str());
        localDeltaTransform = DeltaTransform(lDetElemMisAlignment);

        if (!isMatrixConvertedToAngles(localDeltaTransform.GetRotationMatrix(),
                                       lPsi, lTheta, lPhi)) {
          LOG(error) << "Problem extracting angles for " << sname.c_str();
        }

        lAP.setGlobalParams(localDeltaTransform);
        lAP.applyToGeometry();
        params.emplace_back(lAP);

      } else {

        // "invalid" detector elements come from MTR and are left unchanged
        LOG(info) << fmt::format("Keeping detElement {} unchanged\n", iDetElemId);
      }
    }
  }

  // return params;
}

//______________________________________________________________________
void Aligner::SetAlignmentResolution(const TClonesArray* misAlignArray, int rChId, double chResX, double chResY, double deResX, double deResY)
{

  /// Set alignment resolution to misalign objects to be stored in CDB
  /// if rChId is > 0 set parameters for this chamber only, counting from 1
  TMatrixDSym mChCorrMatrix(6);
  mChCorrMatrix[0][0] = chResX * chResX;
  mChCorrMatrix[1][1] = chResY * chResY;

  TMatrixDSym mDECorrMatrix(6);
  mDECorrMatrix[0][0] = deResX * deResX;
  mDECorrMatrix[1][1] = deResY * deResY;

  o2::detectors::AlignParam* alignMat = nullptr;

  for (int chId = 0; chId <= 9; ++chId) {

    // skip chamber if selection is valid, and does not match
    if (rChId > 0 && chId + 1 != rChId) {
      continue;
    }

    TString chName1;
    TString chName2;
    if (chId < 4) {

      chName1 = Form("GM%d", chId);
      chName2 = Form("GM%d", chId);

    } else {

      chName1 = Form("GM%d", 4 + (chId - 4) * 2);
      chName2 = Form("GM%d", 4 + (chId - 4) * 2 + 1);
    }

    for (int i = 0; i < misAlignArray->GetEntries(); ++i) {

      alignMat = (o2::detectors::AlignParam*)misAlignArray->At(i);
      TString volName(alignMat->getSymName());
      if ((volName.Contains(chName1) &&
           ((volName.Last('/') == volName.Index(chName1) + chName1.Length()) ||
            (volName.Length() == volName.Index(chName1) + chName1.Length()))) ||
          (volName.Contains(chName2) &&
           ((volName.Last('/') == volName.Index(chName2) + chName2.Length()) ||
            (volName.Length() == volName.Index(chName2) + chName2.Length())))) {

        volName.Remove(0, volName.Last('/') + 1);
      }
    }
  }
}

//_____________________________________________________
LocalTrackParam Aligner::RefitStraightTrack(Track& track, double z0) const
{

  // initialize matrices
  TMatrixD AtGASum(4, 4);
  AtGASum.Zero();

  TMatrixD AtGMSum(4, 1);
  AtGMSum.Zero();

  // loop over clusters
  for (auto itTrackParam(track.begin()); itTrackParam != track.end(); ++itTrackParam) {

    // get cluster
    const Cluster* cluster = itTrackParam->getClusterPtr();
    if (!cluster) {
      continue;
    }

    // projection matrix
    TMatrixD A(2, 4);
    A.Zero();
    A(0, 0) = 1;
    A(0, 2) = (cluster->getZ() - z0);
    A(1, 1) = 1;
    A(1, 3) = (cluster->getZ() - z0);

    TMatrixD At(TMatrixD::kTransposed, A);

    // gain matrix
    TMatrixD G(2, 2);
    G.Zero();
    G(0, 0) = 1.0 / Square(cluster->getEx());
    G(1, 1) = 1.0 / Square(cluster->getEy());

    const TMatrixD AtG(At, TMatrixD::kMult, G);
    const TMatrixD AtGA(AtG, TMatrixD::kMult, A);
    AtGASum += AtGA;

    // measurement
    TMatrixD M(2, 1);
    M(0, 0) = cluster->getX();
    M(1, 0) = cluster->getY();
    const TMatrixD AtGM(AtG, TMatrixD::kMult, M);
    AtGMSum += AtGM;
  }

  // perform inversion
  TMatrixD AtGASumInv(TMatrixD::kInverted, AtGASum);
  TMatrixD X(AtGASumInv, TMatrixD::kMult, AtGMSum);

  // fill output parameters
  LocalTrackParam out;
  out.fTrackX = X(0, 0);
  out.fTrackY = X(1, 0);
  out.fTrackZ = z0;
  out.fTrackSlopeX = X(2, 0);
  out.fTrackSlopeY = X(3, 0);

  return out;
}

//_____________________________________________________
void Aligner::FillDetElemData(const Cluster* cluster)
{

  /// Get information of current detection element
  // get detector element number from Alice ID
  const int detElemId = cluster->getDEId();
  fDetElemNumber = GetDetElemNumber(detElemId);
}

//______________________________________________________________________
void Aligner::FillRecPointData(const Cluster* cluster)
{

  /// Get information of current cluster
  fClustPos[0] = cluster->getX();
  fClustPos[1] = cluster->getY();
  fClustPos[2] = cluster->getZ();
}

//______________________________________________________________________
void Aligner::FillTrackParamData(const TrackParam* trackParam)
{

  /// Get information of current track at current cluster
  fTrackPos[0] = trackParam->getNonBendingCoor();
  fTrackPos[1] = trackParam->getBendingCoor();
  fTrackPos[2] = trackParam->getZ();
  fTrackSlope[0] = trackParam->getNonBendingSlope();
  fTrackSlope[1] = trackParam->getBendingSlope();
}

//______________________________________________________________________
void Aligner::LocalEquationX(const double* r)
{
  /// local equation along X

  // 'inverse' (GlobalToLocal) rotation matrix
  // const double* r(fGeoCombiTransInverse.GetRotationMatrix());

  // local derivatives
  SetLocalDerivative(0, r[0]);
  SetLocalDerivative(1, r[0] * (fTrackPos[2] - fTrackPos0[2]));
  // SetLocalDerivative(1, -r[0] * fTrackPos[2]);

  SetLocalDerivative(2, r[1]);
  SetLocalDerivative(3, r[1] * (fTrackPos[2] - fTrackPos0[2]));
  // SetLocalDerivative(3, -r[1] * fTrackPos[2]);

  // global derivatives
  /*
  alignment parameters are
  0: delta_x
  1: delta_y
  2: delta_phiz
  3: delta_z
  */

  SetGlobalDerivative(fDetElemNumber * fgNParCh + 0, -r[0]);
  SetGlobalDerivative(fDetElemNumber * fgNParCh + 1, -r[1]);

  if (fBFieldOn) {

    // use local position for derivatives vs 'delta_phi_z'
    SetGlobalDerivative(fDetElemNumber * fgNParCh + 2, -r[1] * fTrackPos[0] + r[0] * fTrackPos[1]);

    // use local slopes for derivatives vs 'delta_z'
    SetGlobalDerivative(fDetElemNumber * fgNParCh + 3, r[0] * fTrackSlope[0] + r[1] * fTrackSlope[1]);

  } else {

    // local copy of extrapolated track positions
    const double trackPosX = fTrackPos0[0] + fTrackSlope0[0] * (fTrackPos[2] - fTrackPos0[2]);
    const double trackPosY = fTrackPos0[1] + fTrackSlope0[1] * (fTrackPos[2] - fTrackPos0[2]);

    // use properly extrapolated position for derivatives vs 'delta_phi_z'
    SetGlobalDerivative(fDetElemNumber * fgNParCh + 2, -r[1] * trackPosX + r[0] * trackPosY);
    // SetGlobalDerivative(fDetElemNumber * fgNParCh + 2, -r[1] * (trackPosX - r[9]) + r[0] * (trackPosY - r[10]));

    // use slopes at origin for derivatives vs 'delta_z'
    SetGlobalDerivative(fDetElemNumber * fgNParCh + 3, r[0] * fTrackSlope0[0] + r[1] * fTrackSlope0[1]);
  }

  // store local equation
  fMillepede->SetLocalEquation(fGlobalDerivatives, fLocalDerivatives, fMeas[0], fSigma[0]);
}

//______________________________________________________________________
void Aligner::LocalEquationY(const double* r)
{
  /// local equation along Y

  // 'inverse' (GlobalToLocal) rotation matrix
  // const double* r(fGeoCombiTransInverse.GetRotationMatrix());

  // store local derivatives
  SetLocalDerivative(0, r[3]);
  SetLocalDerivative(1, r[3] * (fTrackPos[2] - fTrackPos0[2]));
  // SetLocalDerivative(1, -r[3] * fTrackPos[2]);

  SetLocalDerivative(2, r[4]);
  SetLocalDerivative(3, r[4] * (fTrackPos[2] - fTrackPos0[2]));
  // SetLocalDerivative(3, -r[4] * fTrackPos[2]);

  // set global derivatives
  SetGlobalDerivative(fDetElemNumber * fgNParCh + 0, -r[3]);
  SetGlobalDerivative(fDetElemNumber * fgNParCh + 1, -r[4]);

  if (fBFieldOn) {

    // use local position for derivatives vs 'delta_phi'
    SetGlobalDerivative(fDetElemNumber * fgNParCh + 2, -r[4] * fTrackPos[0] + r[3] * fTrackPos[1]);

    // use local slopes for derivatives vs 'delta_z'
    SetGlobalDerivative(fDetElemNumber * fgNParCh + 3, r[3] * fTrackSlope[0] + r[4] * fTrackSlope[1]);

  } else {

    // local copy of extrapolated track positions
    const double trackPosX = fTrackPos0[0] + fTrackSlope0[0] * (fTrackPos[2] - fTrackPos0[2]);
    const double trackPosY = fTrackPos0[1] + fTrackSlope0[1] * (fTrackPos[2] - fTrackPos0[2]);

    // use properly extrapolated position for derivatives vs 'delta_phi'
    SetGlobalDerivative(fDetElemNumber * fgNParCh + 2, -r[4] * trackPosX + r[3] * trackPosY);

    // use slopes at origin for derivatives vs 'delta_z'
    SetGlobalDerivative(fDetElemNumber * fgNParCh + 3, r[3] * fTrackSlope0[0] + r[4] * fTrackSlope0[1]);
  }

  // store local equation
  fMillepede->SetLocalEquation(fGlobalDerivatives, fLocalDerivatives, fMeas[1], fSigma[1]);
}

//_________________________________________________________________________
TGeoCombiTrans Aligner::DeltaTransform(const double* lMisAlignment) const
{
  /// Get Delta Transformation, based on alignment parameters

  // translation
  const TGeoTranslation deltaTrans(lMisAlignment[0], lMisAlignment[1], lMisAlignment[3]);

  // rotation
  TGeoRotation deltaRot;
  deltaRot.RotateZ(lMisAlignment[2] * 180. / TMath::Pi());

  // combined rotation and translation.
  return TGeoCombiTrans(deltaTrans, deltaRot);
}

bool Aligner::isMatrixConvertedToAngles(const double* rot, double& psi, double& theta, double& phi) const
{
  /// Calculates the Euler angles in "x y z" notation
  /// using the rotation matrix
  /// Returns false in case the rotation angles can not be
  /// extracted from the matrix
  //
  if (std::abs(rot[0]) < 1e-7 || std::abs(rot[8]) < 1e-7) {
    LOG(error) << "Failed to extract roll-pitch-yall angles!";
    return false;
  }
  psi = std::atan2(-rot[5], rot[8]);
  theta = std::asin(rot[2]);
  phi = std::atan2(-rot[1], rot[0]);
  return true;
}

//______________________________________________________________________
void Aligner::AddConstraint(double* par, double value)
{
  /// Constrain equation defined by par to value
  if (!fInitialized) {
    LOG(fatal) << "Millepede is not initialized";
  }

  std::vector<double> vpar(fNGlobal);
  for (int i = 0; i < fNGlobal; i++) {
    vpar[i] = par[i];
  }

  fMillepede->SetGlobalConstraint(vpar, value);
}

//______________________________________________________________________
bool Aligner::DetElemIsValid(int iDetElemId) const
{
  /// return true if given detector element is valid (and belongs to muon tracker)
  const int iCh = iDetElemId / 100;
  const int iDet = iDetElemId % 100;
  return (iCh > 0 && iCh <= fgNCh && iDet < fgNDetElemCh[iCh - 1]);
}

//______________________________________________________________________
int Aligner::GetDetElemNumber(int iDetElemId) const
{
  /// get det element number from ID
  // get chamber and element number in chamber
  const int iCh = iDetElemId / 100;
  const int iDet = iDetElemId % 100;

  // make sure detector index is valid
  if (!(iCh > 0 && iCh <= fgNCh && iDet < fgNDetElemCh[iCh - 1])) {
    LOG(fatal) << "Invalid detector element id: " << iDetElemId;
  }

  // add number of detectors up to this chamber
  return iDet + fgSNDetElemCh[iCh - 1];
}

//______________________________________________________________________
int Aligner::GetChamberId(int iDetElemNumber) const
{
  /// get chamber (counting from 1) matching a given detector element id
  int iCh(0);
  for (iCh = 0; iCh < fgNCh; iCh++) {
    if (iDetElemNumber < fgSNDetElemCh[iCh]) {
      break;
    }
  }

  return iCh;
}

//______________________________________________________________________
TString Aligner::GetParameterMaskString(unsigned int mask) const
{
  TString out;
  if (mask & ParX) {
    out += "X";
  }
  if (mask & ParY) {
    out += "Y";
  }
  if (mask & ParZ) {
    out += "Z";
  }
  if (mask & ParTZ) {
    out += "T";
  }
  return out;
}

//______________________________________________________________________
TString Aligner::GetSidesMaskString(unsigned int mask) const
{
  TString out;
  if (mask & SideTop) {
    out += "T";
  }
  if (mask & SideLeft) {
    out += "L";
  }
  if (mask & SideBottom) {
    out += "B";
  }
  if (mask & SideRight) {
    out += "R";
  }
  return out;
}

} // namespace mch
} // namespace o2