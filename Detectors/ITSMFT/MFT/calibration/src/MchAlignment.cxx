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
/// \file Alignment
/// Alignment class for the ALICE DiMuon spectrometer
///
/// MUON specific alignment class which interface to AliMillepede.
/// For each track ProcessTrack calculates the local and global derivatives
/// at each cluster and fill the corresponding local equations. Provide methods
/// for fixing or constraining detection elements for best results.
///
/// \author Javier Castillo Castellanos
//-----------------------------------------------------------------------------

#include "MCHAlign/Alignment.h"
#include "MCHAlign/MillePede2.h"
#include "MCHAlign/MillePedeRecord.h"
#include <iostream>

#include "MCHTracking/Track.h"
#include "MCHTracking/TrackParam.h"
#include "MCHTracking/Cluster.h"
#include "TGeoManager.h"

// #include "DataFormatsMCH/ROFRecord.h"
// #include "DataFormatsMCH/TrackMCH.h"
// #include "DataFormatsMCH/Cluster.h"
// #include "DataFormatsMCH/Digit.h"

// #include "AliMUONGeometryTransformer.h"
// #include "AliMUONGeometryModuleTransformer.h"
// #include "MCHAlign/AliMUONGeometryDetElement.h"
// #include "AliMUONGeometryBuilder.h"
#include "MCHGeometryCreator/Geometry.h"
#include "MCHGeometryTest/Helpers.h"
#include "MCHGeometryTransformer/Transformations.h"
#include "TGeoManager.h"

// #include "Align/Millepede2Record.h" //to be replaced
// #include "AliMpExMap.h"
// #include "AliMpExMapIterator.h"

#include "DetectorsCommonDataFormats/AlignParam.h"
#include "Framework/Logger.h"

#include <TMath.h>
#include <TMatrixDSym.h>
#include <TMatrixD.h>
#include <TClonesArray.h>
#include <TGraphErrors.h>
#include <TObject.h>

namespace o2
{
namespace mch
{

using namespace std;

//_____________________________________________________________________
// static variables
const Int_t Alignment::fgNDetElemCh[Alignment::fgNCh] = {4, 4, 4, 4, 18, 18, 26, 26, 26, 26};
const Int_t Alignment::fgSNDetElemCh[Alignment::fgNCh + 1] = {0, 4, 8, 12, 16, 34, 52, 78, 104, 130, 156};

// number of detector elements in each half-chamber
const Int_t Alignment::fgNDetElemHalfCh[Alignment::fgNHalfCh] = {2, 2, 2, 2, 2, 2, 2, 2, 9, 9, 9, 9, 13, 13, 13, 13, 13, 13, 13, 13};

// list of detector elements for each half chamber
const Int_t Alignment::fgDetElemHalfCh[Alignment::fgNHalfCh][Alignment::fgNDetHalfChMax] =
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
  Array(void)
  {
    for (Int_t i = 0; i < Alignment::fNGlobal; ++i) {
      values[i] = 0;
    }
  }

  /// array
  Double_t values[Alignment::fNGlobal];

 private:
  /// Not implemented
  Array(const Array&);

  /// Not implemented
  Array& operator=(const Array&);
};

//________________________________________________________________________
Double_t Square(Double_t x) { return x * x; }

//_____________________________________________________________________
Alignment::Alignment()
  : TObject(),
    fInitialized(kFALSE),
    fRunNumber(0),
    fBFieldOn(kFALSE),
    fRefitStraightTracks(kFALSE),
    fStartFac(256),
    fResCutInitial(100),
    fResCut(100),
    fMillepede(0L), // to be modified
    fCluster(0L),
    fNStdDev(3),
    fDetElemNumber(0),
    fTrackRecord(),
    fTransformCreator(),
    fGeoCombiTransInverse(),
    fDoEvaluation(kFALSE),
    fTrackParamOrig(0),
    fTrackParamNew(0),
    fTFile(0),
    fTTree(0)
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
  fMillepede = new MillePede2();
  // fMillepede = new o2::align::Mille("theMilleFile.txt"); // To be replaced by MillePede2

  // initialize degrees of freedom
  // by default all parameters are free
  for (Int_t iPar = 0; iPar < fNGlobal; ++iPar) {
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

//_____________________________________________________________________
// Alignment::~Alignment()
//{
//  /// destructor
//}
// Alignment::~Alignment() = default;
//_____________________________________________________________________
void Alignment::init(void)
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

  // assign proper groupID to free parameters
  Int_t nGlobal = 0;
  for (Int_t iPar = 0; iPar < fNGlobal; ++iPar) {

    if (fGlobalParameterStatus[iPar] == kFixedParId) {
      // fixed parameters are left unchanged
      continue;

    } else if (fGlobalParameterStatus[iPar] == kFreeParId || fGlobalParameterStatus[iPar] == kGroupBaseId) {

      // free parameters or first element of group are assigned a new group id
      fGlobalParameterStatus[iPar] = nGlobal++;
      continue;

    } else if (fGlobalParameterStatus[iPar] < kGroupBaseId) {

      // get detector element id from status, get chamber parameter id
      const Int_t iDeBase(kGroupBaseId - 1 - fGlobalParameterStatus[iPar]);
      const Int_t iParBase = iPar % fgNParCh;

      // check
      if (iDeBase < 0 || iDeBase >= iPar / fgNParCh) {
        LOG(fatal) << "Group for parameter index " << iPar << " has wrong base detector element: " << iDeBase;
      }

      // assign identical group id to current
      fGlobalParameterStatus[iPar] = fGlobalParameterStatus[iDeBase * fgNParCh + iParBase];
      LOG(info) << "Parameter " << iPar << " grouped to detector " << iDeBase << " (" << GetParameterMaskString(1 << iParBase).Data() << ")";

    } else
      LOG(fatal) << "Unrecognized parameter status for index " << iPar << ": " << fGlobalParameterStatus[iPar];
  }

  LOG(info) << "Free Parameters: " << nGlobal << " out of " << fNGlobal;

  // initialize millepede
  // fMillepede->InitMille(fNGlobal, fNLocal, fNStdDev, fResCut, fResCutInitial, fGlobalParameterStatus);
  fMillepede->InitMille(fNGlobal, fNLocal, fNStdDev, fResCut, fResCutInitial); // MillePede2 implementation

  fInitialized = kTRUE;

  // some debug output
  for (Int_t iPar = 0; iPar < fgNParCh; ++iPar) {
    LOG(info) << "fAllowVar[" << iPar << "]= " << fAllowVar[iPar];
  }

  // set allowed variations for all parameters
  for (Int_t iDet = 0; iDet < fgNDetElem; ++iDet) {
    for (Int_t iPar = 0; iPar < fgNParCh; ++iPar) {
      fMillepede->SetParSigma(iDet * fgNParCh + iPar, fAllowVar[iPar]);
    }
  }

  // Set iterations
  if (fStartFac > 1) {
    fMillepede->SetIterations(fStartFac);
  }
  // setup monitoring TFile
  if (fDoEvaluation && fRefitStraightTracks) {
    fTFile = new TFile("Alignment.root", "RECREATE");
    fTTree = new TTree("TreeE", "Evaluation");

    const Int_t kSplitlevel = 98;
    const Int_t kBufsize = 32000;

    fTrackParamOrig = new LocalTrackParam();
    fTTree->Branch("fTrackParamOrig", "LocalTrackParam", &fTrackParamOrig, kBufsize, kSplitlevel);

    fTrackParamNew = new LocalTrackParam();
    fTTree->Branch("fTrackParamNew", "LocalTrackParam", &fTrackParamNew, kBufsize, kSplitlevel);
  }
}

//_____________________________________________________
void Alignment::terminate(void)
{
  LOG(info) << "Closing Evaluation TFile";
  if (fTFile && fTTree) {
    fTFile->cd();
    fTTree->Write();
    fTFile->Close();
  }
}

//_____________________________________________________
MillePedeRecord* Alignment::ProcessTrack(Track& track, Bool_t doAlignment, Double_t weight)
{
  /// process track for alignment minimization
  /**
  returns the alignment records for this track.
  They can be stored in some output for later reprocessing.
  */

  // reset track records
  fTrackRecord.Reset();
  if (fMillepede->GetRecord()) {
    fMillepede->GetRecord()->Reset();
  }

  // loop over clusters to get starting values
  Bool_t first(kTRUE);
  // if (!trackParam)
  // continue;
  for (auto itTrackParam(track.begin()); itTrackParam != track.end(); ++itTrackParam) {

    // get cluster
    const Cluster* Cluster = itTrackParam->getClusterPtr();
    if (!cluster)
      continue;

    // for first valid cluster, save track position as "starting" values
    if (first) {

      first = kFALSE;
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

    // fill evaluation tree
    if (fTrackParamOrig) {
      fTrackParamOrig->fTrackX = fTrackPos0[0];
      fTrackParamOrig->fTrackY = fTrackPos0[1];
      fTrackParamOrig->fTrackZ = fTrackPos0[2];
      fTrackParamOrig->fTrackSlopeX = fTrackSlope[0];
      fTrackParamOrig->fTrackSlopeY = fTrackSlope[1];
    }

    // new ones
    if (fTrackParamNew) {
      fTrackParamNew->fTrackX = trackParam.fTrackX;
      fTrackParamNew->fTrackY = trackParam.fTrackY;
      fTrackParamNew->fTrackZ = trackParam.fTrackZ;
      fTrackParamNew->fTrackSlopeX = trackParam.fTrackSlopeX;
      fTrackParamNew->fTrackSlopeY = trackParam.fTrackSlopeY;
    }

    if (fTTree)
      fTTree->Fill();

    /*
    copy new parameters to stored ones for derivatives calculation
    this is done only if BFieldOn is false, for which these parameters are used
    */
    if (!fBFieldOn) {
      fTrackPos0[0] = trackParam.fTrackX;
      fTrackPos0[1] = trackParam.fTrackY;
      fTrackPos0[2] = trackParam.fTrackZ;
      fTrackSlope[0] = trackParam.fTrackSlopeX;
      fTrackSlope[1] = trackParam.fTrackSlopeY;
    }
  }

  // second loop to perform alignment
  for (auto itTrackParam(track.begin()); itTrackParam != track.end(); ++itTrackParam) {

    // get track parameters
    if (!&*itTrackParam)
      continue;

    // get cluster
    const Cluster* cluster = itTrackParam->getClusterPtr();
    if (!cluster)
      continue;

    // fill local variables for this position --> one measurement
    FillDetElemData(cluster);
    FillRecPointData(cluster);
    FillTrackParamData(&*itTrackParam);

    // 'inverse' (GlobalToLocal) rotation matrix
    const Double_t* r(fGeoCombiTransInverse.GetRotationMatrix());

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

    // Set local equations
    LocalEquationX();
    LocalEquationY();
  }

  // copy track record
  fMillepede->SetRecordRun(fRunNumber);
  fMillepede->SetRecordWeight(weight);
  fTrackRecord = *fMillepede->GetRecord();

  // save record data
  if (doAlignment) {
    fMillepede->SaveRecordData();
    fMillepede->CloseDataRecStorage();
  }

  // return record
  return &fTrackRecord;
}

//______________________________________________________________________________
void Alignment::ProcessTrack(MillePedeRecord* trackRecord)
{
  LOG(fatal) << __PRETTY_FUNCTION__ << " is disabled";

  /// process track record
  if (!trackRecord)
    return;

  // // make sure record storage is initialized
  if (!fMillepede->GetRecord()) {
    fMillepede->InitDataRecStorage(kFalse);
  }
  // // copy content
  *fMillepede->GetRecord() = *trackRecord;

  // save record
  fMillepede->SaveRecordData();
  // write to local file
  fMillepede->CloseDataRecStorage();

  return;
}

//_____________________________________________________________________
void Alignment::FixAll(UInt_t mask)
{
  /// fix parameters matching mask, for all chambers
  LOG(info) << "Fixing " << GetParameterMaskString(mask).Data() << " for all detector elements";

  // fix all stations
  for (Int_t i = 0; i < fgNDetElem; ++i) {
    if (mask & ParX)
      FixParameter(i, 0);
    if (mask & ParY)
      FixParameter(i, 1);
    if (mask & ParTZ)
      FixParameter(i, 2);
    if (mask & ParZ)
      FixParameter(i, 3);
  }
}

//_____________________________________________________________________
void Alignment::FixChamber(Int_t iCh, UInt_t mask)
{
  /// fix parameters matching mask, for all detector elements in a given chamber, counting from 1

  // check boundaries
  if (iCh < 1 || iCh > 10) {
    LOG(fatal) << "Invalid chamber index " << iCh;
  }

  // get first and last element
  const Int_t iDetElemFirst = fgSNDetElemCh[iCh - 1];
  const Int_t iDetElemLast = fgSNDetElemCh[iCh];
  for (Int_t i = iDetElemFirst; i < iDetElemLast; ++i) {

    LOG(info) << "Fixing " << GetParameterMaskString(mask).Data() << " for detector element " << i;

    if (mask & ParX)
      FixParameter(i, 0);
    if (mask & ParY)
      FixParameter(i, 1);
    if (mask & ParTZ)
      FixParameter(i, 2);
    if (mask & ParZ)
      FixParameter(i, 3);
  }
}

//_____________________________________________________________________
void Alignment::FixDetElem(Int_t iDetElemId, UInt_t mask)
{
  /// fix parameters matching mask, for a given detector element, counting from 0
  const Int_t iDet(GetDetElemNumber(iDetElemId));
  if (mask & ParX)
    FixParameter(iDet, 0);
  if (mask & ParY)
    FixParameter(iDet, 1);
  if (mask & ParTZ)
    FixParameter(iDet, 2);
  if (mask & ParZ)
    FixParameter(iDet, 3);
}

//_____________________________________________________________________
void Alignment::FixHalfSpectrometer(const Bool_t* lChOnOff, UInt_t sidesMask, UInt_t mask)
{

  /// Fix parameters matching mask for all detectors in selected chambers and selected sides of the spectrometer
  for (Int_t i = 0; i < fgNDetElem; ++i) {

    // get chamber matching detector
    const Int_t iCh(GetChamberId(i));
    if (!lChOnOff[iCh - 1])
      continue;

    // get detector element in chamber
    Int_t lDetElemNumber = i - fgSNDetElemCh[iCh - 1];

    // skip detector if its side is off
    // stations 1 and 2
    if (iCh >= 1 && iCh <= 4) {
      if (lDetElemNumber == 0 && !(sidesMask & SideTopRight))
        continue;
      if (lDetElemNumber == 1 && !(sidesMask & SideTopLeft))
        continue;
      if (lDetElemNumber == 2 && !(sidesMask & SideBottomLeft))
        continue;
      if (lDetElemNumber == 3 && !(sidesMask & SideBottomRight))
        continue;
    }

    // station 3
    if (iCh >= 5 && iCh <= 6) {
      if (lDetElemNumber >= 0 && lDetElemNumber <= 4 && !(sidesMask & SideTopRight))
        continue;
      if (lDetElemNumber >= 5 && lDetElemNumber <= 10 && !(sidesMask & SideTopLeft))
        continue;
      if (lDetElemNumber >= 11 && lDetElemNumber <= 13 && !(sidesMask & SideBottomLeft))
        continue;
      if (lDetElemNumber >= 14 && lDetElemNumber <= 17 && !(sidesMask & SideBottomRight))
        continue;
    }

    // stations 4 and 5
    if (iCh >= 7 && iCh <= 10) {
      if (lDetElemNumber >= 0 && lDetElemNumber <= 6 && !(sidesMask & SideTopRight))
        continue;
      if (lDetElemNumber >= 7 && lDetElemNumber <= 13 && !(sidesMask & SideTopLeft))
        continue;
      if (lDetElemNumber >= 14 && lDetElemNumber <= 19 && !(sidesMask & SideBottomLeft))
        continue;
      if (lDetElemNumber >= 20 && lDetElemNumber <= 25 && !(sidesMask & SideBottomRight))
        continue;
    }

    // detector is accepted, fix it
    FixDetElem(i, mask);
  }
}

//______________________________________________________________________
void Alignment::FixParameter(Int_t iPar)
{

  /// fix a given parameter, counting from 0
  if (fInitialized) {
    LOG(fatal) << "Millepede already initialized";
  }

  fGlobalParameterStatus[iPar] = kFixedParId;
}

//_____________________________________________________________________
void Alignment::ReleaseChamber(Int_t iCh, UInt_t mask)
{
  /// release parameters matching mask, for all detector elements in a given chamber, counting from 1

  // check boundaries
  if (iCh < 1 || iCh > 10) {
    LOG(fatal) << "Invalid chamber index " << iCh;
  }

  // get first and last element
  const Int_t iDetElemFirst = fgSNDetElemCh[iCh - 1];
  const Int_t iDetElemLast = fgSNDetElemCh[iCh];
  for (Int_t i = iDetElemFirst; i < iDetElemLast; ++i) {

    LOG(info) << "Releasing " << GetParameterMaskString(mask).Data() << " for detector element " << i;

    if (mask & ParX)
      ReleaseParameter(i, 0);
    if (mask & ParY)
      ReleaseParameter(i, 1);
    if (mask & ParTZ)
      ReleaseParameter(i, 2);
    if (mask & ParZ)
      ReleaseParameter(i, 3);
  }
}

//_____________________________________________________________________
void Alignment::ReleaseDetElem(Int_t iDetElemId, UInt_t mask)
{
  /// release parameters matching mask, for a given detector element, counting from 0
  const Int_t iDet(GetDetElemNumber(iDetElemId));
  if (mask & ParX)
    ReleaseParameter(iDet, 0);
  if (mask & ParY)
    ReleaseParameter(iDet, 1);
  if (mask & ParTZ)
    ReleaseParameter(iDet, 2);
  if (mask & ParZ)
    ReleaseParameter(iDet, 3);
}

//______________________________________________________________________
void Alignment::ReleaseParameter(Int_t iPar)
{

  /// release a given parameter, counting from 0
  if (fInitialized) {
    LOG(fatal) << "Millepede already initialized";
  }

  fGlobalParameterStatus[iPar] = kFreeParId;
}

//_____________________________________________________________________
void Alignment::GroupChamber(Int_t iCh, UInt_t mask)
{
  /// group parameters matching mask for all detector elements in a given chamber, counting from 1
  if (iCh < 1 || iCh > fgNCh) {
    LOG(fatal) << "Invalid chamber index " << iCh;
  }

  const Int_t detElemMin = 100 * iCh;
  const Int_t detElemMax = 100 * iCh + fgNDetElemCh[iCh] - 1;
  GroupDetElems(detElemMin, detElemMax, mask);
}

//_____________________________________________________________________
void Alignment::GroupHalfChamber(Int_t iCh, Int_t iHalf, UInt_t mask)
{
  /// group parameters matching mask for all detector elements in a given tracking module (= half chamber), counting from 0
  if (iCh < 1 || iCh > fgNCh) {
    LOG(fatal) << "Invalid chamber index " << iCh;
  }

  if (iHalf < 0 || iHalf > 1) {
    LOG(fatal) << "Invalid half chamber index " << iHalf;
  }

  const Int_t iHalfCh = 2 * (iCh - 1) + iHalf;
  GroupDetElems(&fgDetElemHalfCh[iHalfCh][0], fgNDetElemHalfCh[iHalfCh], mask);
}

//_____________________________________________________________________
void Alignment::GroupDetElems(Int_t detElemMin, Int_t detElemMax, UInt_t mask)
{
  /// group parameters matching mask for all detector elements between min and max
  // check number of detector elements
  const Int_t nDetElem = detElemMax - detElemMin + 1;
  if (nDetElem < 2) {
    LOG(fatal) << "Requested group of DEs " << detElemMin << "-" << detElemMax << " contains less than 2 DE's";
  }

  // create list
  Int_t* detElemList = new int[nDetElem];
  for (Int_t i = 0; i < nDetElem; ++i) {
    detElemList[i] = detElemMin + i;
  }

  // group
  GroupDetElems(detElemList, nDetElem, mask);
  delete[] detElemList;
}

//_____________________________________________________________________
void Alignment::GroupDetElems(const Int_t* detElemList, Int_t nDetElem, UInt_t mask)
{
  /// group parameters matching mask for all detector elements in list
  if (fInitialized) {
    LOG(fatal) << "Millepede already initialized";
  }

  const Int_t iDeBase(GetDetElemNumber(detElemList[0]));
  for (Int_t i = 0; i < nDetElem; ++i) {
    const Int_t iDeCurrent(GetDetElemNumber(detElemList[i]));
    if (mask & ParX)
      fGlobalParameterStatus[iDeCurrent * fgNParCh + 0] = (i == 0) ? kGroupBaseId : (kGroupBaseId - iDeBase - 1);
    if (mask & ParY)
      fGlobalParameterStatus[iDeCurrent * fgNParCh + 1] = (i == 0) ? kGroupBaseId : (kGroupBaseId - iDeBase - 1);
    if (mask & ParTZ)
      fGlobalParameterStatus[iDeCurrent * fgNParCh + 2] = (i == 0) ? kGroupBaseId : (kGroupBaseId - iDeBase - 1);
    if (mask & ParZ)
      fGlobalParameterStatus[iDeCurrent * fgNParCh + 3] = (i == 0) ? kGroupBaseId : (kGroupBaseId - iDeBase - 1);

    if (i == 0)
      LOG(info) << "Creating new group for detector " << detElemList[i] << " and variable " << GetParameterMaskString(mask).Data();
    else
      LOG(info) << "Adding detector element " << detElemList[i] << " to current group";
  }
}

//______________________________________________________________________
void Alignment::SetChamberNonLinear(Int_t iCh, UInt_t mask)
{
  /// Set parameters matching mask as non linear, for all detector elements in a given chamber, counting from 1
  const Int_t iDetElemFirst = fgSNDetElemCh[iCh - 1];
  const Int_t iDetElemLast = fgSNDetElemCh[iCh];
  for (Int_t i = iDetElemFirst; i < iDetElemLast; ++i) {

    if (mask & ParX)
      SetParameterNonLinear(i, 0);
    if (mask & ParY)
      SetParameterNonLinear(i, 1);
    if (mask & ParTZ)
      SetParameterNonLinear(i, 2);
    if (mask & ParZ)
      SetParameterNonLinear(i, 3);
  }
}

//_____________________________________________________________________
void Alignment::SetDetElemNonLinear(Int_t iDetElemId, UInt_t mask)
{
  /// Set parameters matching mask as non linear, for a given detector element, counting from 0
  const Int_t iDet(GetDetElemNumber(iDetElemId));
  if (mask & ParX)
    SetParameterNonLinear(iDet, 0);
  if (mask & ParY)
    SetParameterNonLinear(iDet, 1);
  if (mask & ParTZ)
    SetParameterNonLinear(iDet, 2);
  if (mask & ParZ)
    SetParameterNonLinear(iDet, 3);
}

//______________________________________________________________________
void Alignment::SetParameterNonLinear(Int_t iPar)
{
  /// Set nonlinear flag for parameter iPar
  if (!fInitialized) {
    LOG(fatal) << "Millepede not initialized";
  }

  fMillepede->SetNonLinear(iPar);
  LOG(info) << "Parameter " << iPar << " set to non linear ";
}

//______________________________________________________________________
void Alignment::AddConstraints(const Bool_t* lChOnOff, UInt_t mask)
{
  /// Add constraint equations for selected chambers and degrees of freedom

  Array fConstraintX;
  Array fConstraintY;
  Array fConstraintTZ;
  Array fConstraintZ;

  for (Int_t i = 0; i < fgNDetElem; ++i) {

    // get chamber matching detector
    const Int_t iCh(GetChamberId(i));
    if (lChOnOff[iCh - 1]) {

      if (mask & ParX)
        fConstraintX.values[i * fgNParCh + 0] = 1.0;
      if (mask & ParY)
        fConstraintY.values[i * fgNParCh + 1] = 1.0;
      if (mask & ParTZ)
        fConstraintTZ.values[i * fgNParCh + 2] = 1.0;
      if (mask & ParZ)
        fConstraintTZ.values[i * fgNParCh + 3] = 1.0;
    }
  }

  if (mask & ParX)
    AddConstraint(fConstraintX.values, 0.0);
  if (mask & ParY)
    AddConstraint(fConstraintY.values, 0.0);
  if (mask & ParTZ)
    AddConstraint(fConstraintTZ.values, 0.0);
  if (mask & ParZ)
    AddConstraint(fConstraintZ.values, 0.0);
}

//______________________________________________________________________
void Alignment::AddConstraints(const Bool_t* lChOnOff, const Bool_t* lVarXYT, UInt_t sidesMask)
{
  /*
  questions:
  - is there not redundancy/inconsistency between lDetTLBR and lSpecLROnOff ? shouldn't we use only lDetTLBR ?
  - why is weight ignored for ConstrainT and ConstrainB
  - why is there no constrain on z
  */

  /// Add constraint equations for selected chambers, degrees of freedom and detector half
  Double_t lMeanY = 0.;
  Double_t lSigmaY = 0.;
  Double_t lMeanZ = 0.;
  Double_t lSigmaZ = 0.;
  Int_t lNDetElem = 0;

  for (Int_t i = 0; i < fgNDetElem; ++i) {

    // get chamber matching detector
    const Int_t iCh(GetChamberId(i));

    // skip detector if chamber is off
    if (lChOnOff[iCh - 1])
      continue;

    // get detector element id from detector element number
    const Int_t lDetElemNumber = i - fgSNDetElemCh[iCh - 1];
    const Int_t lDetElemId = iCh * 100 + lDetElemNumber;

    // skip detector if its side is off
    // stations 1 and 2
    if (iCh >= 1 && iCh <= 4) {
      if (lDetElemNumber == 0 && !(sidesMask & SideTopRight))
        continue;
      if (lDetElemNumber == 1 && !(sidesMask & SideTopLeft))
        continue;
      if (lDetElemNumber == 2 && !(sidesMask & SideBottomLeft))
        continue;
      if (lDetElemNumber == 3 && !(sidesMask & SideBottomRight))
        continue;
    }

    // station 3
    if (iCh >= 5 && iCh <= 6) {
      if (lDetElemNumber >= 0 && lDetElemNumber <= 4 && !(sidesMask & SideTopRight))
        continue;
      if (lDetElemNumber >= 5 && lDetElemNumber <= 10 && !(sidesMask & SideTopLeft))
        continue;
      if (lDetElemNumber >= 11 && lDetElemNumber <= 13 && !(sidesMask & SideBottomLeft))
        continue;
      if (lDetElemNumber >= 14 && lDetElemNumber <= 17 && !(sidesMask & SideBottomRight))
        continue;
    }

    // stations 4 and 5
    if (iCh >= 7 && iCh <= 10) {
      if (lDetElemNumber >= 0 && lDetElemNumber <= 6 && !(sidesMask & SideTopRight))
        continue;
      if (lDetElemNumber >= 7 && lDetElemNumber <= 13 && !(sidesMask & SideTopLeft))
        continue;
      if (lDetElemNumber >= 14 && lDetElemNumber <= 19 && !(sidesMask & SideBottomLeft))
        continue;
      if (lDetElemNumber >= 20 && lDetElemNumber <= 25 && !(sidesMask & SideBottomRight))
        continue;
    }

    // get global x, y and z position
    Double_t lDetElemGloX = 0.;
    Double_t lDetElemGloY = 0.;
    Double_t lDetElemGloZ = 0.;

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

  // fill Bool_t sides array based on masks, for convenience
  Bool_t lDetTLBR[4];
  lDetTLBR[0] = sidesMask & SideTop;
  lDetTLBR[1] = sidesMask & SideLeft;
  lDetTLBR[2] = sidesMask & SideBottom;
  lDetTLBR[3] = sidesMask & SideRight;

  for (Int_t i = 0; i < fgNDetElem; ++i) {

    // get chamber matching detector
    const Int_t iCh(GetChamberId(i));

    // skip detector if chamber is off
    if (!lChOnOff[iCh - 1])
      continue;

    // get detector element id from detector element number
    const Int_t lDetElemNumber = i - fgSNDetElemCh[iCh - 1];
    const Int_t lDetElemId = iCh * 100 + lDetElemNumber;

    // get global x, y and z position
    Double_t lDetElemGloX = 0.;
    Double_t lDetElemGloY = 0.;
    Double_t lDetElemGloZ = 0.;

    auto fTransform = fTransformCreator(lDetElemId);
    o2::math_utils::Point3D<double> SlatPos{0.0, 0.0, 0.0};
    o2::math_utils::Point3D<double> GlobalPos;

    fTransform.LocalToMaster(SlatPos, GlobalPos);
    lDetElemGloX = GlobalPos.x();
    lDetElemGloY = GlobalPos.y();
    lDetElemGloZ = GlobalPos.z();
    // fTransform->Local2Global(lDetElemId, 0, 0, 0, lDetElemGloX, lDetElemGloY, lDetElemGloZ);

    // loop over sides
    for (Int_t iSide = 0; iSide < 4; iSide++) {

      // skip if side is not selected
      if (!lDetTLBR[iSide])
        continue;

      // skip detector if it is not in the selected side
      // stations 1 and 2
      if (iCh >= 1 && iCh <= 4) {
        if (lDetElemNumber == 0 && !(iSide == 0 || iSide == 3))
          continue; // top-right
        if (lDetElemNumber == 1 && !(iSide == 0 || iSide == 1))
          continue; // top-left
        if (lDetElemNumber == 2 && !(iSide == 2 || iSide == 1))
          continue; // bottom-left
        if (lDetElemNumber == 3 && !(iSide == 2 || iSide == 3))
          continue; // bottom-right
      }

      // station 3
      if (iCh >= 5 && iCh <= 6) {
        if (lDetElemNumber >= 0 && lDetElemNumber <= 4 && !(iSide == 0 || iSide == 3))
          continue; // top-right
        if (lDetElemNumber >= 5 && lDetElemNumber <= 9 && !(iSide == 0 || iSide == 1))
          continue; // top-left
        if (lDetElemNumber >= 10 && lDetElemNumber <= 13 && !(iSide == 2 || iSide == 1))
          continue; // bottom-left
        if (lDetElemNumber >= 14 && lDetElemNumber <= 17 && !(iSide == 2 || iSide == 3))
          continue; // bottom-right
      }

      // stations 4 and 5
      if (iCh >= 7 && iCh <= 10) {
        if (lDetElemNumber >= 0 && lDetElemNumber <= 6 && !(iSide == 0 || iSide == 3))
          continue; // top-right
        if (lDetElemNumber >= 7 && lDetElemNumber <= 13 && !(iSide == 0 || iSide == 1))
          continue; // top-left
        if (lDetElemNumber >= 14 && lDetElemNumber <= 19 && !(iSide == 2 || iSide == 1))
          continue; // bottom-left
        if (lDetElemNumber >= 20 && lDetElemNumber <= 25 && !(iSide == 2 || iSide == 3))
          continue; // bottom-right
      }

      // constrain x
      if (lVarXYT[0])
        fConstraintX[iSide].values[i * fgNParCh + 0] = 1;

      // constrain y
      if (lVarXYT[1])
        fConstraintY[iSide].values[i * fgNParCh + 1] = 1;

      // constrain phi (rotation around z)
      if (lVarXYT[2])
        fConstraintP[iSide].values[i * fgNParCh + 2] = 1;

      // x-z shearing
      if (lVarXYT[3])
        fConstraintXZ[iSide].values[i * fgNParCh + 0] = (lDetElemGloZ - lMeanZ) / lSigmaZ;

      // y-z shearing
      if (lVarXYT[4])
        fConstraintYZ[iSide].values[i * fgNParCh + 1] = (lDetElemGloZ - lMeanZ) / lSigmaZ;

      // phi-z shearing
      if (lVarXYT[5])
        fConstraintPZ[iSide].values[i * fgNParCh + 2] = (lDetElemGloZ - lMeanZ) / lSigmaZ;

      // x-y shearing
      if (lVarXYT[6])
        fConstraintXY[iSide].values[i * fgNParCh + 0] = (lDetElemGloY - lMeanY) / lSigmaY;

      // y-y shearing
      if (lVarXYT[7])
        fConstraintYY[iSide].values[i * fgNParCh + 1] = (lDetElemGloY - lMeanY) / lSigmaY;

      // phi-y shearing
      if (lVarXYT[8])
        fConstraintPY[iSide].values[i * fgNParCh + 2] = (lDetElemGloY - lMeanY) / lSigmaY;
    }
  }

  // pass constraints to millepede
  for (Int_t iSide = 0; iSide < 4; iSide++) {
    // skip if side is not selected
    if (!lDetTLBR[iSide])
      continue;

    if (lVarXYT[0])
      AddConstraint(fConstraintX[iSide].values, 0.0);
    if (lVarXYT[1])
      AddConstraint(fConstraintY[iSide].values, 0.0);
    if (lVarXYT[2])
      AddConstraint(fConstraintP[iSide].values, 0.0);
    if (lVarXYT[3])
      AddConstraint(fConstraintXZ[iSide].values, 0.0);
    if (lVarXYT[4])
      AddConstraint(fConstraintYZ[iSide].values, 0.0);
    if (lVarXYT[5])
      AddConstraint(fConstraintPZ[iSide].values, 0.0);
    if (lVarXYT[6])
      AddConstraint(fConstraintXY[iSide].values, 0.0);
    if (lVarXYT[7])
      AddConstraint(fConstraintYY[iSide].values, 0.0);
    if (lVarXYT[8])
      AddConstraint(fConstraintPY[iSide].values, 0.0);
  }
}

//______________________________________________________________________
void Alignment::InitGlobalParameters(Double_t* par)
{
  /// Initialize global parameters with par array
  if (!fInitialized) {
    LOG(fatal) << "Millepede is not initialized";
  }

  fMillepede->SetGlobalParameters(par);
}

//______________________________________________________________________
void Alignment::SetAllowedVariation(Int_t iPar, Double_t value)
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
void Alignment::SetSigmaXY(Double_t sigmaX, Double_t sigmaY)
{

  /// Set expected measurement resolution
  fSigma[0] = sigmaX;
  fSigma[1] = sigmaY;

  // print
  for (Int_t i = 0; i < 2; ++i) {
    LOG(info) << "fSigma[" << i << "] =" << fSigma[i];
  }
}

//_____________________________________________________
void Alignment::GlobalFit(Double_t* parameters, Double_t* errors, Double_t* pulls)
{

  /// Call global fit; Global parameters are stored in parameters
  fMillepede->GlobalFit(parameters, errors, pulls);

  LOG(info) << "Done fitting global parameters";
  for (int iDet = 0; iDet < fgNDetElem; ++iDet) {
    LOG(info) << iDet << " " << parameters[iDet * fgNParCh + 0] << " " << parameters[iDet * fgNParCh + 1] << " " << parameters[iDet * fgNParCh + 3] << " " << parameters[iDet * fgNParCh + 2];
  }
}

//_____________________________________________________
void Alignment::PrintGlobalParameters() const
{
  fMillepede->PrintGlobalParameters();
}

//_____________________________________________________
Double_t Alignment::GetParError(Int_t iPar) const
{
  return fMillepede->GetParError(iPar);
}

// //______________________________________________________________________
// AliMUONGeometryTransformer* Alignment::ReAlign(
//   const AliMUONGeometryTransformer* transformer,
//   const double* misAlignments, Bool_t)
// {

//   /// Returns a new AliMUONGeometryTransformer with the found misalignments
//   /// applied.

//   // Takes the internal geometry module transformers, copies them
//   // and gets the Detection Elements from them.
//   // Takes misalignment parameters and applies these
//   // to the local transform of the Detection Element
//   // Obtains the global transform by multiplying the module transformer
//   // transformation with the local transformation
//   // Applies the global transform to a new detection element
//   // Adds the new detection element to a new module transformer
//   // Adds the new module transformer to a new geometry transformer
//   // Returns the new geometry transformer

//   Double_t lModuleMisAlignment[fgNParCh] = {0};
//   Double_t lDetElemMisAlignment[fgNParCh] = {0};
//   const TClonesArray* oldMisAlignArray(transformer->GetMisAlignmentData());

//   AliMUONGeometryTransformer* newGeometryTransformer = new AliMUONGeometryTransformer();
//   for (Int_t iMt = 0; iMt < transformer->GetNofModuleTransformers(); ++iMt) {

//     // module transformers
//     const AliMUONGeometryModuleTransformer* kModuleTransformer = transformer->GetModuleTransformer(iMt, kTRUE);

//     AliMUONGeometryModuleTransformer* newModuleTransformer = new AliMUONGeometryModuleTransformer(iMt);
//     newGeometryTransformer->AddModuleTransformer(newModuleTransformer);

//     // get transformation
//     TGeoHMatrix deltaModuleTransform(DeltaTransform(lModuleMisAlignment));

//     // update module
//     TGeoHMatrix moduleTransform(*kModuleTransformer->GetTransformation());
//     TGeoHMatrix newModuleTransform(AliMUONGeometryBuilder::Multiply(deltaModuleTransform, moduleTransform));
//     newModuleTransformer->SetTransformation(newModuleTransform);

//     // Get matching old alignment and update current matrix accordingly
//     if (oldMisAlignArray) {

//       const AliAlignObjMatrix* oldAlignObj(0);
//       const Int_t moduleId(kModuleTransformer->GetModuleId());
//       const Int_t volId = AliGeomManager::LayerToVolUID(AliGeomManager::kMUON, moduleId);
//       for (Int_t pos = 0; pos < oldMisAlignArray->GetEntriesFast(); ++pos) {

//         const AliAlignObjMatrix* localAlignObj(dynamic_cast<const AliAlignObjMatrix*>(oldMisAlignArray->At(pos)));
//         if (localAlignObj && localAlignObj->GetVolUID() == volId) {
//           oldAlignObj = localAlignObj;
//           break;
//         }
//       }

//       // multiply
//       if (oldAlignObj) {

//         TGeoHMatrix oldMatrix;
//         oldAlignObj->GetMatrix(oldMatrix);
//         deltaModuleTransform.Multiply(&oldMatrix);
//       }
//     }

//     // Create module mis alignment matrix
//     newGeometryTransformer->AddMisAlignModule(kModuleTransformer->GetModuleId(), deltaModuleTransform);

//     AliMpExMap* detElements = kModuleTransformer->GetDetElementStore();

//     TIter next(detElements->CreateIterator());
//     AliMUONGeometryDetElement* detElement;
//     Int_t iDe(-1);
//     while ((detElement = static_cast<AliMUONGeometryDetElement*>(next()))) {
//       ++iDe;
//       // make a new detection element
//       AliMUONGeometryDetElement* newDetElement = new AliMUONGeometryDetElement(detElement->GetId(), detElement->GetVolumePath());
//       TString lDetElemName(detElement->GetDEName());
//       lDetElemName.ReplaceAll("DE", "");

//       // store detector element id and number
//       const Int_t iDetElemId = lDetElemName.Atoi();
//       if (DetElemIsValid(iDetElemId)) {

//         const Int_t iDetElemNumber(GetDetElemNumber(iDetElemId));

//         for (int i = 0; i < fgNParCh; ++i) {
//           lDetElemMisAlignment[i] = 0.0;
//           if (iMt < fgNTrkMod) {
//             lDetElemMisAlignment[i] = misAlignments[iDetElemNumber * fgNParCh + i];
//           }
//         }

//         // get transformation
//         TGeoHMatrix deltaGlobalTransform(DeltaTransform(lDetElemMisAlignment));

//         // update module
//         TGeoHMatrix globalTransform(*detElement->GetGlobalTransformation());
//         TGeoHMatrix newGlobalTransform(AliMUONGeometryBuilder::Multiply(deltaGlobalTransform, globalTransform));
//         newDetElement->SetGlobalTransformation(newGlobalTransform);
//         newModuleTransformer->GetDetElementStore()->Add(newDetElement->GetId(), newDetElement);

//         // Get matching old alignment and update current matrix accordingly
//         if (oldMisAlignArray) {

//           const AliAlignObjMatrix* oldAlignObj(0);
//           const int detElemId(detElement->GetId());
//           const Int_t volId = AliGeomManager::LayerToVolUID(AliGeomManager::kMUON, detElemId);
//           for (Int_t pos = 0; pos < oldMisAlignArray->GetEntriesFast(); ++pos) {

//             const AliAlignObjMatrix* localAlignObj(dynamic_cast<const AliAlignObjMatrix*>(oldMisAlignArray->At(pos)));
//             if (localAlignObj && localAlignObj->GetVolUID() == volId) {
//               oldAlignObj = localAlignObj;
//               break;
//             }
//           }

//           // multiply
//           if (oldAlignObj) {

//             TGeoHMatrix oldMatrix;
//             oldAlignObj->GetMatrix(oldMatrix);
//             deltaGlobalTransform.Multiply(&oldMatrix);
//           }
//         }

//         // Create misalignment matrix
//         newGeometryTransformer->AddMisAlignDetElement(detElement->GetId(), deltaGlobalTransform);

//       } else {

//         // "invalid" detector elements come from MTR and are left unchanged
//         Aliinfo(Form("Keeping detElement %i unchanged", iDetElemId));

//         // update module
//         TGeoHMatrix globalTransform(*detElement->GetGlobalTransformation());
//         newDetElement->SetGlobalTransformation(globalTransform);
//         newModuleTransformer->GetDetElementStore()->Add(newDetElement->GetId(), newDetElement);

//         // Get matching old alignment and update current matrix accordingly
//         if (oldMisAlignArray) {

//           const AliAlignObjMatrix* oldAlignObj(0);
//           const int detElemId(detElement->GetId());
//           const Int_t volId = AliGeomManager::LayerToVolUID(AliGeomManager::kMUON, detElemId);
//           for (Int_t pos = 0; pos < oldMisAlignArray->GetEntriesFast(); ++pos) {

//             const AliAlignObjMatrix* localAlignObj(dynamic_cast<const AliAlignObjMatrix*>(oldMisAlignArray->At(pos)));
//             if (localAlignObj && localAlignObj->GetVolUID() == volId) {
//               oldAlignObj = localAlignObj;
//               break;
//             }
//           }

//           // multiply
//           if (oldAlignObj) {

//             TGeoHMatrix oldMatrix;
//             oldAlignObj->GetMatrix(oldMatrix);
//             newGeometryTransformer->AddMisAlignDetElement(detElement->GetId(), oldMatrix);
//           }
//         }
//       }
//     }

//     newGeometryTransformer->AddModuleTransformer(newModuleTransformer);
//   }

//   return newGeometryTransformer;
// }

//______________________________________________________________________
void Alignment::SetAlignmentResolution(const TClonesArray* misAlignArray, Int_t rChId, Double_t chResX, Double_t chResY, Double_t deResX, Double_t deResY)
{

  /// Set alignment resolution to misalign objects to be stored in CDB
  /// if rChId is > 0 set parameters for this chamber only, counting from 1
  TMatrixDSym mChCorrMatrix(6);
  mChCorrMatrix[0][0] = chResX * chResX;
  mChCorrMatrix[1][1] = chResY * chResY;

  TMatrixDSym mDECorrMatrix(6);
  mDECorrMatrix[0][0] = deResX * deResX;
  mDECorrMatrix[1][1] = deResY * deResY;

  o2::detectors::AlignParam* alignMat = 0x0;

  for (Int_t chId = 0; chId <= 9; ++chId) {

    // skip chamber if selection is valid, and does not match
    if (rChId > 0 && chId + 1 != rChId)
      continue;

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
        // if (volName.Contains("GM")){
        //   alignMat->SetCorrMatrix(mChCorrMatrix);
        // }else if (volName.Contains("DE")){
        //   alignMat->SetCorrMatrix(mDECorrMatrix);
        // }
      }
    }
  }
}

//_____________________________________________________
LocalTrackParam Alignment::RefitStraightTrack(Track& track, Double_t z0) const
{

  // initialize matrices
  TMatrixD AtGASum(4, 4);
  AtGASum.Zero();

  TMatrixD AtGMSum(4, 1);
  AtGMSum.Zero();

  // loop over clusters
  for (auto itTrackParam(track.begin()); itTrackParam != track.end(); ++itTrackParam) {

    // get track parameters
    if (!&*itTrackParam)
      continue;

    // get cluster
    const Cluster* cluster = itTrackParam->getClusterPtr();
    if (!cluster)
      continue;

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

  //   // TODO: compare with initial track parameters
  //   Aliinfo( Form( "x: %.3f vs %.3f", fTrackPos0[0], X(0,0) ) );
  //   Aliinfo( Form( "y: %.3f vs %.3f", fTrackPos0[1], X(1,0) ) );
  //   Aliinfo( Form( "dxdz: %.6g vs %.6g", fTrackSlope0[0], X(2,0) ) );
  //   Aliinfo( Form( "dydz: %.6g vs %.6g\n", fTrackSlope0[1], X(3,0) ) );

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
void Alignment::FillDetElemData(const Cluster* cluster)
{
  // LOG(fatal) << __PRETTY_FUNCTION__ << " is disabled";
  LOG(info) << __PRETTY_FUNCTION__ << " is enabled";

  /// Get information of current detection element
  // get detector element number from Alice ID
  const Int_t detElemId = cluster->getDEId();
  fDetElemNumber = GetDetElemNumber(detElemId);

  // get detector element
  // const AliMUONGeometryDetElement detElement(detElemId);
  auto fTransform = fTransformCreator(detElemId);
  /*
  get the global transformation matrix and store its inverse, in order to manually perform
  the global to Local transformations needed to calculate the derivatives
  */
  // fTransform = fTransform.Inverse();
  // fTransform.GetTransformMatrix(fGeoCombiTransInverse);
}

//______________________________________________________________________
void Alignment::FillRecPointData(const Cluster* cluster)
{

  /// Get information of current cluster
  fClustPos[0] = cluster->getX();
  fClustPos[1] = cluster->getY();
  fClustPos[2] = cluster->getZ();
}

//______________________________________________________________________
void Alignment::FillTrackParamData(const TrackParam* trackParam)
{

  /// Get information of current track at current cluster
  fTrackPos[0] = trackParam->getNonBendingCoor();
  fTrackPos[1] = trackParam->getBendingCoor();
  fTrackPos[2] = trackParam->getZ();
  fTrackSlope[0] = trackParam->getNonBendingSlope();
  fTrackSlope[1] = trackParam->getBendingSlope();
}

//______________________________________________________________________
void Alignment::LocalEquationX(void)
{
  /// local equation along X

  // 'inverse' (GlobalToLocal) rotation matrix
  const Double_t* r(fGeoCombiTransInverse.GetRotationMatrix());

  // local derivatives
  SetLocalDerivative(0, r[0]);
  SetLocalDerivative(1, r[0] * (fTrackPos[2] - fTrackPos0[2]));
  SetLocalDerivative(2, r[1]);
  SetLocalDerivative(3, r[1] * (fTrackPos[2] - fTrackPos0[2]));

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
    const Double_t trackPosX = fTrackPos0[0] + fTrackSlope0[0] * (fTrackPos[2] - fTrackPos0[2]);
    const Double_t trackPosY = fTrackPos0[1] + fTrackSlope0[1] * (fTrackPos[2] - fTrackPos0[2]);

    // use properly extrapolated position for derivatives vs 'delta_phi_z'
    SetGlobalDerivative(fDetElemNumber * fgNParCh + 2, -r[1] * trackPosX + r[0] * trackPosY);

    // use slopes at origin for derivatives vs 'delta_z'
    SetGlobalDerivative(fDetElemNumber * fgNParCh + 3, r[0] * fTrackSlope0[0] + r[1] * fTrackSlope0[1]);
  }

  // store local equation
  fMillepede->SetLocalEquation(fGlobalDerivatives, fLocalDerivatives, fMeas[0], fSigma[0]);
}

//______________________________________________________________________
void Alignment::LocalEquationY(void)
{
  /// local equation along Y

  // 'inverse' (GlobalToLocal) rotation matrix
  const Double_t* r(fGeoCombiTransInverse.GetRotationMatrix());

  // store local derivatives
  SetLocalDerivative(0, r[3]);
  SetLocalDerivative(1, r[3] * (fTrackPos[2] - fTrackPos0[2]));
  SetLocalDerivative(2, r[4]);
  SetLocalDerivative(3, r[4] * (fTrackPos[2] - fTrackPos0[2]));

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
    const Double_t trackPosX = fTrackPos0[0] + fTrackSlope0[0] * (fTrackPos[2] - fTrackPos0[2]);
    const Double_t trackPosY = fTrackPos0[1] + fTrackSlope0[1] * (fTrackPos[2] - fTrackPos0[2]);

    // use properly extrapolated position for derivatives vs 'delta_phi'
    SetGlobalDerivative(fDetElemNumber * fgNParCh + 2, -r[4] * trackPosX + r[3] * trackPosY);

    // use slopes at origin for derivatives vs 'delta_z'
    SetGlobalDerivative(fDetElemNumber * fgNParCh + 3, r[3] * fTrackSlope0[0] + r[4] * fTrackSlope0[1]);
  }

  // store local equation
  fMillepede->SetLocalEquation(fGlobalDerivatives, fLocalDerivatives, fMeas[1], fSigma[1]);
}

//_________________________________________________________________________
TGeoCombiTrans Alignment::DeltaTransform(const double* lMisAlignment) const
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

//______________________________________________________________________
void Alignment::AddConstraint(Double_t* par, Double_t value)
{
  /// Constrain equation defined by par to value
  if (!fInitialized) {
    LOG(fatal) << "Millepede is not initialized";
  }

  fMillepede->SetGlobalConstraint(par, value);
}

//______________________________________________________________________
Bool_t Alignment::DetElemIsValid(Int_t iDetElemId) const
{
  /// return true if given detector element is valid (and belongs to muon tracker)
  const Int_t iCh = iDetElemId / 100;
  const Int_t iDet = iDetElemId % 100;
  return (iCh > 0 && iCh <= fgNCh && iDet < fgNDetElemCh[iCh - 1]);
}

//______________________________________________________________________
Int_t Alignment::GetDetElemNumber(Int_t iDetElemId) const
{
  /// get det element number from ID
  // get chamber and element number in chamber
  const Int_t iCh = iDetElemId / 100;
  const Int_t iDet = iDetElemId % 100;

  // make sure detector index is valid
  if (!(iCh > 0 && iCh <= fgNCh && iDet < fgNDetElemCh[iCh - 1])) {
    LOG(fatal) << "Invalid detector element id: " << iDetElemId;
  }

  // add number of detectors up to this chamber
  return iDet + fgSNDetElemCh[iCh - 1];
}

//______________________________________________________________________
Int_t Alignment::GetChamberId(Int_t iDetElemNumber) const
{
  /// get chamber (counting from 1) matching a given detector element id
  Int_t iCh(0);
  for (iCh = 0; iCh < fgNCh; iCh++) {
    if (iDetElemNumber < fgSNDetElemCh[iCh])
      break;
  }

  return iCh;
}

//______________________________________________________________________
TString Alignment::GetParameterMaskString(UInt_t mask) const
{
  TString out;
  if (mask & ParX)
    out += "X";
  if (mask & ParY)
    out += "Y";
  if (mask & ParZ)
    out += "Z";
  if (mask & ParTZ)
    out += "T";
  return out;
}

//______________________________________________________________________
TString Alignment::GetSidesMaskString(UInt_t mask) const
{
  TString out;
  if (mask & SideTop)
    out += "T";
  if (mask & SideLeft)
    out += "L";
  if (mask & SideBottom)
    out += "B";
  if (mask & SideRight)
    out += "R";
  return out;
}

} // namespace mch
} // namespace o2