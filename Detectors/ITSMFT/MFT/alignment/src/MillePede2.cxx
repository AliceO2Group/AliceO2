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

/// @file MillePede2.cxx

#include "MFTAlignment/MillePede2.h"
#include "Framework/Logger.h"
#include <TStopwatch.h>
#include <TFile.h>
#include <TChain.h>
#include <TMath.h>
#include <TVectorD.h>
#include <TArrayL.h>
#include <TArrayF.h>
#include <TSystem.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <fstream>

//#define _DUMP_EQ_BEFORE_
//#define _DUMP_EQ_AFTER_

//#define _DUMPEQ_BEFORE_
//#define _DUMPEQ_AFTER_

using std::ifstream;
using namespace o2::mft;

ClassImp(MillePede2);

bool MillePede2::fgInvChol = true;                   // Invert global matrix with Cholesky solver
bool MillePede2::fgWeightSigma = true;               // weight local constraint by module statistics
bool MillePede2::fgIsMatGloSparse = false;           // use faster dense matrix by default
int MillePede2::fgMinResCondType = 1;                // Jacoby preconditioner by default
double MillePede2::fgMinResTol = 1.e-11;             // default tolerance
int MillePede2::fgMinResMaxIter = 10000;             // default max number of iterations
int MillePede2::fgIterSol = MinResSolve::kSolMinRes; // default iterative solver
int MillePede2::fgNKrylovV = 240;                    // default number of Krylov vectors to keep

//_____________________________________________________________________________
MillePede2::MillePede2()
  : fNLocPar(0),
    fNGloPar(0),
    fNGloParIni(0),
    fNGloSize(0),
    fNLocEquations(0),
    fIter(0),
    fMaxIter(10),
    fNStdDev(3),
    fNGloConstraints(0),
    fNLagrangeConstraints(0),
    fNLocFits(0),
    fNLocFitsRejected(0),
    fNGloFix(0),
    fGloSolveStatus(kFailed),
    fChi2CutFactor(1.),
    fChi2CutRef(1.),
    fResCutInit(100.),
    fResCut(100.),
    fMinPntValid(1),
    fNGroupsSet(0),
    fMatCLoc(nullptr),
    fMatCGlo(nullptr),
    fMatCGloLoc(nullptr),
    fRecChi2File(nullptr),
    fRecChi2FName("chi2_records.root"),
    fRecChi2TreeName("chi2Records"),
    fTreeChi2(nullptr),
    fSumChi2(0.),
    fIsChi2BelowLimit(true),
    fRecNDoF(0),
    fRecord(nullptr),
    fCurrRecDataID(0),
    fCurrRecConstrID(0),
    fLocFitAdd(true),
    fUseRecordWeight(true),
    fMinRecordLength(1),
    fSelFirst(1),
    fSelLast(-1),
    fRejRunList(nullptr),
    fAccRunList(nullptr),
    fAccRunListWgh(nullptr),
    fRunWgh(1),
    fRecordWriter(nullptr),
    fConstraintsRecWriter(nullptr),
    fRecordReader(nullptr),
    fConstraintsRecReader(nullptr)
{
  fWghScl[0] = fWghScl[1] = -1;
  LOGF(info, "MillePede2 instantiated");
}

//_____________________________________________________________________________
MillePede2::MillePede2(const MillePede2& src)
  : fNLocPar(0),
    fNGloPar(0),
    fNGloParIni(0),
    fNGloSize(0),
    fNLocEquations(0),
    fIter(0),
    fMaxIter(10),
    fNStdDev(3),
    fNGloConstraints(0),
    fNLagrangeConstraints(0),
    fNLocFits(0),
    fNLocFitsRejected(0),
    fNGloFix(0),
    fGloSolveStatus(kFailed),
    fChi2CutFactor(1.),
    fChi2CutRef(1.),
    fResCutInit(100.),
    fResCut(100.),
    fMinPntValid(1),
    fNGroupsSet(0),
    fMatCLoc(nullptr),
    fMatCGlo(nullptr),
    fMatCGloLoc(nullptr),
    fRecChi2File(nullptr),
    fRecChi2FName("chi2_records.root"),
    fRecChi2TreeName("chi2Records"),
    fTreeChi2(nullptr),
    fSumChi2(0.),
    fIsChi2BelowLimit(true),
    fRecNDoF(0),
    fRecord(nullptr),
    fCurrRecDataID(0),
    fCurrRecConstrID(0),
    fLocFitAdd(true),
    fUseRecordWeight(true),
    fMinRecordLength(1),
    fSelFirst(1),
    fSelLast(-1),
    fRejRunList(nullptr),
    fAccRunList(nullptr),
    fAccRunListWgh(nullptr),
    fRunWgh(1),
    fRecordWriter(nullptr),
    fConstraintsRecWriter(nullptr),
    fRecordReader(nullptr),
    fConstraintsRecReader(nullptr)
{
  fWghScl[0] = src.fWghScl[0];
  fWghScl[1] = src.fWghScl[1];
  printf("Dummy\n");
}

//_____________________________________________________________________________
MillePede2::~MillePede2()
{
  if (fMatCLoc) {
    delete fMatCLoc;
  }
  if (fMatCGlo) {
    delete fMatCGlo;
  }
  if (fMatCGloLoc) {
    delete fMatCGloLoc;
  }

  if (fRejRunList) {
    delete fRejRunList;
  }
  if (fAccRunList) {
    delete fAccRunList;
  }
  if (fAccRunListWgh) {
    delete fAccRunListWgh;
  }
  if (fRecChi2File) {
    fRecChi2File->Close();
    LOG(info) << "MillePede2 - Closed file "
              << GetRecChi2FName();
    delete fRecChi2File; // this should automatically delete the TTree that belongs to the file
  }
  LOGF(debug, "MillePede2 destroyed");
}

//_____________________________________________________________________________
int MillePede2::InitMille(int nGlo, const int nLoc,
                          const int lNStdDev, const double lResCut,
                          const double lResCutInit, const std::vector<int>& regroup)
{
  fNGloParIni = nGlo;
  if (regroup.size()) { // regrouping is requested
    fkReGroup = regroup;
    int ng = 0; // recalculate N globals
    int maxPID = -1;
    for (int i = 0; i < nGlo; i++) {
      if (regroup[i] >= 0) {
        ng++;
        if (regroup[i] > maxPID) {
          maxPID = regroup[i];
        }
      }
    }
    maxPID++;
    LOGF(info,
         "MillePede2 - Regrouping is requested: from %d raw to %d formal globals grouped to %d real globals",
         nGlo, ng, maxPID);
    nGlo = maxPID;
  }
  if (nLoc > 0) {
    fNLocPar = nLoc;
  }
  if (nGlo > 0) {
    fNGloPar = nGlo;
  }
  if (lResCutInit > 0) {
    fResCutInit = lResCutInit;
  }
  if (lResCut > 0) {
    fResCut = lResCut;
  }
  if (lNStdDev > 0) {
    fNStdDev = lNStdDev;
  }
  LOGF(info, "MillePede2 - NLoc: %d NGlo: %d", fNLocPar, fNGloPar);

  fNGloSize = fNGloPar;

  if (fgIsMatGloSparse) {
    fMatCGlo = new MatrixSparse(fNGloPar);
    fMatCGlo->SetSymmetric(true);
  } else {
    fMatCGlo = new SymMatrix(fNGloPar);
  }

  fFillIndex = std::vector<int>(fNGloPar);
  fFillValue = std::vector<double>(fNGloPar);

  fMatCLoc = new SymMatrix(fNLocPar);
  fMatCGloLoc = new RectMatrix(fNGloPar, fNLocPar);

  fParamGrID = std::vector<int>(fNGloPar);
  fProcPnt = std::vector<int>(fNGloPar);
  fVecBLoc = std::vector<double>(fNGloPar);
  fDiagCGlo = std::vector<double>(fNGloPar);

  fInitPar = std::vector<double>(fNGloPar);
  fDeltaPar = std::vector<double>(fNGloPar);
  fSigmaPar = std::vector<double>(fNGloPar);
  fIsLinear = std::vector<bool>(fNGloPar);

  fGlo2CGlo = std::vector<int>(fNGloPar);
  fCGlo2Glo = std::vector<int>(fNGloPar);

  for (int i = fNGloPar; i--;) {
    fGlo2CGlo[i] = fCGlo2Glo[i] = -1;
    fIsLinear[i] = true;
    fParamGrID[i] = -1;
  }

  fWghScl[0] = -1;
  fWghScl[1] = -1;
  return 1;
}

//_____________________________________________________________________________
bool MillePede2::InitChi2Storage(const int nEntriesAutoSave)
{
  if (fTreeChi2) {
    LOG(warning) << "MillePede2::InitChi2Storage() - output tree already initialized";
    return false;
  }
  if (fRecChi2File == nullptr) {
    fRecChi2File = new TFile(GetRecChi2FName(), "recreate", "", 505);
  }
  if ((!fRecChi2File) || (fRecChi2File->IsZombie())) {
    LOGF(error,
         "MillePede2::InitChi2Storage() - failed to initialise chi2 storage file %s!",
         GetRecChi2FName());
    return false;
  }
  if (fTreeChi2 == nullptr) {
    fRecChi2File->cd();
    fTreeChi2 = new TTree(fRecChi2TreeName.Data(), "Sum of chi2 per records");
    fTreeChi2->SetAutoSave(nEntriesAutoSave); // flush the TTree to disk every N entries
    fTreeChi2->SetImplicitMT(true);
    fTreeChi2->Branch("sumChi2", &fSumChi2, "fSumChi2/F");
    fTreeChi2->Branch("accepted", &fIsChi2BelowLimit, "fIsChi2BelowLimit/O");
    fTreeChi2->Branch("nDoF", &fRecNDoF, "fRecNDoF/I");
  }
  if (!fTreeChi2) {
    LOG(error) << "MillePede2::InitChi2Storage() - failed to initialise TTree !";
    return false;
  }
  LOGF(info, "MillePede2::InitChi2Storage() - chi2 storage file %s",
       GetRecChi2FName());
  return true;
}

//_____________________________________________________________________________
void MillePede2::EndChi2Storage()
{
  if (fRecChi2File && fRecChi2File->IsWritable() && fTreeChi2) {
    fRecChi2File->cd();
    fTreeChi2->Write();
    LOG(info) << "MillePede2::EndChi2Storage() - wrote tree "
              << fRecChi2TreeName.Data();
  }
}

//_____________________________________________________________________________
void MillePede2::SetLocalEquation(std::vector<double>& dergb, std::vector<double>& derlc,
                                  const double lMeas, const double lSigma)
{
  if (!fRecordWriter) {
    LOG(fatal) << "MillePede2::SetLocalEquation() - aborted: null pointer to record writer";
    return;
  }
  if (!fRecordWriter->isInitOk()) {
    LOG(fatal) << "MillePede2::SetLocalEquation() - aborted: unintialised record writer";
    return;
  }
  SetRecord(fRecordWriter->getRecord());

  // write data of single measurement
  if (lSigma <= 0.0) { // If parameter is fixed, then no equation
    for (int i = fNLocPar; i--;) {
      derlc[i] = 0.0;
    }
    for (int i = fNGloParIni; i--;) {
      dergb[i] = 0.0;
    }
    return;
  }

  fRecord->AddResidual(lMeas);

  // Retrieve local param interesting indices
  for (int i = 0; i < fNLocPar; i++) {
    if (!IsZero(derlc[i])) {
      fRecord->AddIndexValue(i, derlc[i]);
      derlc[i] = 0.0;
    }
  }

  fRecord->AddWeight(1.0 / lSigma / lSigma);

  // Idem for global parameters
  for (int i = 0; i < fNGloParIni; i++) {
    if (!IsZero(dergb[i])) {
      fRecord->AddIndexValue(i, dergb[i]);
      dergb[i] = 0.0;
      int idrg = GetRGId(i);
      fRecord->MarkGroup(idrg < 0 ? -1 : fParamGrID[i]);
    }
  }
}

//_____________________________________________________________________________
void MillePede2::SetLocalEquation(std::vector<int>& indgb, std::vector<double>& dergb,
                                  int ngb, std::vector<int>& indlc,
                                  std::vector<double>& derlc, const int nlc,
                                  const double lMeas, const double lSigma)
{
  if (!fRecordWriter) {
    LOG(fatal) << "MillePede2::SetLocalEquation() - aborted: null pointer to record writer";
    return;
  }
  if (!fRecordWriter->isInitOk()) {
    LOG(fatal) << "MillePede2::SetLocalEquation() - aborted: unintialised record writer";
    return;
  }
  SetRecord(fRecordWriter->getRecord());

  if (lSigma <= 0.0) { // If parameter is fixed, then no equation
    for (int i = nlc; i--;) {
      derlc[i] = 0.0;
    }
    for (int i = ngb; i--;) {
      dergb[i] = 0.0;
    }
    return;
  }

  fRecord->AddResidual(lMeas);

  // Retrieve local param interesting indices
  for (int i = 0; i < nlc; i++) {
    if (!IsZero(derlc[i])) {
      fRecord->AddIndexValue(indlc[i], derlc[i]);
      derlc[i] = 0.;
      indlc[i] = 0;
    }
  }

  fRecord->AddWeight(1. / lSigma / lSigma);

  // Idem for global parameters
  for (int i = 0; i < ngb; i++) {
    if (!IsZero(dergb[i])) {
      fRecord->AddIndexValue(indgb[i], dergb[i]);
      dergb[i] = 0.;
      indgb[i] = 0;
    }
  }
}

//_____________________________________________________________________________
void MillePede2::SetGlobalConstraint(const std::vector<double>& dergb,
                                     const double val, const double sigma,
                                     const bool doPrint)
{
  if (!fConstraintsRecWriter) {
    LOG(fatal) << "MillePede2::SetGlobalConstraint() - aborted: null pointer to record writer";
    return;
  }
  if (!fConstraintsRecWriter->isInitOk()) {
    LOG(fatal) << "MillePede2::SetGlobalConstraint() - aborted: unintialised record writer";
    return;
  }
  SetRecord(fConstraintsRecWriter->getRecord());

  fRecord->Reset();
  fRecord->AddResidual(val);
  fRecord->AddWeight(sigma);
  for (int i = 0; i < fNGloParIni; i++) {
    if (!IsZero(dergb[i])) {
      fRecord->AddIndexValue(i, dergb[i]);
    }
  }
  fNGloConstraints++;
  if (IsZero(sigma)) {
    fNLagrangeConstraints++;
  }
  if (doPrint) {
    LOG(info) << "MillePede2::SetGlobalConstraint() - new constraints added";
  }
  fConstraintsRecWriter->fillRecordTree(doPrint);
}

//_____________________________________________________________________________
void MillePede2::SetGlobalConstraint(const std::vector<int>& indgb,
                                     const std::vector<double>& dergb,
                                     const int ngb, const double val,
                                     const double sigma, const bool doPrint)
{
  if (!fConstraintsRecWriter) {
    LOG(fatal) << "MillePede2::SetGlobalConstraint() - aborted: null pointer to record writer";
    return;
  }
  if (!fConstraintsRecWriter->isInitOk()) {
    LOG(fatal) << "MillePede2::SetGlobalConstraint() - aborted: unintialised record writer";
    return;
  }
  SetRecord(fConstraintsRecWriter->getRecord());

  fRecord->Reset();
  fRecord->AddResidual(val);
  fRecord->AddWeight(sigma); // dummy
  for (int i = 0; i < ngb; i++) {
    if (!IsZero(dergb[i])) {
      fRecord->AddIndexValue(indgb[i], dergb[i]);
    }
  }
  fNGloConstraints++;
  if (IsZero(sigma)) {
    fNLagrangeConstraints++;
  }
  if (doPrint) {
    LOG(info) << "MillePede2::SetGlobalConstraint() - new constraints added";
  }
  fConstraintsRecWriter->fillRecordTree(doPrint);
}

//_____________________________________________________________________________
void MillePede2::ReadRecordData(const long recID, bool doPrint)
{
  if (fRecordReader == nullptr) {
    LOG(error) << "MillePede2::ReadRecordData() - aborted, input record reader is a null pointer";
    return;
  }
  SetRecord(fRecordReader->getRecord());
  fRecordReader->readEntry(recID, doPrint);
  fCurrRecDataID = recID;
}

//_____________________________________________________________________________
void MillePede2::ReadRecordConstraint(const long recID, bool doPrint)
{
  if (fConstraintsRecReader == nullptr) {
    LOG(error) << "MillePede2::ReadRecordConstraint() - aborted, input record reader is a null pointer";
    return;
  }
  SetRecord(fConstraintsRecReader->getRecord());
  fConstraintsRecReader->readEntry(recID, doPrint);
  fCurrRecConstrID = recID;
}

//_____________________________________________________________________________
int MillePede2::LocalFit(std::vector<double>& localParams)
{
  static int nrefSize = 0;
  //  static TArrayI refLoc,refGlo,nrefLoc,nrefGlo;
  static int *refLoc = 0, *refGlo = 0, *nrefLoc = 0, *nrefGlo = 0;
  int nPoints = 0;
  fIsChi2BelowLimit = true;

  SymMatrix& matCLoc = *fMatCLoc;
  MatrixSq& matCGlo = *fMatCGlo;
  RectMatrix& matCGloLoc = *fMatCGloLoc;

  std::fill(fVecBLoc.begin(), fVecBLoc.end(), 0.);
  matCLoc.Reset();

  int cnt = 0;
  int recSz = fRecord->GetSize();

  while (cnt < recSz) { // Transfer the measurement records to matrices
    // extract addresses of residual, weight and pointers on local and global derivatives for each point
    if (nrefSize <= nPoints) {
      int* tmpA = 0;
      nrefSize = 2 * (nPoints + 1);
      tmpA = refLoc;
      refLoc = new int[nrefSize];
      if (tmpA) {
        memcpy(refLoc, tmpA, nPoints * sizeof(int));
      }
      tmpA = refGlo;
      refGlo = new int[nrefSize];
      if (tmpA) {
        memcpy(refGlo, tmpA, nPoints * sizeof(int));
      }
      tmpA = nrefLoc;
      nrefLoc = new int[nrefSize];
      if (tmpA) {
        memcpy(nrefLoc, tmpA, nPoints * sizeof(int));
      }
      tmpA = nrefGlo;
      nrefGlo = new int[nrefSize];
      if (tmpA) {
        memcpy(nrefGlo, tmpA, nPoints * sizeof(int));
      }
    }

    refLoc[nPoints] = ++cnt;
    int nLoc = 0;
    while (!fRecord->IsWeight(cnt)) {
      nLoc++;
      cnt++;
    }
    nrefLoc[nPoints] = nLoc;

    refGlo[nPoints] = ++cnt;
    int nGlo = 0;
    while (!fRecord->IsResidual(cnt) && cnt < recSz) {
      nGlo++;
      cnt++;
    }
    nrefGlo[nPoints] = nGlo;

    nPoints++;
  }
  if (fMinRecordLength > 0 && nPoints < fMinRecordLength) {
    return 0; // ignore
  }

  double vl;

  double gloWgh = fRunWgh;
  if (fUseRecordWeight) {
    gloWgh *= fRecord->GetWeight(); // global weight for this set
  }
  int maxLocUsed = 0;

  for (int ip = nPoints; ip--;) { // Transfer the measurement records to matrices
    double resid = fRecord->GetValue(refLoc[ip] - 1);
    double weight = fRecord->GetValue(refGlo[ip] - 1) * gloWgh;
    int odd = (ip & 0x1);
    if (fWghScl[odd] > 0) {
      weight *= fWghScl[odd];
    }
    double* derLoc = fRecord->GetValue() + refLoc[ip];
    double* derGlo = fRecord->GetValue() + refGlo[ip];
    int* indLoc = fRecord->GetIndex() + refLoc[ip];
    int* indGlo = fRecord->GetIndex() + refGlo[ip];

    for (int i = nrefGlo[ip]; i--;) { // suppress the global part (only relevant with iterations)

      // if regrouping was requested, do it here
      if (fkReGroup.size()) {
        int idtmp = fkReGroup[indGlo[i]];
        if (idtmp == kFixParID) {
          indGlo[i] = kFixParID; // fixed param in regrouping
        } else {
          indGlo[i] = idtmp;
        }
      }

      int iID = indGlo[i]; // Global param indice
      if (iID < 0 || fSigmaPar[iID] <= 0.) {
        continue; // fixed parameter RRRCheck
      }
      if (fIsLinear[iID]) {
        resid -= derGlo[i] * (fInitPar[iID] + fDeltaPar[iID]); // linear parameter
      } else {
        resid -= derGlo[i] * fDeltaPar[iID]; // nonlinear parameter
      }
    }

    // Symmetric matrix, don't bother j>i coeffs
    for (int i = nrefLoc[ip]; i--;) { // Fill local matrix and vector
      fVecBLoc[indLoc[i]] += weight * resid * derLoc[i];
      if (indLoc[i] > maxLocUsed) {
        maxLocUsed = indLoc[i];
      }
      for (int j = i + 1; j--;) {
        matCLoc(indLoc[i], indLoc[j]) += weight * derLoc[i] * derLoc[j];
      }
    }

  } // end of the transfer of the measurement record to matrices

  matCLoc.SetSizeUsed(++maxLocUsed); // data with B=0 may use less than declared nLocals

  /* //RRR
  fRecord->Print("l");
  printf("\nBefore\nLocalMatrix: "); matCLoc.Print("l");
  printf("RHSLoc: "); for (int i=0;i<fNLocPar;i++) printf("%+e |",fVecBLoc[i]); printf("\n");
  */
  // first try to solve by faster Cholesky decomposition, then by Gaussian elimination
  double* pVecBLoc = &fVecBLoc[0];
  if (!matCLoc.SolveChol(pVecBLoc, true)) {
    LOG(warning) << "MillePede2 - Failed to solve locals by Cholesky, trying Gaussian Elimination";
    if (!matCLoc.SolveSpmInv(pVecBLoc, true)) {
      LOG(warning) << "MillePede2 - Failed to solve locals by Gaussian Elimination, skip...";
      matCLoc.Print("d");
      return 0; // failed to solve
    }
  }

  // If requested, store the track params and errors
  // RRR  printf("locfit: "); for (int i=0;i<fNLocPar;i++) printf("%+e |",fVecBLoc[i]); printf("\n");

  if (localParams.size())
    for (int i = maxLocUsed; i--;) {
      localParams[2 * i] = fVecBLoc[i];
      localParams[2 * i + 1] = TMath::Sqrt(TMath::Abs(matCLoc.QueryDiag(i)));
    }

  float lChi2 = 0;
  int nEq = 0;

  for (int ip = nPoints; ip--;) { // Calculate residuals
    double resid = fRecord->GetValue(refLoc[ip] - 1);
    double weight = fRecord->GetValue(refGlo[ip] - 1) * gloWgh;
    int odd = (ip & 0x1);
    if (fWghScl[odd] > 0) {
      weight *= fWghScl[odd];
    }
    double* derLoc = fRecord->GetValue() + refLoc[ip];
    double* derGlo = fRecord->GetValue() + refGlo[ip];
    int* indLoc = fRecord->GetIndex() + refLoc[ip];
    int* indGlo = fRecord->GetIndex() + refGlo[ip];

    // Suppress local and global contribution in residuals;
    for (int i = nrefLoc[ip]; i--;) {
      resid -= derLoc[i] * fVecBLoc[indLoc[i]];
    } // local part

    for (int i = nrefGlo[ip]; i--;) { // global part
      int iID = indGlo[i];
      if (iID < 0 || fSigmaPar[iID] <= 0.) {
        continue;
      } // fixed parameter RRRCheck
      if (fIsLinear[iID]) {
        resid -= derGlo[i] * (fInitPar[iID] + fDeltaPar[iID]); // linear parameter
      } else {
        resid -= derGlo[i] * fDeltaPar[iID]; // nonlinear parameter
      }
    }

    // reject the track if the residual is too large (outlier)
    double absres = TMath::Abs(resid);
    if ((absres >= fResCutInit && fIter == 1) || (absres >= fResCut && fIter > 1)) {
      if (fLocFitAdd) {
        fNLocFitsRejected++;
      }
      LOGF(info, "MillePede2 - reject res %+e in record %5ld ", resid, fCurrRecDataID); // A.R. comment
      return 0;
    }

    lChi2 += weight * resid * resid; // total chi^2
    nEq++;                           // number of equations
  }                                  // end of Calculate residuals

  lChi2 /= gloWgh;
  int nDoF = nEq - maxLocUsed;
  lChi2 = (nDoF > 0) ? lChi2 / nDoF : 0; // Chi^2/dof
  fSumChi2 = lChi2;
  fRecNDoF = nDoF;

  if (fNStdDev != 0 && nDoF > 0 && lChi2 > Chi2DoFLim(fNStdDev, nDoF) * fChi2CutFactor) { // check final chi2
    fIsChi2BelowLimit = false;
    if (GetCurrentIteration() == 1 && fTreeChi2) {
      fTreeChi2->Fill();
    }
    if (fLocFitAdd) {
      fNLocFitsRejected++;
    }
    LOGF(debug, "MillePede2 - reject chi2 %+e record %5ld: (nDOF %d)", lChi2, fCurrRecDataID, nDoF); // A.R. comment
    // fRecord->Print();                                                                                // A.R. comment
    return 0;
  }

  if (fLocFitAdd) {
    fNLocFits++;
    fNLocEquations += nEq;
  } else {
    fNLocFits--;
    fNLocEquations -= nEq;
  }

  //  local operations are finished, track is accepted
  //  We now update the global parameters (other matrices)

  int nGloInFit = 0;

  for (int ip = nPoints; ip--;) { // Update matrices
    double resid = fRecord->GetValue(refLoc[ip] - 1);
    double weight = fRecord->GetValue(refGlo[ip] - 1) * gloWgh;
    int odd = (ip & 0x1);
    if (fWghScl[odd] > 0) {
      weight *= fWghScl[odd];
    }
    double* derLoc = fRecord->GetValue() + refLoc[ip];
    double* derGlo = fRecord->GetValue() + refGlo[ip];
    int* indLoc = fRecord->GetIndex() + refLoc[ip];
    int* indGlo = fRecord->GetIndex() + refGlo[ip];

    for (int i = nrefGlo[ip]; i--;) { // suppress the global part
      int iID = indGlo[i];            // Global param indice
      if (iID < 0 || fSigmaPar[iID] <= 0.) {
        continue;
      } // fixed parameter RRRCheck
      if (fIsLinear[iID]) {
        resid -= derGlo[i] * (fInitPar[iID] + fDeltaPar[iID]); // linear parameter
      } else {
        resid -= derGlo[i] * fDeltaPar[iID]; // nonlinear parameter
      }
    }

    for (int ig = nrefGlo[ip]; ig--;) {
      int iIDg = indGlo[ig]; // Global param indice (the matrix line)
      if (iIDg < 0 || fSigmaPar[iIDg] <= 0.) {
        continue;
      } // fixed parameter RRRCheck
      if (fLocFitAdd) {
        fVecBGlo[iIDg] += weight * resid * derGlo[ig];
      } else {
        fVecBGlo[iIDg] -= weight * resid * derGlo[ig];
      }

      // First of all, the global/global terms (exactly like local matrix)
      int nfill = 0;
      for (int jg = ig + 1; jg--;) { // matCGlo is symmetric by construction
        int jIDg = indGlo[jg];
        if (jIDg < 0 || fSigmaPar[jIDg] <= 0.) {
          continue;
        } // fixed parameter RRRCheck
        if (!IsZero(vl = weight * derGlo[ig] * derGlo[jg])) {
          fFillIndex[nfill] = jIDg;
          fFillValue[nfill++] = fLocFitAdd ? vl : -vl;
        }
      }
      if (nfill) {
        matCGlo.AddToRow(iIDg, fFillValue.data(), fFillIndex.data(), nfill);
      }

      // Now we have also rectangular matrices containing global/local terms.
      int iCIDg = fGlo2CGlo[iIDg]; // compressed Index of index
      if (iCIDg == -1) {
        double* rowGL = matCGloLoc(nGloInFit);
        for (int k = maxLocUsed; k--;) {
          rowGL[k] = 0.0;
        } // reset the row
        iCIDg = fGlo2CGlo[iIDg] = nGloInFit;
        fCGlo2Glo[nGloInFit++] = iIDg;
      }

      double* rowGLIDg = matCGloLoc(iCIDg);
      for (int il = nrefLoc[ip]; il--;) {
        rowGLIDg[indLoc[il]] += weight * derGlo[ig] * derLoc[il];
      }
      fProcPnt[iIDg] += fLocFitAdd ? 1 : -1; // update counter
    }
  } // end of Update matrices
  //
  /*//RRR
  LOG(info) << "MillePede2 - After GLO";
  printf("MatCLoc: "); fMatCLoc->Print("l");
  printf("MatCGlo: "); fMatCGlo->Print("l");
  printf("MatCGlLc:"); fMatCGloLoc->Print("l");
  printf("BGlo: "); for (int i=0; i<fNGloPar; i++) printf("%+e |",fVecBGlo[i]); printf("\n");
  */
  // calculate fMatCGlo -= fMatCGloLoc * fMatCLoc * fMatCGloLoc^T
  // and       fVecBGlo -= fMatCGloLoc * fVecBLoc
  //
  //-------------------------------------------------------------- >>>
  double vll;
  for (int iCIDg = 0; iCIDg < nGloInFit; iCIDg++) {
    int iIDg = fCGlo2Glo[iCIDg];

    vl = 0;
    double* rowGLIDg = matCGloLoc(iCIDg);
    for (int kl = 0; kl < maxLocUsed; kl++)
      if (rowGLIDg[kl]) {
        vl += rowGLIDg[kl] * fVecBLoc[kl];
      }
    if (!IsZero(vl)) {
      fVecBGlo[iIDg] -= fLocFitAdd ? vl : -vl;
    }

    int nfill = 0;
    for (int jCIDg = 0; jCIDg <= iCIDg; jCIDg++) {
      int jIDg = fCGlo2Glo[jCIDg];

      vl = 0;
      double* rowGLJDg = matCGloLoc(jCIDg);
      for (int kl = 0; kl < maxLocUsed; kl++) {
        // diag terms
        if ((!IsZero(vll = rowGLIDg[kl] * rowGLJDg[kl]))) {
          vl += matCLoc.QueryDiag(kl) * vll;
        }
        //
        // off-diag terms
        for (int ll = 0; ll < kl; ll++) {
          if (!IsZero(vll = rowGLIDg[kl] * rowGLJDg[ll])) {
            vl += matCLoc(kl, ll) * vll;
          }
          if (!IsZero(vll = rowGLIDg[ll] * rowGLJDg[kl])) {
            vl += matCLoc(kl, ll) * vll;
          }
        }
      }
      if (!IsZero(vl)) {
        fFillIndex[nfill] = jIDg;
        fFillValue[nfill++] = fLocFitAdd ? -vl : vl;
      }
    }
    if (nfill) {
      matCGlo.AddToRow(iIDg, fFillValue.data(), fFillIndex.data(), nfill);
    }
  }

  // reset compressed index array

  /*//RRR
  LOG(info) << "MillePede2 - After GLOLoc";
  printf("MatCGlo: "); fMatCGlo->Print("");
  printf("BGlo: "); for (int i=0; i<fNGloPar; i++) printf("%+e |",fVecBGlo[i]); printf("\n");
  */
  for (int i = nGloInFit; i--;) {
    fGlo2CGlo[fCGlo2Glo[i]] = -1;
    fCGlo2Glo[i] = -1;
  }
  //
  //---------------------------------------------------- <<<
  if (GetCurrentIteration() == 1 && fTreeChi2) {
    fTreeChi2->Fill();
  }
  return 1;
}

//_____________________________________________________________________________
int MillePede2::GlobalFit(double* par, double* error, double* pull)
{
  if (fRecordReader == nullptr) {
    LOG(fatal) << "MillePede2::GlobalFit() - aborted, input record reader is a null pointer";
    return 1;
  }
  fIter = 1;

  TStopwatch sw;
  sw.Start();

  int res = 0;
  LOG(info) << "MillePede2 - Starting Global fit.";
  while (fIter <= fMaxIter) {
    //
    res = GlobalFitIteration();
    if (!res) {
      break;
    }
    //
    if (!IsZero(fChi2CutFactor - fChi2CutRef)) {
      fChi2CutFactor = TMath::Sqrt(fChi2CutFactor);
      if (fChi2CutFactor < 1.2 * fChi2CutRef) {
        fChi2CutFactor = fChi2CutRef;
        // fIter = fMaxIter - 1; // RRR // Last iteration
      }
    }
    fIter++;
  }

  sw.Stop();
  LOGF(info, "MillePede2 - Global fit %s, CPU time: %.1f", res ? "Converged" : "Failed", sw.CpuTime());
  if (!res) {
    return 0;
  }

  if (par) {
    for (int i = fNGloParIni; i--;) {
      par[i] = GetFinalParam(i);
    }
  }

  if (fGloSolveStatus == kInvert) { // errors on params are available
    if (error) {
      for (int i = fNGloParIni; i--;) {
        error[i] = GetFinalError(i);
      }
    }
    if (pull) {
      for (int i = fNGloParIni; i--;) {
        pull[i] = GetPull(i);
      }
    }
  }

  return 1;
}

//_____________________________________________________________________________
int MillePede2::GlobalFitIteration()
{
  LOGF(info, "MillePede2 - Global Fit Iteration#%2d (Local Fit Chi^2 cut factor: %.2f)", fIter, fChi2CutFactor);

  if (!fRecordReader) {
    LOG(info) << "MillePede2::GlobalFitIteration() - no record is accessible, stopping iteration";
    return 0;
  }
  if (!fNGloPar) {
    LOG(info) << "MillePede2::GlobalFitIteration() - zero global parameter, stopping iteration";
    return 0;
  }
  TStopwatch sw, sws;
  sw.Start();
  sws.Stop();

  if (!fConstrUsed.size()) {
    fConstrUsed = std::vector<bool>(fNGloConstraints);
    std::fill(fConstrUsed.begin(), fConstrUsed.end(), 0);
  }
  // Reset all info specific for this step
  MatrixSq& matCGlo = *fMatCGlo;
  matCGlo.Reset();
  std::fill(fProcPnt.begin(), fProcPnt.end(), 0);

  fNGloConstraints = fConstraintsRecReader ? fConstraintsRecReader->getNEntries() : 0;

  // count number of Lagrange constraints: they need new row/cols to be added
  fNLagrangeConstraints = 0;
  for (int i = 0; i < fNGloConstraints; i++) {
    ReadRecordConstraint(i);
    if (!fConstraintsRecReader->isReadEntryOk()) {
      continue;
    }
    if (IsZero(fRecord->GetValue(1))) {
      fNLagrangeConstraints++; // exact constraint (no error) -> Lagrange multiplier
    }
  }

  // if needed, readjust the size of the global vector (for matrices this is done automatically)
  if (!fVecBGlo.size() || fNGloSize != fNGloPar + fNLagrangeConstraints) {
    fVecBGlo.clear(); // in case some constraint was added between the two manual iterations
    fNGloSize = fNGloPar + fNLagrangeConstraints;
    fVecBGlo = std::vector<double>(fNGloSize);
  }

  fNLocFits = 0;
  fNLocFitsRejected = 0;
  fNLocEquations = 0;

  //  Process data records and build the matrices
  long ndr = fRecordReader->getNEntries();
  long first = fSelFirst > 0 ? fSelFirst : 0;
  long last = fSelLast < 1 ? ndr : (fSelLast >= ndr ? ndr : fSelLast + long(1));
  ndr = last - first;

  LOGF(info, "MillePede2 - Building the Global matrix from data records %ld : %ld", first, last);
  if (ndr < 1) {
    return 0;
  }

  TStopwatch swt;
  swt.Start();
  fLocFitAdd = true; // add contributions of matching tracks
  for (long i = 0; i < ndr; i++) {
    long iev = i + first;
    ReadRecordData(iev);
    if (!IsRecordAcceptable() || !fRecordReader->isReadEntryOk()) {
      continue;
    }
    std::vector<double> emptyLocalParams = {};
    LocalFit(emptyLocalParams);
    if ((i % int(0.2 * ndr)) == 0) {
      printf("%.1f%% of local fits done\n", double(100. * i) / ndr);
    }
  }
  swt.Stop();
  LOGF(info, "MillePede2 - %ld local fits done: ", ndr);
  /*
  printf("MatCGlo: "); fMatCGlo->Print("l");
  printf("BGlo: "); for (int i=0; i<fNGloPar; i++) printf("%+e |",fVecBGlo[i]); printf("\n");
  swt.Print();
  */
  sw.Start(false);

  // ---------------------- Reject parameters with low statistics ------------>>
  fNGloFix = 0;
  if (fMinPntValid > 1 && fNGroupsSet) {
    //
    LOGF(info, "MillePede2 - Checking parameters with statistics < %d", fMinPntValid);
    TStopwatch swsup;
    swsup.Start();
    // 1) build the list of parameters to fix
    int fixArrSize = 10;
    int nFixedGroups = 0;
    TArrayI fixGroups(fixArrSize);
    //
    int grIDold = -2;
    int oldStart = -1;
    double oldMin = 1.e20;
    double oldMax = -1.e20;

    for (int i = fNGloPar; i--;) { // Reset row and column of fixed params and add 1/sig^2 to free ones
      int grID = fParamGrID[i];
      if (grID < 0) {
        continue; // not in the group
      }

      if (grID != grIDold) {                                        // starting new group
        if (grIDold >= 0) {                                         // decide if the group has enough statistics
          if (oldMin < fMinPntValid && oldMax < 2 * fMinPntValid) { // suppress group
            for (int iold = oldStart; iold > i; iold--) {
              fProcPnt[iold] = 0;
            }
            bool fnd = false; // check if the group is already accounted
            for (int j = nFixedGroups; j--;)
              if (fixGroups[j] == grIDold) {
                fnd = true;
                break;
              }
            if (!fnd) {
              if (nFixedGroups >= fixArrSize) {
                fixArrSize *= 2;
                fixGroups.Set(fixArrSize);
              }
              fixGroups[nFixedGroups++] = grIDold; // add group to fix
            }
          }
        }
        grIDold = grID; // mark the start of the new group
        oldStart = i;
        oldMin = 1.e20;
        oldMax = -1.e20;
      }
      if (oldMin > fProcPnt[i]) {
        oldMin = fProcPnt[i];
      }
      if (oldMax < fProcPnt[i]) {
        oldMax = fProcPnt[i];
      }
    }
    // extra check for the last group
    if (grIDold >= 0 && oldMin < fMinPntValid && oldMax < 2 * fMinPntValid) { // suppress group
      for (int iold = oldStart; iold--;) {
        fProcPnt[iold] = 0;
      }
      bool fnd = false; // check if the group is already accounted
      for (int j = nFixedGroups; j--;) {
        if (fixGroups[j] == grIDold) {
          fnd = true;
          break;
        }
      }
      if (!fnd) {
        if (nFixedGroups >= fixArrSize) {
          fixArrSize *= 2;
          fixGroups.Set(fixArrSize);
        }
        fixGroups[nFixedGroups++] = grIDold; // add group to fix
      }
    }

    // 2) loop over records and add contributions of fixed groups with negative sign
    fLocFitAdd = false;

    for (long i = 0; i < ndr; i++) {
      long iev = i + first;
      ReadRecordData(iev);
      if (!IsRecordAcceptable() || !fRecordReader->isReadEntryOk()) {
        continue;
      }
      bool suppr = false;
      for (int ifx = nFixedGroups; ifx--;) {
        if (fRecord->IsGroupPresent(fixGroups[ifx])) {
          suppr = true;
        }
      }
      if (suppr) {
        std::vector<double> emptyLocalParams = {};
        LocalFit(emptyLocalParams);
      }
    }
    fLocFitAdd = true;

    if (nFixedGroups) {
      LOGF(info, "MillePede2 - Suppressed contributions of groups with NPoints < %d :", fMinPntValid);
      for (int i = 0; i < nFixedGroups; i++)
        printf("%d ", fixGroups[i]);
      printf("\n");
    }
    swsup.Stop();
    swsup.Print();
  }
  // ---------------------- Reject parameters with low statistics ------------<<

  // add large number to diagonal of fixed params

  for (int i = fNGloPar; i--;) { // Reset row and column of fixed params and add 1/sig^2 to free ones
                                 // printf("#%3d : Nproc : %5d grp: %d\n",i,fProcPnt[i],fParamGrID[i]);
    if (fProcPnt[i] < 1) {
      fNGloFix++;
      fVecBGlo[i] = 0.;
      matCGlo.DiagElem(i) = 1.;
      // float(fNLocEquations*fNLocEquations);
      // matCGlo.DiagElem(i) = float(fNLocEquations*fNLocEquations);
    } else {
      matCGlo.DiagElem(i) += (fgWeightSigma ? fProcPnt[i] : 1.) / (fSigmaPar[i] * fSigmaPar[i]);
    }
  }

  for (int i = fNGloPar; i--;) {
    fDiagCGlo[i] = matCGlo.QueryDiag(i); // save the diagonal elements
  }

  // add constraint equations
  int nVar = fNGloPar; // Current size of global matrix
  for (int i = 0; i < fNGloConstraints; i++) {
    ReadRecordConstraint(i);
    if (!fConstraintsRecReader->isReadEntryOk()) {
      continue;
    }
    double val = fRecord->GetValue(0);
    double sig = fRecord->GetValue(1);
    int* indV = fRecord->GetIndex() + 2;
    double* der = fRecord->GetValue() + 2;
    int csize = fRecord->GetSize() - 2;
    //
    if (fkReGroup.size()) {
      for (int jp = csize; jp--;) {
        int idp = indV[jp];
        if (fkReGroup[idp] < 0) {
          LOGF(fatal, "MillePede2 - Constain is requested for suppressed parameter #%d", indV[jp]);
        }
        indV[jp] = idp;
      }
    }
    // check if after suppression of fixed variables there are non-0 derivatives
    // and determine the max statistics of involved params
    int nSuppressed = 0;
    int maxStat = 1;
    for (int j = csize; j--;) {
      if (fProcPnt[indV[j]] < 1) {
        nSuppressed++;
      } else {
        maxStat = TMath::Max(maxStat, fProcPnt[indV[j]]);
      }
    }

    if (nSuppressed == csize) {
      // LOG(info << Form("Neglecting constraint %d of %d derivatives since no free parameters left",i,csize));

      // was this constraint ever created ?
      if (sig == 0 && fConstrUsed[i]) { // this is needed only for constraints with Lagrange multiplier
        // to avoid empty row impose dummy constraint on "Lagrange multiplier"
        matCGlo.DiagElem(nVar) = 1.;
        fVecBGlo[nVar++] = 0;
      }
      continue;
    }

    // account for already accumulated corrections
    for (int j = csize; j--;) {
      val -= der[j] * (fInitPar[indV[j]] + fDeltaPar[indV[j]]);
    }

    if (sig > 0) { // this is a gaussian constriant: no Lagrange multipliers are added

      double sig2i = (fgWeightSigma ? TMath::Sqrt(maxStat) : 1.) / sig / sig;
      for (int ir = 0; ir < csize; ir++) {
        int iID = indV[ir];
        for (int ic = 0; ic <= ir; ic++) { // matrix is symmetric
          int jID = indV[ic];
          double vl = der[ir] * der[ic] * sig2i;
          if (!IsZero(vl)) {
            matCGlo(iID, jID) += vl;
          }
        }
        fVecBGlo[iID] += val * der[ir] * sig2i;
      }
    } else { // this is exact constriant:  Lagrange multipliers must be added
      for (int j = csize; j--;) {
        int jID = indV[j];
        if (fProcPnt[jID] < 1) { // this parameter was fixed, don't put it into constraint
          continue;
        }
        matCGlo(nVar, jID) = float(fNLocEquations) * der[j]; // fMatCGlo is symmetric, only lower triangle is filled
      }

      if (matCGlo.QueryDiag(nVar)) {
        matCGlo.DiagElem(nVar) = 0.0;
      }
      fVecBGlo[nVar++] = float(fNLocEquations) * val; // RS ? should we use here fNLocFits ?
      fConstrUsed[i] = true;
    }
  }

  LOGF(info, "MillePede2 - Obtained %-7ld equations from %-7ld records (%-7ld rejected). Fixed %-4d globals",
       fNLocEquations, fNLocFits, fNLocFitsRejected, fNGloFix);

  sws.Start();

#ifdef _DUMP_EQ_BEFORE_
  const char* faildumpB = Form("mp2eq_before%d.dat", fIter);
  int defoutB = dup(1);
  if (defoutB < 0) {
    LOG(fatal) << "Failed on dup";
    exit(1);
  }
  int slvDumpB = open(faildumpB, O_RDWR | O_CREAT, 0666);
  if (slvDumpB >= 0) {
    dup2(slvDumpB, 1);
    printf("Solving%d for %d params\n", fIter, fNGloSize);
    matCGlo.Print("10");
    for (int i = 0; i < fNGloSize; i++)
      printf("b%2d : %+.10f\n", i, fVecBGlo[i]);
  }
  dup2(defoutB, 1);
  close(slvDumpB);
  close(defoutB);

#endif
  /*
  printf("Solving:\n");
  matCGlo.Print("l");
  for (int i=0;i<fNGloSize;i++) printf("b%2d : %+e\n",i,fVecBGlo[i]);
  */
#ifdef _DUMPEQ_BEFORE_
  const char* faildumpB = Form("mp2eq_before%d.dat", fIter);
  int defoutB = dup(1);
  int slvDumpB = open(faildumpB, O_RDWR | O_CREAT, 0666);
  dup2(slvDumpB, 1);
  //
  printf("#Equation before step %d\n", fIter);
  fMatCGlo->Print("10");
  printf("#RHS/STAT : NGlo:%d NGloSize:%d\n", fNGloPar, fNGloSize);
  for (int i = 0; i < fNGloSize; i++) {
    printf("%d %+.10f %d\n", i, fVecBGlo[i], fProcPnt[i]);
  }
  //
  dup2(defoutB, 1);
  close(slvDumpB);
  close(defoutB);
#endif
  //
  fGloSolveStatus = SolveGlobalMatEq(); // obtain solution for this step
#ifdef _DUMPEQ_AFTER_
  const char* faildumpA = Form("mp2eq_after%d.dat", fIter);
  int defoutA = dup(1);
  int slvDumpA = open(faildumpA, O_RDWR | O_CREAT, 0666);
  dup2(slvDumpA, 1);
  //
  printf("#Matrix after step %d\n", fIter);
  fMatCGlo->Print("10");
  printf("#RHS/STAT : NGlo:%d NGloSize:%d\n", fNGloPar, fNGloSize);
  for (int i = 0; i < fNGloSize; i++) {
    printf("%d %+.10f %d\n", i, fVecBGlo[i], fProcPnt[i]);
  }
  //
  dup2(defoutA, 1);
  close(slvDumpA);
  close(defoutA);
#endif
  //
  sws.Stop();
  printf("Solve %d |", fIter);
  sws.Print();
  //
  sw.Stop();
  LOGF(info, "MillePede2 - Iteration#%2d %s. CPU time: %.1f", fIter, fGloSolveStatus == kFailed ? "Failed" : "Converged", sw.CpuTime());
  if (fGloSolveStatus == kFailed)
    return 0;
  //
  for (int i = fNGloPar; i--;) { // Update global parameters values (for iterations)
    fDeltaPar[i] += fVecBGlo[i];
  }

#ifdef _DUMP_EQ_AFTER_
  const char* faildumpA = Form("mp2eq_after%d.dat", fIter);
  int defoutA = dup(1);
  if (defoutA < 0) {
    LOG(fatal) << "Failed on dup";
    exit(1);
  }
  int slvDumpA = open(faildumpA, O_RDWR | O_CREAT, 0666);
  if (slvDumpA >= 0) {
    dup2(slvDumpA, 1);
    printf("Solving%d for %d params\n", fIter, fNGloSize);
    matCGlo.Print("10");
    for (int i = 0; i < fNGloSize; i++) {
      printf("b%2d : %+.10f\n", i, fVecBGlo[i]);
    }
  }
  dup2(defoutA, 1);
  close(slvDumpA);
  close(defoutA);
#endif
  //
  /*
  printf("Solved:\n");
  matCGlo.Print("l");
  for (int i=0;i<fNGloSize;i++) printf("b%2d : %+e (->%+e)\n",i,fVecBGlo[i], fDeltaPar[i]);
  */

  PrintGlobalParameters();
  return 1;
}

//_____________________________________________________________________________
int MillePede2::SolveGlobalMatEq()
{
  /*
  printf("GlobalMatrix\n");
  fMatCGlo->Print("l");
  printf("RHS\n");
  for (int i=0;i<fNGloPar;i++) printf("%d %+e\n",i,fVecBGlo[i]);
  */

  if (!fgIsMatGloSparse) {
    if (fNLagrangeConstraints == 0) { // pos-def systems are faster to solve by Cholesky
      if (((SymMatrix*)fMatCGlo)->SolveChol(fVecBGlo.data(), fgInvChol)) {
        return fgInvChol ? kInvert : kNoInversion;
      } else {
        LOG(warning) << "MillePede2 - Solution of Global Dense System by Cholesky failed, trying Gaussian Elimiation";
      }
    }
    if (((SymMatrix*)fMatCGlo)->SolveSpmInv(fVecBGlo.data(), true)) {
      return kInvert;
    } else {
      LOG(warning) << "MillePede2 - Solution of Global Dense System by Gaussian Elimination failed, trying iterative methods";
    }
  }
  // try to solve by minres
  TVectorD sol(fNGloSize);

  MinResSolve* slv = new MinResSolve(fMatCGlo, fVecBGlo.data());
  if (!slv)
    return kFailed;

  bool res = false;
  if (fgIterSol == MinResSolve::kSolMinRes) {
    res = slv->SolveMinRes(sol, fgMinResCondType, fgMinResMaxIter, fgMinResTol);
  } else {
    if (fgIterSol == MinResSolve::kSolFGMRes) {
      res = slv->SolveFGMRES(sol, fgMinResCondType, fgMinResMaxIter, fgMinResTol, fgNKrylovV);
    } else {
      LOGF(warning, "MillePede2 - Undefined Iteritive Solver ID=%d, only %d are defined", fgIterSol, MinResSolve::kNSolvers);
    }
  }

  if (!res) {
    const char* faildump = "fgmr_failed.dat";
    int defout = dup(1);
    if (defout < 0) {
      LOG(warning) << "Failed on dup";
      return kFailed;
    }
    int slvDump = open(faildump, O_RDWR | O_CREAT, 0666);
    if (slvDump >= 0) {
      dup2(slvDump, 1);
      printf("#Failed to solve using solver %d with PreCond: %d MaxIter: %d Tol: %e NKrylov: %d\n",
             fgIterSol, fgMinResCondType, fgMinResMaxIter, fgMinResTol, fgNKrylovV);
      printf("#Dump of matrix:\n");
      fMatCGlo->Print("10");
      printf("#Dump of RHS:\n");
      for (int i = 0; i < fNGloSize; i++) {
        printf("%d %+.10f\n", i, fVecBGlo[i]);
      }
      dup2(defout, 1);
      close(slvDump);
      close(defout);
      printf("#Dumped failed matrix and RHS to %s\n", faildump);
    } else {
      LOG(warning) << "MillePede2 - Failed on file open for matrix dumping";
    }
    close(defout);
    return kFailed;
  }
  for (int i = fNGloSize; i--;) {
    fVecBGlo[i] = sol[i];
  }

  return kNoInversion;
}

//_____________________________________________________________________________
Float_t MillePede2::Chi2DoFLim(int nSig, int nDoF) const
{
  int lNSig;
  float sn[3] = {0.47523, 1.690140, 2.782170};
  float table[3][30] = {{1.0000, 1.1479, 1.1753, 1.1798, 1.1775, 1.1730, 1.1680, 1.1630,
                         1.1581, 1.1536, 1.1493, 1.1454, 1.1417, 1.1383, 1.1351, 1.1321,
                         1.1293, 1.1266, 1.1242, 1.1218, 1.1196, 1.1175, 1.1155, 1.1136,
                         1.1119, 1.1101, 1.1085, 1.1070, 1.1055, 1.1040},
                        {4.0000, 3.0900, 2.6750, 2.4290, 2.2628, 2.1415, 2.0481, 1.9736,
                         1.9124, 1.8610, 1.8171, 1.7791, 1.7457, 1.7161, 1.6897, 1.6658,
                         1.6442, 1.6246, 1.6065, 1.5899, 1.5745, 1.5603, 1.5470, 1.5346,
                         1.5230, 1.5120, 1.5017, 1.4920, 1.4829, 1.4742},
                        {9.0000, 5.9146, 4.7184, 4.0628, 3.6410, 3.3436, 3.1209, 2.9468,
                         2.8063, 2.6902, 2.5922, 2.5082, 2.4352, 2.3711, 2.3143, 2.2635,
                         2.2178, 2.1764, 2.1386, 2.1040, 2.0722, 2.0428, 2.0155, 1.9901,
                         1.9665, 1.9443, 1.9235, 1.9040, 1.8855, 1.8681}};

  if (nDoF < 1) {
    return 0.0;
  } else {
    lNSig = TMath::Max(1, TMath::Min(nSig, 3));

    if (nDoF <= 30) {
      return table[lNSig - 1][nDoF - 1];
    } else { // approximation
      return ((sn[lNSig - 1] + TMath::Sqrt(float(2 * nDoF - 3))) *
              (sn[lNSig - 1] + TMath::Sqrt(float(2 * nDoF - 3)))) /
             float(2 * nDoF - 2);
    }
  }
}

//_____________________________________________________________________________
int MillePede2::SetIterations(const double lChi2CutFac)
{
  fChi2CutFactor = TMath::Max(1.0, lChi2CutFac);
  LOGF(info, "MillePede2 - Initial cut factor is %f", fChi2CutFactor);
  fIter = 1; // Initializes the iteration process
  return 1;
}

//_____________________________________________________________________________
double MillePede2::GetParError(int iPar) const
{
  if (fGloSolveStatus == kInvert) {
    if (fkReGroup.size()) {
      iPar = fkReGroup[iPar];
    }
    if (iPar < 0) {
      // LOG(debug) << Form("Parameter %d was suppressed in the regrouping",iPar));
      return 0;
    }
    double res = fMatCGlo->QueryDiag(iPar);
    if (res >= 0) {
      return TMath::Sqrt(res);
    }
  }
  return 0.;
}

//_____________________________________________________________________________
double MillePede2::GetPull(int iPar) const
{
  if (fGloSolveStatus == kInvert) {
    if (fkReGroup.size()) {
      iPar = fkReGroup[iPar];
    }
    if (iPar < 0) {
      // LOG(debug) << Form("Parameter %d was suppressed in the regrouping",iPar));
      return 0;
    }
    return fProcPnt[iPar] > 0 && (fSigmaPar[iPar] * fSigmaPar[iPar] - fMatCGlo->QueryDiag(iPar)) > 0. && fSigmaPar[iPar] > 0
             ? fDeltaPar[iPar] / TMath::Sqrt(fSigmaPar[iPar] * fSigmaPar[iPar] - fMatCGlo->QueryDiag(iPar))
             : 0;
  }
  return 0.;
}

//_____________________________________________________________________________
int MillePede2::PrintGlobalParameters() const
{
  double lError = 0.;
  double lGlobalCor = 0.;

  printf("\nMillePede2 output\n");
  printf("   Result of fit for global parameters\n");
  printf("   ===================================\n");
  printf("    I       initial       final       differ        lastcor        error       gcor       Npnt\n");
  printf("----------------------------------------------------------------------------------------------\n");
  //
  int lastPrintedId = -1;
  for (int i0 = 0; i0 < fNGloParIni; i0++) {
    int i = GetRGId(i0);
    if (i < 0) {
      continue;
    }
    if (i != i0 && lastPrintedId >= 0 && i <= lastPrintedId) { // grouped param
      continue;
    }
    lastPrintedId = i;
    lError = GetParError(i0);
    lGlobalCor = 0.0;
    double dg;
    if (fGloSolveStatus == kInvert && TMath::Abs((dg = fMatCGlo->QueryDiag(i)) * fDiagCGlo[i]) > 0) {
      lGlobalCor = TMath::Sqrt(TMath::Abs(1.0 - 1.0 / (dg * fDiagCGlo[i])));
      printf("%4d(%4d)\t %+.6f\t %+.6f\t %+.6f\t %.6f\t %.6f\t %.6f\t %6d\n",
             i, i0, fInitPar[i], fInitPar[i] + fDeltaPar[i], fDeltaPar[i], fVecBGlo[i], lError, lGlobalCor, fProcPnt[i]);
    } else {
      printf("%4d (%4d)\t %+.6f\t %+.6f\t %+.6f\t %.6f\t OFF\t OFF\t %6d\n", i, i0, fInitPar[i], fInitPar[i] + fDeltaPar[i],
             fDeltaPar[i], fVecBGlo[i], fProcPnt[i]);
    }
  }
  return 1;
}

//_____________________________________________________________________________
bool MillePede2::IsRecordAcceptable()
{
  static long prevRunID = kMaxInt;
  static bool prevAns = true;
  long runID = fRecord->GetRunID();
  if (runID != prevRunID) {
    int n = 0;
    fRunWgh = 1.;
    prevRunID = runID;
    // is run to be rejected?
    if (fRejRunList && (n = fRejRunList->GetSize())) {
      prevAns = true;
      for (int i = n; i--;) {
        if (runID == (*fRejRunList)[i]) {
          prevAns = false;
          LOGF(info, "MillePede2 - New Run to reject: %ld", runID);
          break;
        }
      }
    } else if (fAccRunList && (n = fAccRunList->GetSize())) { // is run specifically selected
      prevAns = false;
      for (int i = n; i--;) {
        if (runID == (*fAccRunList)[i]) {
          prevAns = true;
          if (fAccRunListWgh) {
            fRunWgh = (*fAccRunListWgh)[i];
          }
          LOGF(info, "MillePede2 - New Run to accept explicitly: %ld, weight=%f", runID, fRunWgh);
          break;
        }
      }
      if (!prevAns)
        LOGF(info, "New Run is not in the list to accept: %ld", runID);
    }
  }

  return prevAns;
}

//_____________________________________________________________________________
void MillePede2::SetRejRunList(const int* runs, const int nruns)
{
  if (fRejRunList) {
    delete fRejRunList;
  }
  fRejRunList = 0;
  if (nruns < 1 || !runs) {
    return;
  }
  fRejRunList = new TArrayL(nruns);
  for (int i = 0; i < nruns; i++) {
    (*fRejRunList)[i] = runs[i];
  }
}

//_____________________________________________________________________________
void MillePede2::SetAccRunList(const int* runs, const int nruns, const float* wghList)
{
  if (fAccRunList) {
    delete fAccRunList;
  }
  if (fAccRunListWgh) {
    delete fAccRunListWgh;
  }
  fAccRunList = 0;
  if (nruns < 1 || !runs) {
    return;
  }
  fAccRunList = new TArrayL(nruns);
  fAccRunListWgh = new TArrayF(nruns);
  for (int i = 0; i < nruns; i++) {
    (*fAccRunList)[i] = runs[i];
    (*fAccRunListWgh)[i] = wghList ? wghList[i] : 1.0;
  }
}

//_____________________________________________________________________________
void MillePede2::SetInitPars(const double* par)
{
  for (int i = 0; i < fNGloParIni; i++) {
    int id = GetRGId(i);
    if (id < 0) {
      continue;
    }
    fInitPar[id] = par[i];
  }
}

//_____________________________________________________________________________
void MillePede2::SetSigmaPars(const double* par)
{
  for (int i = 0; i < fNGloParIni; i++) {
    int id = GetRGId(i);
    if (id < 0) {
      continue;
    }
    fSigmaPar[id] = par[i];
  }
}

//_____________________________________________________________________________
void MillePede2::SetInitPar(int i, double par)
{
  int id = GetRGId(i);
  if (id < 0) {
    return;
  }
  fInitPar[id] = par;
}

//_____________________________________________________________________________
void MillePede2::SetSigmaPar(int i, double par)
{
  int id = GetRGId(i);
  if (id < 0) {
    return;
  }
  fSigmaPar[id] = par;
}
