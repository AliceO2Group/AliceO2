#include "MCHAlign/MillePede2.h"
#include "Framework/Logger.h"
#include <TStopwatch.h>
#include <TFile.h>
#include <TChain.h>
#include <TMath.h>
#include <TVectorD.h>
#include <TArrayL.h>
#include <TArrayF.h>
#include <TSystem.h>
#include "MCHAlign/MatrixSq.h"
#include "MCHAlign/SymMatrix.h"
#include "MCHAlign/RectMatrix.h"
#include "MCHAlign/MatrixSparse.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <fstream>

// #define _DUMP_EQ_BEFORE_
// #define _DUMP_EQ_AFTER_

// #define _DUMPEQ_BEFORE_
// #define _DUMPEQ_AFTER_

using std::ifstream;
using namespace o2::mch;

ClassImp(MillePede2);

bool MillePede2::fgInvChol = true;                   // Invert global matrix with Cholesky solver
bool MillePede2::fgWeightSigma = true;               // weight local constraint by module statistics
bool MillePede2::fgIsMatGloSparse = false;           // use faster dense matrix by default
int MillePede2::fgMinResCondType = 1;                // Jacoby preconditioner by default
double MillePede2::fgMinResTol = 1.e-11;             // default tolerance
int MillePede2::fgMinResMaxIter = 10000;             // default max number of iterations
int MillePede2::fgIterSol = MinResSolve::kSolMinRes; // default iterative solver
int MillePede2::fgNKrylovV = 240;                    // default number of Krylov vectors to keep

//_____________________________________________________________________________________________
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
    fParamGrID(nullptr),
    fProcPnt(nullptr),
    fVecBLoc(nullptr),
    fDiagCGlo(nullptr),
    fVecBGlo(nullptr),
    fInitPar(nullptr),
    fDeltaPar(nullptr),
    fSigmaPar(nullptr),
    fIsLinear(nullptr),
    fConstrUsed(nullptr),
    fGlo2CGlo(nullptr),
    fCGlo2Glo(nullptr),
    fMatCLoc(nullptr),
    fMatCGlo(nullptr),
    fMatCGloLoc(nullptr),
    fFillIndex(nullptr),
    fFillValue(nullptr),
    fRecDataTreeName("MillePedeRecords_Data"),
    fRecConsTreeName("MillePedeRecords_Consaints"),
    fRecDataBranchName("Record_Data"),
    fRecConsBranchName("Record_Consaints"),
    fDataRecFName("/tmp/mp2_data_records.root"),
    fRecord(nullptr),
    fDataRecFile(nullptr),
    fTreeData(nullptr),
    fRecFileStatus(0),
    fConstrRecFName("/tmp/mp2_constraints_records.root"),
    fTreeConstr(nullptr),
    fConsRecFile(nullptr),
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
    fkReGroup(nullptr)
{
  fWghScl[0] = fWghScl[1] = -1;
}

//_____________________________________________________________________________________________
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
    fGloSolveStatus(0),
    fChi2CutFactor(0),
    fChi2CutRef(0),
    fResCutInit(0),
    fResCut(0),
    fMinPntValid(1),
    fNGroupsSet(0),
    fParamGrID(nullptr),
    fProcPnt(nullptr),
    fVecBLoc(nullptr),
    fDiagCGlo(nullptr),
    fVecBGlo(nullptr),
    fInitPar(nullptr),
    fDeltaPar(nullptr),
    fSigmaPar(nullptr),
    fIsLinear(nullptr),
    fConstrUsed(nullptr),
    fGlo2CGlo(nullptr),
    fCGlo2Glo(nullptr),
    fMatCLoc(nullptr),
    fMatCGlo(nullptr),
    fMatCGloLoc(nullptr),
    fFillIndex(nullptr),
    fFillValue(nullptr),
    fRecDataTreeName(0),
    fRecConsTreeName(0),
    fRecDataBranchName(0),
    fRecConsBranchName(0),
    fDataRecFName(0),
    fRecord(nullptr),
    fDataRecFile(nullptr),
    fTreeData(nullptr),
    fRecFileStatus(0),
    fConstrRecFName(0),
    fTreeConstr(nullptr),
    fConsRecFile(nullptr),
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
    fkReGroup(0)
{
  fWghScl[0] = src.fWghScl[0];
  fWghScl[1] = src.fWghScl[1];
  printf("Dummy\n");
}

//_____________________________________________________________________________________________
MillePede2::~MillePede2()
{
  // destructor
  CloseDataRecStorage();
  CloseConsRecStorage();

  if (fParamGrID) {
    delete[] fParamGrID;
  }
  if (fProcPnt) {
    delete[] fProcPnt;
  }
  if (fVecBLoc) {
    delete[] fVecBLoc;
  }
  if (fDiagCGlo) {
    delete[] fDiagCGlo;
  }
  if (fVecBGlo) {
    delete[] fVecBGlo;
  }
  if (fInitPar) {
    delete[] fInitPar;
  }
  if (fDeltaPar) {
    delete[] fDeltaPar;
  }
  if (fSigmaPar) {
    delete[] fSigmaPar;
  }
  if (fGlo2CGlo) {
    delete[] fGlo2CGlo;
  }
  if (fCGlo2Glo) {
    delete[] fCGlo2Glo;
  }
  if (fIsLinear) {
    delete[] fIsLinear;
  }
  if (fConstrUsed) {
    delete[] fConstrUsed;
  }
  if (fFillIndex) {
    delete[] fFillIndex;
  }
  if (fFillValue) {
    delete[] fFillValue;
  }

  if (fRecord) {
    delete fRecord;
  }
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
}

//_____________________________________________________________________________________________
int MillePede2::InitMille(int nGlo, int nLoc,
                          const int lNStdDev, const double lResCut,
                          const double lResCutInit, const int* regroup)
{
  // init all
  fNGloParIni = nGlo;
  if (regroup) { // regrouping is requested
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
    LOG(info) << Form("Regrouping is requested: from %d raw to %d formal globals grouped to %d real globals", nGlo, ng, maxPID);
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
  LOG(info) << Form("NLoc: %d NGlo: %d", fNLocPar, fNGloPar);

  fNGloSize = fNGloPar;

  if (fgIsMatGloSparse) {
    fMatCGlo = new MatrixSparse(fNGloPar);
    fMatCGlo->SetSymmetric(true);
  } else {
    fMatCGlo = new SymMatrix(fNGloPar);
  }

  fFillIndex = new int[fNGloPar];
  fFillValue = new double[fNGloPar];

  fMatCLoc = new SymMatrix(fNLocPar);
  fMatCGloLoc = new RectMatrix(fNGloPar, fNLocPar);

  fParamGrID = new int[fNGloPar];
  fProcPnt = new int[fNGloPar];
  fVecBLoc = new double[fNLocPar];
  fDiagCGlo = new double[fNGloPar];

  fInitPar = new double[fNGloPar];
  fDeltaPar = new double[fNGloPar];
  fSigmaPar = new double[fNGloPar];
  fIsLinear = new bool[fNGloPar];

  fGlo2CGlo = new int[fNGloPar];
  fCGlo2Glo = new int[fNGloPar];

  memset(fVecBLoc, 0, fNLocPar * sizeof(double));
  memset(fDiagCGlo, 0, fNGloPar * sizeof(double));
  memset(fInitPar, 0, fNGloPar * sizeof(double));
  memset(fDeltaPar, 0, fNGloPar * sizeof(double));
  memset(fSigmaPar, 0, fNGloPar * sizeof(double));
  memset(fProcPnt, 0, fNGloPar * sizeof(int));

  for (int i = fNGloPar; i--;) {
    fGlo2CGlo[i] = fCGlo2Glo[i] = -1;
    fIsLinear[i] = true;
    fParamGrID[i] = -1;
  }

  fWghScl[0] = -1;
  fWghScl[1] = -1;
  return 1;
}

//_____________________________________________________________________________________________
bool MillePede2::ImposeDataRecFile(const char* fname)
{
  // set filename for records
  CloseDataRecStorage();
  SetDataRecFName(fname);
  return InitDataRecStorage(true); // open in read mode
}

//_____________________________________________________________________________________________
bool MillePede2::ImposeConsRecFile(const char* fname)
{
  // set filename for constraints
  CloseConsRecStorage();
  SetConsRecFName(fname);
  return InitConsRecStorage(true); // open in read mode
}

//_____________________________________________________________________________________________
bool MillePede2::InitDataRecStorage(bool read)
{
  // initialize the buffer for processed measurements records
  //
  if (fTreeData) {
    LOG(info) << "Data Records File is already initialized";
    return false;
  }
  //
  if (!fRecord) {
    fRecord = new MillePedeRecord();
  }

  if (!read) { // write mode: cannot use chain
    fDataRecFile = TFile::Open(GetDataRecFName(), "recreate");
    if (!fDataRecFile) {
      LOG(fatal) << Form("Failed to initialize data records file %s", GetDataRecFName());
      return false;
    }
    LOG(info) << Form("File %s used for derivatives records", GetDataRecFName());
    fTreeData = new TTree(GetRecDataTreeName(), "Data Records for MillePede2");
    fTreeData->Branch(GetRecDataBranchName(), "MillePedeRecord", &fRecord, 32000, 99);
  } else { // use chain
    TChain* ch = new TChain(GetRecDataTreeName());
    //
    if (fDataRecFName.EndsWith(".root")) {
      ch->AddFile(fDataRecFName);
    } else { // assume text file with list of filenames
      //
      ifstream inpf(fDataRecFName.Data());
      if (!inpf.good()) {
        LOG(info) << Form("Failed on input records list %s\n", fDataRecFName.Data());
        return false;
      }
      //
      TString recfName;
      while (!(recfName.ReadLine(inpf)).eof()) {
        recfName = recfName.Strip(TString::kBoth, ' ');
        if (recfName.BeginsWith("//") || recfName.BeginsWith("#") || !recfName.EndsWith(".root")) { // comment
          LOG(info) << Form("Skip %s\n", recfName.Data());
          continue;
        }
        //
        recfName = recfName.Strip(TString::kBoth, ',');
        recfName = recfName.Strip(TString::kBoth, '"');
        gSystem->ExpandPathName(recfName);
        printf("Adding %s\n", recfName.Data());
        ch->AddFile(recfName.Data());
      }
    }
    //
    Long64_t nent = ch->GetEntries();
    if (nent < 1) {
      LOG(info) << "Obtained chain is empty";
      return false;
    }
    fTreeData = ch;
    fTreeData->SetBranchAddress(GetRecDataBranchName(), &fRecord);
    LOG(info) << Form("Found %lld derivatives records", nent);
  }
  fCurrRecDataID = -1;
  fRecFileStatus = read ? 1 : 2;
  //
  return true;
}

//_____________________________________________________________________________________________
bool MillePede2::InitConsRecStorage(bool read)
{
  // initialize the buffer for processed measurements records
  //
  if (fConsRecFile) {
    LOG(info) << "Constraints Records File is already initialized";
    return false;
  }
  //
  if (!fRecord) {
    fRecord = new MillePedeRecord();
  }

  fConsRecFile = TFile::Open(GetConsRecFName(), read ? "" : "recreate");
  if (!fConsRecFile) {
    LOG(info) << Form("Failed to initialize constraints records file %s", GetConsRecFName());
    return false;
  }
  //
  LOG(info) << Form("File %s used for constraints records", GetConsRecFName());
  if (read) {
    fTreeConstr = (TTree*)fConsRecFile->Get(GetRecConsTreeName());
    if (!fTreeConstr) {
      LOG(info) << Form("Did not find constraints records tree in %s", GetConsRecFName());
      return false;
    }
    fTreeConstr->SetBranchAddress(GetRecConsBranchName(), &fRecord);
    LOG(info) << Form("Found %lld constraints records", fTreeConstr->GetEntries());
    //
  } else {
    //
    fTreeConstr = new TTree(GetRecConsTreeName(), "Constraints Records for MillePede2");
    fTreeConstr->Branch(GetRecConsBranchName(), "MillePedeRecord", &fRecord, 32000, 99);
  }
  fCurrRecConstrID = -1;
  //
  return true;
}

//_____________________________________________________________________________________________
void MillePede2::CloseDataRecStorage()
{
  // close records file
  // LOG(info) << "I'm closing mille recording...";
  if (fTreeData) {
    if (fDataRecFile && fDataRecFile->IsWritable()) {
      // LOG(info) << "Wtriting file";
      fDataRecFile->cd();
      fTreeData->Write();
    }
    delete fTreeData;
    fTreeData = 0;
    if (fDataRecFile) {
      fDataRecFile->Close();
      delete fDataRecFile;
      fDataRecFile = 0;
    }
  }
  fRecFileStatus = 0;
}

//_____________________________________________________________________________________________
void MillePede2::CloseConsRecStorage()
{
  // close constraints file
  if (fTreeConstr) {
    if (fConsRecFile->IsWritable()) {
      fConsRecFile->cd();
      fTreeConstr->Write();
    }
    delete fTreeConstr;
    fTreeConstr = 0;
    fConsRecFile->Close();
    delete fConsRecFile;
    fConsRecFile = 0;
  }
  //
}

//_____________________________________________________________________________________________
bool MillePede2::ReadNextRecordData()
{
  // read next data record (if any)
  if (!fTreeData || ++fCurrRecDataID >= fTreeData->GetEntries()) {
    fCurrRecDataID--;
    return false;
  }
  fTreeData->GetEntry(fCurrRecDataID);
  return true;
}

//_____________________________________________________________________________________________
bool MillePede2::ReadNextRecordConstraint()
{
  // read next constraint record (if any)
  if (!fTreeConstr || ++fCurrRecConstrID >= fTreeConstr->GetEntries()) {
    fCurrRecConstrID--;
    return false;
  }
  fTreeConstr->GetEntry(fCurrRecConstrID);
  return true;
}

//_____________________________________________________________________________________________
void MillePede2::SetRecordWeight(double wgh)
{
  // assign weight
  if (fRecFileStatus < 2) {
    InitDataRecStorage(); // create a buffer to store the data
  }
  fRecord->SetWeight(wgh);
}

//_____________________________________________________________________________________________
void MillePede2::SetRecordRun(int run)
{
  // assign run
  if (fRecFileStatus < 2) {
    InitDataRecStorage(); // create a buffer to store the data
  }
  fRecord->SetRunID(run);
}

//_____________________________________________________________________________________________
void MillePede2::SetLocalEquation(double* dergb, double* derlc,
                                  const double lMeas, const double lSigma)
{
  // assing derivs of loc.eq.
  if (fRecFileStatus < 2) {
    InitDataRecStorage(); // create a buffer to store the data
  }
  //
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
  for (int i = 0; i < fNLocPar; i++)
    if (!IsZero(derlc[i])) {
      fRecord->AddIndexValue(i, derlc[i]);
      derlc[i] = 0.0;
    }

  fRecord->AddWeight(1.0 / lSigma / lSigma);

  // Idem for global parameters
  for (int i = 0; i < fNGloParIni; i++)
    if (!IsZero(dergb[i])) {
      fRecord->AddIndexValue(i, dergb[i]);
      dergb[i] = 0.0;
      int idrg = GetRGId(i);
      fRecord->MarkGroup(idrg < 0 ? -1 : fParamGrID[i]);
    }
}

//_____________________________________________________________________________________________
void MillePede2::SetLocalEquation(int* indgb, double* dergb,
                                  int ngb, int* indlc,
                                  double* derlc, const int nlc,
                                  const double lMeas, const double lSigma)
{
  // write data of single measurement. Note: the records ignore regrouping, store direct parameters
  if (lSigma <= 0.0) { // If parameter is fixed, then no equation
    for (int i = nlc; i--;) {
      derlc[i] = 0.0;
    }
    for (int i = ngb; i--;) {
      dergb[i] = 0.0;
    }
    return;
  }

  if (fRecFileStatus < 2) {
    InitDataRecStorage(); // create a buffer to store the data
  }

  fRecord->AddResidual(lMeas);

  // Retrieve local param interesting indices
  for (int i = 0; i < nlc; i++)
    if (!IsZero(derlc[i])) {
      fRecord->AddIndexValue(indlc[i], derlc[i]);
      derlc[i] = 0.;
      indlc[i] = 0;
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

//_____________________________________________________________________________________________
void MillePede2::SetGlobalConstraint(const double* dergb, const double val, const double sigma)
{
  // Define a constraint equation.
  if (!fConsRecFile || !fConsRecFile->IsWritable()) {
    InitConsRecStorage(); // create a buffer to store the data
  }

  fRecord->Reset();
  fRecord->AddResidual(val);
  fRecord->AddWeight(sigma);
  for (int i = 0; i < fNGloParIni; i++) {
    if (!IsZero(dergb[i]))
      fRecord->AddIndexValue(i, dergb[i]);
  }
  fNGloConstraints++;
  if (IsZero(sigma)) {
    fNLagrangeConstraints++;
  }
  //  printf("NewConstraint:\n"); fRecord->Print(); //RRR
  SaveRecordConstraint();
  //
}

//_____________________________________________________________________________________________
void MillePede2::SetGlobalConstraint(const int* indgb,
                                     const double* dergb,
                                     const int ngb,
                                     const double val,
                                     const double sigma)
{
  // Define a constraint equation.
  if (!fConsRecFile || !fConsRecFile->IsWritable()) {
    InitConsRecStorage(); // create a buffer to store the data
  }
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
  SaveRecordConstraint();
  //
}

//_____________________________________________________________________________________________
int MillePede2::LocalFit(double* localParams)
{
  /*
    Perform local parameters fit once all the local equations have been set
    -----------------------------------------------------------
    localParams = (if !=0) will contain the fitted track parameters and
    related errors
  */
  static int nrefSize = 0;
  //  static TArrayI refLoc,refGlo,nrefLoc,nrefGlo;
  static int *refLoc = 0, *refGlo = 0, *nrefLoc = 0, *nrefGlo = 0;
  int nPoints = 0;
  //
  SymMatrix& matCLoc = *fMatCLoc;
  MatrixSq& matCGlo = *fMatCGlo;
  RectMatrix& matCGloLoc = *fMatCGloLoc;
  //
  memset(fVecBLoc, 0, fNLocPar * sizeof(double));
  matCLoc.Reset();
  //
  int cnt = 0;
  int recSz = fRecord->GetSize();
  //
  while (cnt < recSz) { // Transfer the measurement records to matrices
    //
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
    //
    refLoc[nPoints] = ++cnt;
    int nLoc = 0;
    while (!fRecord->IsWeight(cnt)) {
      nLoc++;
      cnt++;
    }
    nrefLoc[nPoints] = nLoc;
    //
    refGlo[nPoints] = ++cnt;
    int nGlo = 0;
    while (!fRecord->IsResidual(cnt) && cnt < recSz) {
      nGlo++;
      cnt++;
    }
    nrefGlo[nPoints] = nGlo;
    //
    nPoints++;
  }
  if (fMinRecordLength > 0 && nPoints < fMinRecordLength) {
    return 0; // ignore
  }
  //
  double vl;
  //
  double gloWgh = fRunWgh;
  if (fUseRecordWeight) {
    gloWgh *= fRecord->GetWeight(); // global weight for this set
  }
  int maxLocUsed = 0;
  //
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
    //
    for (int i = nrefGlo[ip]; i--;) { // suppress the global part (only relevant with iterations)
      //
      // if regrouping was requested, do it here
      if (fkReGroup) {
        int idtmp = fkReGroup[indGlo[i]];
        if (idtmp == kFixParID) {
          indGlo[i] = kFixParID; // fixed param in regrouping
        } else {
          indGlo[i] = idtmp;
        }
      }
      //
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
    //
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
    //
  } // end of the transfer of the measurement record to matrices
  //
  matCLoc.SetSizeUsed(++maxLocUsed); // data with B=0 may use less than declared nLocals
  //
  /* //RRR
  fRecord->Print("l");
  printf("\nBefore\nLocalMatrix: "); matCLoc.Print("l");
  printf("RHSLoc: "); for (int i=0;i<fNLocPar;i++) printf("%+e |",fVecBLoc[i]); printf("\n");
  */
  // first try to solve by faster Cholesky decomposition, then by Gaussian elimination
  if (!matCLoc.SolveChol(fVecBLoc, true)) {
    LOG(info) << "Failed to solve locals by Cholesky, trying Gaussian Elimination";
    if (!matCLoc.SolveSpmInv(fVecBLoc, true)) {
      LOG(info) << "Failed to solve locals by Gaussian Elimination, skip...";
      matCLoc.Print("d");
      return 0; // failed to solve
    }
  }
  //
  // If requested, store the track params and errors
  // RRR  printf("locfit: "); for (int i=0;i<fNLocPar;i++) printf("%+e |",fVecBLoc[i]); printf("\n");

  if (localParams) {
    for (int i = maxLocUsed; i--;) {
      localParams[2 * i] = fVecBLoc[i];
      localParams[2 * i + 1] = TMath::Sqrt(TMath::Abs(matCLoc.QueryDiag(i)));
    }
  }
  //
  float lChi2 = 0;
  int nEq = 0;
  //
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
    //
    // Suppress local and global contribution in residuals;
    for (int i = nrefLoc[ip]; i--;) {
      resid -= derLoc[i] * fVecBLoc[indLoc[i]]; // local part
    }
    //
    for (int i = nrefGlo[ip]; i--;) { // global part
      int iID = indGlo[i];
      if (iID < 0 || fSigmaPar[iID] <= 0.) {
        continue; // fixed parameter RRRCheck
      }
      if (fIsLinear[iID]) {
        resid -= derGlo[i] * (fInitPar[iID] + fDeltaPar[iID]); // linear parameter
      } else {
        resid -= derGlo[i] * fDeltaPar[iID]; // nonlinear parameter
      }
    }
    //
    // reject the track if the residual is too large (outlier)
    double absres = TMath::Abs(resid);
    if ((absres >= fResCutInit && fIter == 1) ||
        (absres >= fResCut && fIter > 1)) {
      if (fLocFitAdd) {
        fNLocFitsRejected++;
      }
      //      printf("reject res %5ld %+e\n",fCurrRecDataID,resid);
      return 0;
    }
    //
    lChi2 += weight * resid * resid; // total chi^2
    nEq++;                           // number of equations
  }                                  // end of Calculate residuals
  //
  lChi2 /= gloWgh;
  int nDoF = nEq - maxLocUsed;
  lChi2 = (nDoF > 0) ? lChi2 / nDoF : 0; // Chi^2/dof
  //
  if (fNStdDev != 0 && nDoF > 0 && lChi2 > Chi2DoFLim(fNStdDev, nDoF) * fChi2CutFactor) { // check final chi2
    if (fLocFitAdd) {
      fNLocFitsRejected++;
    }
    //    printf("reject chi2 %5ld: %+e\n",fCurrRecDataID, lChi2);
    return 0;
  }
  //
  if (fLocFitAdd) {
    fNLocFits++;
    fNLocEquations += nEq;
  } else {
    fNLocFits--;
    fNLocEquations -= nEq;
  }
  //
  //  local operations are finished, track is accepted
  //  We now update the global parameters (other matrices)
  //
  int nGloInFit = 0;
  //
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
    //
    for (int i = nrefGlo[ip]; i--;) { // suppress the global part
      int iID = indGlo[i];            // Global param indice
      if (iID < 0 || fSigmaPar[iID] <= 0.) {
        continue; // fixed parameter RRRCheck
      }
      if (fIsLinear[iID]) {
        resid -= derGlo[i] * (fInitPar[iID] + fDeltaPar[iID]); // linear parameter
      } else {
        resid -= derGlo[i] * fDeltaPar[iID]; // nonlinear parameter
      }
    }
    //
    for (int ig = nrefGlo[ip]; ig--;) {
      int iIDg = indGlo[ig]; // Global param indice (the matrix line)
      if (iIDg < 0 || fSigmaPar[iIDg] <= 0.) {
        continue; // fixed parameter RRRCheck
      }
      if (fLocFitAdd) {
        fVecBGlo[iIDg] += weight * resid * derGlo[ig]; //!!!
      } else {
        fVecBGlo[iIDg] -= weight * resid * derGlo[ig]; //!!!
      }
      //
      // First of all, the global/global terms (exactly like local matrix)
      int nfill = 0;
      for (int jg = ig + 1; jg--;) { // matCGlo is symmetric by construction
        int jIDg = indGlo[jg];
        if (jIDg < 0 || fSigmaPar[jIDg] <= 0.) {
          continue; // fixed parameter RRRCheck
        }
        if (!IsZero(vl = weight * derGlo[ig] * derGlo[jg])) {
          fFillIndex[nfill] = jIDg;
          fFillValue[nfill++] = fLocFitAdd ? vl : -vl;
        }
      }
      if (nfill)
        matCGlo.AddToRow(iIDg, fFillValue, fFillIndex, nfill);
      //
      // Now we have also rectangular matrices containing global/local terms.
      int iCIDg = fGlo2CGlo[iIDg]; // compressed Index of index
      if (iCIDg == -1) {
        double* rowGL = matCGloLoc(nGloInFit);
        for (int k = maxLocUsed; k--;) {
          rowGL[k] = 0.0; // reset the row
        }
        iCIDg = fGlo2CGlo[iIDg] = nGloInFit;
        fCGlo2Glo[nGloInFit++] = iIDg;
      }
      //
      double* rowGLIDg = matCGloLoc(iCIDg);
      for (int il = nrefLoc[ip]; il--;) {
        rowGLIDg[indLoc[il]] += weight * derGlo[ig] * derLoc[il];
      }
      fProcPnt[iIDg] += fLocFitAdd ? 1 : -1; // update counter
      //
    }
  } // end of Update matrices
  //
  /*//RRR
  printf("After GLO\n");
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
    //
    vl = 0;
    double* rowGLIDg = matCGloLoc(iCIDg);
    for (int kl = 0; kl < maxLocUsed; kl++)
      if (rowGLIDg[kl]) {
        vl += rowGLIDg[kl] * fVecBLoc[kl];
      }
    if (!IsZero(vl)) {
      fVecBGlo[iIDg] -= fLocFitAdd ? vl : -vl;
    }
    //
    int nfill = 0;
    for (int jCIDg = 0; jCIDg <= iCIDg; jCIDg++) {
      int jIDg = fCGlo2Glo[jCIDg];
      //
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
      matCGlo.AddToRow(iIDg, fFillValue, fFillIndex, nfill);
    }
  }
  //
  // reset compressed index array
  //
  /*//RRR
  printf("After GLOLoc\n");
  printf("MatCGlo: "); fMatCGlo->Print("");
  printf("BGlo: "); for (int i=0; i<fNGloPar; i++) printf("%+e |",fVecBGlo[i]); printf("\n");
  */
  for (int i = nGloInFit; i--;) {
    fGlo2CGlo[fCGlo2Glo[i]] = -1;
    fCGlo2Glo[i] = -1;
  }
  //
  //---------------------------------------------------- <<<
  return 1;
}

//_____________________________________________________________________________________________
int MillePede2::GlobalFit(double* par, double* error, double* pull)
{
  // performs a requested number of global iterations
  fIter = 1;
  //
  TStopwatch sw;
  sw.Start();
  //
  int res = 0;
  LOG(info) << "Starting Global fit.";
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
        // RRR	fIter = fMaxIter - 1;     // Last iteration
      }
    }
    fIter++;
  }
  //
  sw.Stop();
  // LOG(info) << Form("res is: %i", res);
  LOG(info) << Form("Global fit %s, CPU time: %.1f", res ? "Converged" : "Failed", sw.CpuTime());
  if (!res) {
    return 0;
  }
  //
  if (par) {
    LOG(info) << "Filling params";
    for (int i = fNGloParIni; i--;) {
      par[i] = GetFinalParam(i);
    }
  }
  //
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
  //
  return 1;
}

//_____________________________________________________________________________________________
int MillePede2::GlobalFitIteration()
{
  // perform global parameters fit once all the local equations have been fitted
  //
  LOG(info) << Form("Global Fit Iteration#%2d (Local Fit Chi^2 cut factor: %.2f)", fIter, fChi2CutFactor);
  //
  if (!fNGloPar || !fTreeData) {
    LOG(info) << "No data was stored, stopping iteration";
    return 0;
  }
  TStopwatch sw, sws;
  sw.Start();
  sws.Stop();
  //
  if (!fConstrUsed) {
    fConstrUsed = new bool[fNGloConstraints];
    memset(fConstrUsed, 0, fNGloConstraints * sizeof(bool));
  }
  // Reset all info specific for this step
  MatrixSq& matCGlo = *fMatCGlo;
  matCGlo.Reset();
  memset(fProcPnt, 0, fNGloPar * sizeof(int));
  //
  fNGloConstraints = fTreeConstr ? fTreeConstr->GetEntries() : 0;
  //
  // count number of Lagrange constraints: they need new row/cols to be added
  fNLagrangeConstraints = 0;
  for (int i = 0; i < fNGloConstraints; i++) {
    ReadRecordConstraint(i);
    if (IsZero(fRecord->GetValue(1))) {
      fNLagrangeConstraints++; // exact constraint (no error) -> Lagrange multiplier
    }
  }
  //
  // if needed, readjust the size of the global vector (for matrices this is done automatically)
  if (!fVecBGlo || fNGloSize != fNGloPar + fNLagrangeConstraints) {
    delete[] fVecBGlo; // in case some constraint was added between the two manual iterations
    fNGloSize = fNGloPar + fNLagrangeConstraints;
    fVecBGlo = new double[fNGloSize];
  }
  memset(fVecBGlo, 0, fNGloSize * sizeof(double));
  //
  fNLocFits = 0;
  fNLocFitsRejected = 0;
  fNLocEquations = 0;
  //
  //  Process data records and build the matrices
  Long_t ndr = fTreeData->GetEntries();
  Long_t first = fSelFirst > 0 ? fSelFirst : 0;
  Long_t last = fSelLast < 1 ? ndr : (fSelLast >= ndr ? ndr : fSelLast + Long_t(1));
  ndr = last - first;
  //
  LOG(info) << Form("Building the Global matrix from data records %ld : %ld", first, last);
  if (ndr < 1) {
    return 0;
  }
  //
  TStopwatch swt;
  swt.Start();
  fLocFitAdd = true; // add contributions of matching tracks
  for (Long_t i = 0; i < ndr; i++) {
    // printf("Reading RecordData %ld\n", i);
    Long_t iev = i + first;
    ReadRecordData(iev);
    // printf("Read RecordData %ld\n", iev);
    if (!IsRecordAcceptable()) {
      continue;
    }
    LocalFit();
    if ((i % int(0.2 * ndr)) == 0) {
      printf("%.1f%% of local fits done\n", double(100. * i) / ndr);
    }
  }
  swt.Stop();
  printf("%ld local fits done: ", ndr);
  /*
  printf("MatCGlo: "); fMatCGlo->Print("l");
  printf("BGlo: "); for (int i=0; i<fNGloPar; i++) printf("%+e |",fVecBGlo[i]); printf("\n");
  swt.Print();
  */
  sw.Start(false);
  //
  //
  // ---------------------- Reject parameters with low statistics ------------>>
  fNGloFix = 0;
  if (fMinPntValid > 1 && fNGroupsSet) {
    //
    printf("Checking parameters with statistics < %d\n", fMinPntValid);
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
    //
    for (int i = fNGloPar; i--;) { // // Reset row and column of fixed params and add 1/sig^2 to free ones
      int grID = fParamGrID[i];
      if (grID < 0) {
        continue; // not in the group
      }
      //
      if (grID != grIDold) {                                        // starting new group
        if (grIDold >= 0) {                                         // decide if the group has enough statistics
          if (oldMin < fMinPntValid && oldMax < 2 * fMinPntValid) { // suppress group
            for (int iold = oldStart; iold > i; iold--) {
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
      //
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
    //
    // 2) loop over records and add contributions of fixed groups with negative sign
    fLocFitAdd = false;
    //
    for (Long_t i = 0; i < ndr; i++) {
      Long_t iev = i + first;
      ReadRecordData(iev);
      if (!IsRecordAcceptable()) {
        continue;
      }
      bool suppr = false;
      for (int ifx = nFixedGroups; ifx--;) {
        if (fRecord->IsGroupPresent(fixGroups[ifx])) {
          suppr = true;
        }
      }
      if (suppr) {
        LocalFit();
      }
    }
    fLocFitAdd = true;
    //
    if (nFixedGroups) {
      printf("Suppressed contributions of groups with NPoints<%d :\n", fMinPntValid);
      for (int i = 0; i < nFixedGroups; i++) {
        printf("%d ", fixGroups[i]);
      }
      printf("\n");
    }
    swsup.Stop();
    swsup.Print();
  }
  // ---------------------- Reject parameters with low statistics ------------<<
  //
  // add large number to diagonal of fixed params
  //
  for (int i = fNGloPar; i--;) { // // Reset row and column of fixed params and add 1/sig^2 to free ones
    //    printf("#%3d : Nproc : %5d   grp: %d\n",i,fProcPnt[i],fParamGrID[i]);
    if (fProcPnt[i] < 1) {
      fNGloFix++;
      fVecBGlo[i] = 0.;
      matCGlo.DiagElem(i) = 1.; // float(fNLocEquations*fNLocEquations);
      //      matCGlo.DiagElem(i) = float(fNLocEquations*fNLocEquations);
    } else {
      matCGlo.DiagElem(i) += (fgWeightSigma ? fProcPnt[i] : 1.) / (fSigmaPar[i] * fSigmaPar[i]);
    }
  }
  //
  for (int i = fNGloPar; i--;) {
    fDiagCGlo[i] = matCGlo.QueryDiag(i); // save the diagonal elements
  }
  //
  // add constraint equations
  int nVar = fNGloPar; // Current size of global matrix
  for (int i = 0; i < fNGloConstraints; i++) {
    ReadRecordConstraint(i);
    double val = fRecord->GetValue(0);
    double sig = fRecord->GetValue(1);
    int* indV = fRecord->GetIndex() + 2;
    double* der = fRecord->GetValue() + 2;
    int csize = fRecord->GetSize() - 2;
    //
    if (fkReGroup) {
      for (int jp = csize; jp--;) {
        int idp = indV[jp];
        if (fkReGroup[idp] < 0) {
          LOG(fatal) << Form("Constain is requested for suppressed parameter #%d", indV[jp]);
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
    //
    if (nSuppressed == csize) {
      // was this constraint ever created ?
      if (sig == 0 && fConstrUsed[i]) { // this is needed only for constraints with Lagrange multiplier
        // to avoid empty row impose dummy constraint on "Lagrange multiplier"
        matCGlo.DiagElem(nVar) = 1.;
        fVecBGlo[nVar++] = 0;
      }
      continue;
    }
    //
    // account for already accumulated corrections
    for (int j = csize; j--;) {
      val -= der[j] * (fInitPar[indV[j]] + fDeltaPar[indV[j]]);
    }
    //
    if (sig > 0) { // this is a gaussian constriant: no Lagrange multipliers are added
      //
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
        if (fProcPnt[jID] < 1) {
          continue; // this parameter was fixed, don't put it into constraint
        }
        matCGlo(nVar, jID) = float(fNLocEquations) * der[j]; // fMatCGlo is symmetric, only lower triangle is filled
      }
      //
      if (matCGlo.QueryDiag(nVar)) {
        matCGlo.DiagElem(nVar) = 0.0;
      }
      fVecBGlo[nVar++] = float(fNLocEquations) * val; // RS ? should we use here fNLocFits ?
      fConstrUsed[i] = true;
    }
  }
  //
  LOG(info) << Form("Obtained %-7ld equations from %-7ld records (%-7ld rejected). Fixed %-4d globals",
                    fNLocEquations, fNLocFits, fNLocFitsRejected, fNGloFix);

  //
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
    for (int i = 0; i < fNGloSize; i++) {
      printf("b%2d : %+.10f\n", i, fVecBGlo[i]);
    }
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
  LOG(info) << Form("Iteration#%2d %s. CPU time: %.1f", fIter, fGloSolveStatus == kFailed ? "Failed" : "Converged", sw.CpuTime());
  if (fGloSolveStatus == kFailed) {
    return 0;
  }
  //
  for (int i = fNGloPar; i--;) {
    fDeltaPar[i] += fVecBGlo[i]; // Update global parameters values (for iterations)
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

  PrintGlobalParameters();
  return 1;
}

//_____________________________________________________________________________________________
int MillePede2::SolveGlobalMatEq()
{
  // solve global matrix equation MatCGlob*X=VecBGlo and store the result in the VecBGlo

  if (!fgIsMatGloSparse) {
    //
    if (fNLagrangeConstraints == 0) { // pos-def systems are faster to solve by Cholesky
      if (((SymMatrix*)fMatCGlo)->SolveChol(fVecBGlo, fgInvChol)) {
        return fgInvChol ? kInvert : kNoInversion;
      } else {
        LOG(info) << "Solution of Global Dense System by Cholesky failed, trying Gaussian Elimiation";
      }
    }
    //
    if (((SymMatrix*)fMatCGlo)->SolveSpmInv(fVecBGlo, true)) {
      return kInvert;
    } else {
      LOG(info) << "Solution of Global Dense System by Gaussian Elimination failed, trying iterative methods";
    }
  }
  // try to solve by minres
  TVectorD sol(fNGloSize);
  //
  MinResSolve* slv = new MinResSolve(fMatCGlo, fVecBGlo);
  if (!slv) {
    return kFailed;
  }
  //
  bool res = false;
  if (fgIterSol == MinResSolve::kSolMinRes) {
    res = slv->SolveMinRes(sol, fgMinResCondType, fgMinResMaxIter, fgMinResTol);
  } else if (fgIterSol == MinResSolve::kSolFGMRes) {
    res = slv->SolveFGMRES(sol, fgMinResCondType, fgMinResMaxIter, fgMinResTol, fgNKrylovV);
  } else {
    LOG(info) << Form("Undefined Iteritive Solver ID=%d, only %d are defined", fgIterSol, MinResSolve::kNSolvers);
  }
  //
  if (!res) {
    const char* faildump = "fgmr_failed.dat";
    int defout = dup(1);
    if (defout < 0) {
      LOG(info) << "Failed on dup";
      return kFailed;
    }
    int slvDump = open(faildump, O_RDWR | O_CREAT, 0666);
    if (slvDump >= 0) {
      dup2(slvDump, 1);
      //
      printf("#Failed to solve using solver %d with PreCond: %d MaxIter: %d Tol: %e NKrylov: %d\n",
             fgIterSol, fgMinResCondType, fgMinResMaxIter, fgMinResTol, fgNKrylovV);
      printf("#Dump of matrix:\n");
      fMatCGlo->Print("10");
      printf("#Dump of RHS:\n");
      for (int i = 0; i < fNGloSize; i++) {
        printf("%d %+.10f\n", i, fVecBGlo[i]);
      }
      //
      dup2(defout, 1);
      close(slvDump);
      close(defout);
      printf("#Dumped failed matrix and RHS to %s\n", faildump);
    } else {
      LOG(info) << "Failed on file open for matrix dumping";
    }
    close(defout);
    return kFailed;
  }
  for (int i = fNGloSize; i--;) {
    fVecBGlo[i] = sol[i];
  }
  //
  return kNoInversion;
  //
}

//_____________________________________________________________________________________________
float MillePede2::Chi2DoFLim(int nSig, int nDoF) const
{
  /// return the limit in chi^2/nd for n sigmas stdev authorized
  // Only n=1, 2, and 3 are expected in input
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

//_____________________________________________________________________________________________
int MillePede2::SetIterations(double lChi2CutFac)
{
  // Number of iterations is calculated from lChi2CutFac
  fChi2CutFactor = TMath::Max(1.0, lChi2CutFac);
  //
  LOG(info) << Form("Initial cut factor is %f", fChi2CutFactor);
  fIter = 1; // Initializes the iteration process
  return 1;
}

//_____________________________________________________________________________________________
double MillePede2::GetParError(int iPar) const
{
  // return error for parameter iPar
  if (fGloSolveStatus == kInvert) {
    if (fkReGroup) {
      iPar = fkReGroup[iPar];
    }
    if (iPar < 0) {
      return 0;
    }
    double res = fMatCGlo->QueryDiag(iPar);
    if (res >= 0) {
      return TMath::Sqrt(res);
    }
  }
  return 0.;
}

//_____________________________________________________________________________________________
double MillePede2::GetPull(int iPar) const
{
  // return pull for parameter iPar
  if (fGloSolveStatus == kInvert) {
    if (fkReGroup) {
      iPar = fkReGroup[iPar];
    }
    if (iPar < 0) {
      return 0;
    }
    //
    return fProcPnt[iPar] > 0 && (fSigmaPar[iPar] * fSigmaPar[iPar] - fMatCGlo->QueryDiag(iPar)) > 0. && fSigmaPar[iPar] > 0
             ? fDeltaPar[iPar] / TMath::Sqrt(fSigmaPar[iPar] * fSigmaPar[iPar] - fMatCGlo->QueryDiag(iPar))
             : 0;
  }
  return 0.;
}

//_____________________________________________________________________________________________
int MillePede2::PrintGlobalParameters() const
{
  ///  Print the final results into the logfile
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
    if (i != i0 && lastPrintedId >= 0 && i <= lastPrintedId) {
      continue; // grouped param
    }
    lastPrintedId = i;
    lError = GetParError(i0);
    lGlobalCor = 0.0;
    //
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

//_____________________________________________________________________________________________
bool MillePede2::IsRecordAcceptable()
{
  // validate record according run lists set by the user
  static Long_t prevRunID = kMaxInt;
  static bool prevAns = true;
  Long_t runID = fRecord->GetRunID();
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
          LOG(info) << Form("New Run to reject: %ld", runID);
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
          LOG(info) << Form("New Run to accept explicitly: %ld, weight=%f", runID, fRunWgh);
          break;
        }
      }
      if (!prevAns) {
        LOG(info) << Form("New Run is not in the list to accept: %ld", runID);
      }
    }
  }
  //
  return prevAns;
  //
}

//_____________________________________________________________________________________________
void MillePede2::SetRejRunList(const int* runs, const int nruns)
{
  // set the list of runs to be rejected
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

//_____________________________________________________________________________________________
void MillePede2::SetAccRunList(const int* runs, const int nruns, const float* wghList)
{
  // set the list of runs to be selected
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

//_____________________________________________________________________________________________
void MillePede2::SetInitPars(const double* par)
{
  // initialize parameters, account for eventual grouping
  for (int i = 0; i < fNGloParIni; i++) {
    int id = GetRGId(i);
    if (id < 0) {
      continue;
    }
    fInitPar[id] = par[i];
  }
}

//_____________________________________________________________________________________________
void MillePede2::SetSigmaPars(const double* par)
{
  // initialize sigmas, account for eventual grouping
  for (int i = 0; i < fNGloParIni; i++) {
    int id = GetRGId(i);
    if (id < 0) {
      continue;
    }
    fSigmaPar[id] = par[i];
  }
}

//_____________________________________________________________________________________________
void MillePede2::SetInitPar(int i, double par)
{
  // initialize param, account for eventual grouping
  int id = GetRGId(i);
  if (id < 0) {
    return;
  }
  fInitPar[id] = par;
}

//_____________________________________________________________________________________________
void MillePede2::SetSigmaPar(int i, double par)
{
  // initialize sigma, account for eventual grouping
  int id = GetRGId(i);
  if (id < 0) {
    return;
  }
  fSigmaPar[id] = par;
}
