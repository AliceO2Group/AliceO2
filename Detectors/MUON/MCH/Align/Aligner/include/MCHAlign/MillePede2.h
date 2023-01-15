#ifndef ALICEO2_MCH_MILLEPEDE2_H
#define ALICEO2_MCH_MILLEPEDE2_H

#include <TObject.h>
#include <TString.h>
#include <TTree.h>
#include "MCHAlign/MinResSolve.h"
#include "MCHAlign/MillePedeRecord.h"

class TFile;
class TStopwatch;
class TArrayL;
class TArrayF;

namespace o2
{
namespace mch
{

class MatrixSq;
class SymMatrix;
class RectMatrix;
class MatrixSparse;

class MillePede2 : public TObject
{
 public:
  enum { kFailed,
         kInvert,
         kNoInversion };   // used global matrix solution methods
  enum { kFixParID = -1 }; // dummy id for fixed param

  MillePede2();
  MillePede2(const MillePede2& src);
  virtual ~MillePede2();
  MillePede2& operator=(const MillePede2&)
  {
    printf("Dummy\n");
    return *this;
  }

  int InitMille(const int nGlo, const int nLoc,
                const int lNStdDev = -1, const double lResCut = -1.,
                const double lResCutInit = -1., const int* regroup = 0);

  int GetNGloPar() const { return fNGloPar; }
  int GetNGloParIni() const { return fNGloParIni; }
  const int* GetRegrouping() const { return fkReGroup; }
  int GetNLocPar() const { return fNLocPar; }
  long GetNLocalEquations() const { return fNLocEquations; }
  int GetCurrentIteration() const { return fIter; }
  int GetNMaxIterations() const { return fMaxIter; }
  int GetNStdDev() const { return fNStdDev; }
  int GetNGlobalConstraints() const { return fNGloConstraints; }
  int GetNLagrangeConstraints() const { return fNLagrangeConstraints; }
  int GetNLocalFits() const { return fNLocFits; }
  int GetNLocalFitsRejected() const { return fNLocFitsRejected; }
  int GetNGlobalsFixed() const { return fNGloFix; }
  int GetGlobalSolveStatus() const { return fGloSolveStatus; }
  int GetChi2CutFactor() const { return fChi2CutFactor; }
  float GetChi2CutRef() const { return fChi2CutRef; }
  float GetResCurInit() const { return fResCutInit; }
  float GetResCut() const { return fResCut; }
  int GetMinPntValid() const { return fMinPntValid; }
  int GetRGId(int i) const { return fkReGroup ? (fkReGroup[i] < 0 ? -1 : fkReGroup[i]) : i; }
  int GetProcessedPoints(int i) const
  {
    int ir = GetRGId(i);
    return ir <= 0 ? 0 : fProcPnt[ir];
  }
  int* GetProcessedPoints() const { return fProcPnt; }
  int GetParamGrID(int i) const
  {
    int ir = GetRGId(i);
    return ir <= 0 ? 0 : fParamGrID[ir];
  }

  MatrixSq* GetGlobalMatrix() const { return fMatCGlo; }
  SymMatrix* GetLocalMatrix() const { return fMatCLoc; }
  double* GetGlobals() const { return fVecBGlo; }
  double* GetDeltaPars() const { return fDeltaPar; }
  double* GetInitPars() const { return fInitPar; }
  double* GetSigmaPars() const { return fSigmaPar; }
  bool* GetIsLinear() const { return fIsLinear; }
  double GetFinalParam(int i) const
  {
    int ir = GetRGId(i);
    return ir < 0 ? 0 : fDeltaPar[ir] + fInitPar[ir];
  }
  double GetFinalError(int i) const { return GetParError(i); }
  double GetPull(int i) const;
  //
  double GetGlobal(int i) const
  {
    int ir = GetRGId(i);
    return ir < 0 ? 0 : fVecBGlo[ir];
  }
  double GetInitPar(int i) const
  {
    int ir = GetRGId(i);
    return ir < 0 ? 0 : fInitPar[ir];
  }
  double GetSigmaPar(int i) const
  {
    int ir = GetRGId(i);
    return ir < 0 ? 0 : fSigmaPar[ir];
  }
  bool GetIsLinear(int i) const
  {
    int ir = GetRGId(i);
    return ir < 0 ? 0 : fIsLinear[ir];
  }
  static bool IsGlobalMatSparse() { return fgIsMatGloSparse; }
  static bool IsWeightSigma() { return fgWeightSigma; }
  void SetWghScale(const double wOdd = 1, const double wEven = 1)
  {
    fWghScl[0] = wOdd;
    fWghScl[1] = wEven;
  }
  void SetUseRecordWeight(const bool v = true) { fUseRecordWeight = v; }
  bool GetUseRecordWeight() const { return fUseRecordWeight; }
  void SetMinRecordLength(int v = 1) { fMinRecordLength = v; }
  int GetMinRecordLength() const { return fMinRecordLength; }
  //
  void SetParamGrID(const int grID, int i)
  {
    int ir = GetRGId(i);
    if (ir < 0) {
      return;
    }
    fParamGrID[ir] = grID;
    if (fNGroupsSet < grID) {
      fNGroupsSet = grID;
    }
  }
  void SetNGloPar(const int n) { fNGloPar = n; }
  void SetNLocPar(const int n) { fNLocPar = n; }
  void SetNMaxIterations(const int n = 10) { fMaxIter = n; }
  void SetNStdDev(const int n) { fNStdDev = n; }
  void SetChi2CutFactor(const float v) { fChi2CutFactor = v; }
  void SetChi2CutRef(const float v) { fChi2CutRef = v; }
  void SetResCurInit(const float v) { fResCutInit = v; }
  void SetResCut(const float v) { fResCut = v; }
  void SetMinPntValid(const int n) { fMinPntValid = n > 0 ? n : 1; }
  static void SetGlobalMatSparse(const bool v = true) { fgIsMatGloSparse = v; }
  static void SetWeightSigma(const bool v = true) { fgWeightSigma = v; }

  void SetInitPars(const double* par);
  void SetSigmaPars(const double* par);
  void SetInitPar(int i, double par);
  void SetSigmaPar(int i, double par);

  int GlobalFit(double* par = 0, double* error = 0, double* pull = 0);
  int GlobalFitIteration();
  int SolveGlobalMatEq();
  static void SetInvChol(const bool v = true) { fgInvChol = v; }
  static void SetMinResPrecondType(const int tp = 0) { fgMinResCondType = tp; }
  static void SetMinResTol(double val = 1e-12) { fgMinResTol = val; }
  static void SetMinResMaxIter(const int val = 2000) { fgMinResMaxIter = val; }
  static void SetIterSolverType(const int val = MinResSolve::kSolMinRes) { fgIterSol = val; }
  static void SetNKrylovV(const int val = 60) { fgNKrylovV = val; }

  static bool GetInvChol() { return fgInvChol; }
  static int GetMinResPrecondType() { return fgMinResCondType; }
  static double GetMinResTol() { return fgMinResTol; }
  static int GetMinResMaxIter() { return fgMinResMaxIter; }
  static int GetIterSolverType() { return fgIterSol; }
  static int GetNKrylovV() { return fgNKrylovV; }

  double GetParError(int iPar) const;
  int PrintGlobalParameters() const;
  void SetRejRunList(const int* runs, const int nruns);
  void SetAccRunList(const int* runs, const int nruns, const float* wghList = 0);
  bool IsRecordAcceptable();

  int SetIterations(const double lChi2CutFac);

  // constraints
  void SetGlobalConstraint(const double* dergb, const double val, const double sigma = 0);
  void SetGlobalConstraint(const int* indgb, const double* dergb, const int ngb, const double val, const double sigma = 0);

  // processing of the local measurement
  void SetRecordRun(int run);
  void SetRecordWeight(double wgh);
  void SetLocalEquation(double* dergb, double* derlc, const double lMeas, const double lSigma);
  void SetLocalEquation(int* indgb, double* dergb, int ngb, int* indlc,
                        double* derlc, const int nlc, const double lMeas, const double lSigma);

  // manipilation with processed data and costraints records and its buffer
  void SetDataRecFName(const std::string flname) { fDataRecFName = flname; }
  const char* GetDataRecFName() const { return fDataRecFName.Data(); }
  void SetConsRecFName(const std::string flname) { fConstrRecFName = flname; }
  const char* GetConsRecFName() const { return fConstrRecFName.Data(); }
  //
  void SetRecDataTreeName(const char* name = 0)
  {
    fRecDataTreeName = name;
    if (fRecDataTreeName.IsNull()) {
      fRecDataTreeName = "MillePedeRecords_Data";
    }
  }
  void SetRecConsTreeName(const char* name = 0)
  {
    fRecConsTreeName = name;
    if (fRecConsTreeName.IsNull()) {
      fRecConsTreeName = "MillePedeRecords_Consaints";
    }
  }
  void SetRecDataBranchName(const char* name = 0)
  {
    fRecDataBranchName = name;
    if (fRecDataBranchName.IsNull()) {
      fRecDataBranchName = "Record_Data";
    }
  }
  void SetRecConsBranchName(const char* name = 0)
  {
    fRecConsBranchName = name;
    if (fRecConsBranchName.IsNull()) {
      fRecConsBranchName = "Record_Consaints";
    }
  }
  const char* GetRecDataTreeName() const { return fRecDataTreeName.Data(); }
  const char* GetRecConsTreeName() const { return fRecConsTreeName.Data(); }
  const char* GetRecDataBranchName() const { return fRecDataBranchName.Data(); }
  const char* GetRecConsBranchName() const { return fRecConsBranchName.Data(); }

  bool InitDataRecStorage(bool read = false);
  bool InitConsRecStorage(bool read = false);
  bool ImposeDataRecFile(const char* fname);
  bool ImposeConsRecFile(const char* fname);
  void CloseDataRecStorage();
  void CloseConsRecStorage();
  void ReadRecordData(long recID);
  void ReadRecordConstraint(long recID);
  bool ReadNextRecordData();
  bool ReadNextRecordConstraint();
  void SaveRecordData();
  void SaveRecordConstraint();
  MillePedeRecord* GetRecord() const { return fRecord; }
  long GetSelFirst() const { return fSelFirst; }
  long GetSelLast() const { return fSelLast; }
  void SetSelFirst(const long v) { fSelFirst = v; }
  void SetSelLast(const long v) { fSelLast = v; }

  float Chi2DoFLim(int nSig, int nDoF) const;

  // aliases for compatibility with millipede1
  void SetParSigma(int i, int par) { SetSigmaPar(i, par); }
  void SetGlobalParameters(double* par) { SetInitPars(par); }
  void SetNonLinear(int index, bool v = true)
  {
    int id = GetRGId(index);
    if (id < 0) {
      return;
    }
    fIsLinear[id] = !v;
  }

 protected:
  int LocalFit(double* localParams = 0);
  bool IsZero(double v, double eps = 1e-16) const { return TMath::Abs(v) < eps; }

 protected:
  int fNLocPar;    // number of local parameters
  int fNGloPar;    // number of global parameters
  int fNGloParIni; // number of global parameters before grouping
  int fNGloSize;   // final size of the global matrix (NGloPar+NConstraints)

  long fNLocEquations;       // Number of local equations
  int fIter;                 // Current iteration
  int fMaxIter;              // Maximum number of iterations
  int fNStdDev;              // Number of standard deviations for chi2 cut
  int fNGloConstraints;      // Number of constraint equations
  int fNLagrangeConstraints; // Number of constraint equations requiring Lagrange multiplier
  long fNLocFits;            // Number of local fits
  long fNLocFitsRejected;    // Number of local fits rejected
  int fNGloFix;              // Number of globals fixed by user
  int fGloSolveStatus;       // Status of global solver at current step

  float fChi2CutFactor; // Cut factor for chi2 cut to accept local fit
  float fChi2CutRef;    // Reference cut for chi2 cut to accept local fit
  float fResCutInit;    // Cut in residual for first iterartion
  float fResCut;        // Cut in residual for other iterartiona
  int fMinPntValid;     // min number of points for global to vary

  int fNGroupsSet;   // number of groups set
  int* fParamGrID;   //[fNGloPar] group id for the every parameter
  int* fProcPnt;     //[fNGloPar] N of processed points per global variable
  double* fVecBLoc;  //[fNLocPar] Vector B local (parameters)
  double* fDiagCGlo; //[fNGloPar] Initial diagonal elements of C global matrix
  double* fVecBGlo;  //! Vector B global (parameters)

  double* fInitPar;  //[fNGloPar] Initial global parameters
  double* fDeltaPar; //[fNGloPar] Variation of global parameters
  double* fSigmaPar; //[fNGloPar] Sigma of allowed variation of global parameter

  bool* fIsLinear;   //[fNGloPar] Flag for linear parameters
  bool* fConstrUsed; //! Flag for used constraints

  int* fGlo2CGlo; //[fNGloPar] global ID to compressed ID buffer
  int* fCGlo2Glo; //[fNGloPar] compressed ID to global ID buffer

  // Matrices
  SymMatrix* fMatCLoc;     // Matrix C local
  MatrixSq* fMatCGlo;      // Matrix C global
  RectMatrix* fMatCGloLoc; // Rectangular matrix C g*l
  int* fFillIndex;         //[fNGloPar] auxilary index array for fast matrix fill
  double* fFillValue;      //[fNGloPar] auxilary value array for fast matrix fill

  // processed data record bufferization
  TString fRecDataTreeName;   // Name of data records tree
  TString fRecConsTreeName;   // Name of constraints records tree
  TString fRecDataBranchName; // Name of data records branch name
  TString fRecConsBranchName; // Name of constraints records branch name

  TString fDataRecFName;    // Name of File for data records
  MillePedeRecord* fRecord; // Buffer of measurements records
  TFile* fDataRecFile;      // File of processed measurements records
  TTree* fTreeData;         // Tree of processed measurements records
  int fRecFileStatus;       // state of the record file (0-no, 1-read, 2-rw)

  TString fConstrRecFName; // Name of File for constraints records
  TTree* fTreeConstr;      //! Tree of constraint records
  TFile* fConsRecFile;     //! File of processed constraints records
  long fCurrRecDataID;     // ID of the current data record
  long fCurrRecConstrID;   // ID of the current constraint record
  bool fLocFitAdd;         // Add contribution of carrent track (and not eliminate it)
  bool fUseRecordWeight;   // force or ignore the record weight
  int fMinRecordLength;    // ignore shorter records
  int fSelFirst;           // event selection start
  int fSelLast;            // event selection end
  TArrayL* fRejRunList;    // list of runs to reject (if any)
  TArrayL* fAccRunList;    // list of runs to select (if any)
  TArrayF* fAccRunListWgh; // optional weights for data of accepted runs (if any)
  double fRunWgh;          // run weight
  double fWghScl[2];       // optional rescaling for odd/even residual weights (see its usage in LocalFit)
  const int* fkReGroup;    // optional regrouping of parameters wrt ID's from the records

  static bool fgInvChol;        // Invert global matrix in Cholesky solver
  static bool fgWeightSigma;    // weight parameter constraint by statistics
  static bool fgIsMatGloSparse; // Type of the global matrix (sparse ...)
  static int fgMinResCondType;  // Type of the preconditioner for MinRes method
  static double fgMinResTol;    // Tolerance for MinRes solution
  static int fgMinResMaxIter;   // Max number of iterations for the MinRes method
  static int fgIterSol;         // type of iterative solution: MinRes or FGMRES
  static int fgNKrylovV;        // size of Krylov vectors buffer in FGMRES
  //
  ClassDef(MillePede2, 1)
};

//_____________________________________________________________________________________________
inline void MillePede2::ReadRecordData(long recID)
{
  fTreeData->GetEntry(recID);
  fCurrRecDataID = recID;
}

//_____________________________________________________________________________________________
inline void MillePede2::ReadRecordConstraint(long recID)
{
  fTreeConstr->GetEntry(recID);
  fCurrRecConstrID = recID;
}

//_____________________________________________________________________________________________
inline void MillePede2::SaveRecordData()
{
  fTreeData->Fill();
  fRecord->Reset();
  fCurrRecDataID++;
}

//_____________________________________________________________________________________________
inline void MillePede2::SaveRecordConstraint()
{
  fTreeConstr->Fill();
  fRecord->Reset();
  fCurrRecConstrID++;
}

} // namespace mch
} // namespace o2
#endif
