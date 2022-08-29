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

/// \file MillePede2.h
/// \authors ruben.shahoyan@cern.ch, arakotoz@cern.ch
/// \brief General class for alignment with large number of degrees of freedom, adapted from AliROOT
///
/// Based on the original milliped2 by Volker Blobel
/// http://www.desy.de/~blobel/mptalks.html

#ifndef ALICEO2_MFT_MILLEPEDE2_H
#define ALICEO2_MFT_MILLEPEDE2_H

#include <vector>
#include <TString.h>
#include <TTree.h>
#include "MFTAlignment/MinResSolve.h"
#include "MFTAlignment/MillePedeRecord.h"
#include "MFTAlignment/SymMatrix.h"
#include "MFTAlignment/RectMatrix.h"
#include "MFTAlignment/MatrixSparse.h"
#include "MFTAlignment/MatrixSq.h"
#include "MFTAlignment/MilleRecordWriter.h"
#include "MFTAlignment/MilleRecordReader.h"

class TFile;
class TStopwatch;
class TArrayL;
class TArrayF;

namespace o2
{
namespace mft
{

class MillePede2
{
 public:
  //
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

  /// \brief init all
  int InitMille(int nGlo, const int nLoc,
                const int lNStdDev = -1, const double lResCut = -1.,
                const double lResCutInit = -1., const std::vector<int>& regroup = {});

  int GetNGloPar() const { return fNGloPar; }
  int GetNGloParIni() const { return fNGloParIni; }
  std::vector<int> GetRegrouping() const { return fkReGroup; }
  int GetNLocPar() const { return fNLocPar; }
  long GetNLocalEquations() const { return fNLocEquations; }
  int GetCurrentIteration() const { return fIter; }
  int GetNMaxIterations() const { return fMaxIter; }
  int GetNStdDev() const { return fNStdDev; }
  int GetNGlobalConstraints() const { return fNGloConstraints; }
  int GetNLagrangeConstraints() const { return fNLagrangeConstraints; }
  int GetNLocalFits() const { return fNLocFits; }
  long GetNLocalFitsRejected() const { return fNLocFitsRejected; }
  int GetNGlobalsFixed() const { return fNGloFix; }
  int GetGlobalSolveStatus() const { return fGloSolveStatus; }
  float GetChi2CutFactor() const { return fChi2CutFactor; }
  float GetChi2CutRef() const { return fChi2CutRef; }
  float GetResCurInit() const { return fResCutInit; }
  float GetResCut() const { return fResCut; }
  int GetMinPntValid() const { return fMinPntValid; }
  int GetRGId(int i) const { return fkReGroup.size() ? (fkReGroup[i] < 0 ? -1 : fkReGroup[i]) : i; }
  int GetProcessedPoints(int i) const
  {
    int ir = GetRGId(i);
    return ir <= 0 ? 0 : fProcPnt[ir];
  }
  std::vector<int> GetProcessedPoints() const { return fProcPnt; }
  int GetParamGrID(int i) const
  {
    int ir = GetRGId(i);
    return ir <= 0 ? 0 : fParamGrID[ir];
  }

  MatrixSq* GetGlobalMatrix() const { return fMatCGlo; }
  SymMatrix* GetLocalMatrix() const { return fMatCLoc; }
  std::vector<double> GetGlobals() const { return fVecBGlo; }
  std::vector<double> GetDeltaPars() const { return fDeltaPar; }
  std::vector<double> GetInitPars() const { return fInitPar; }
  std::vector<double> GetSigmaPars() const { return fSigmaPar; }
  std::vector<bool> GetIsLinear() const { return fIsLinear; }
  double GetFinalParam(int i) const
  {
    int ir = GetRGId(i);
    return ir < 0 ? 0 : fDeltaPar[ir] + fInitPar[ir];
  }
  double GetFinalError(int i) const { return GetParError(i); }

  /// \brief return pull for parameter iPar
  double GetPull(int i) const;

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
  void SetMinRecordLength(const int v = 1) { fMinRecordLength = v; }
  int GetMinRecordLength() const { return fMinRecordLength; }

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

  /// \brief initialize parameters, account for eventual grouping
  void SetInitPars(const double* par);

  /// \brief initialize sigmas, account for eventual grouping
  void SetSigmaPars(const double* par);

  /// \brief initialize param, account for eventual grouping
  void SetInitPar(int i, double par);

  /// \brief initialize sigma, account for eventual grouping
  void SetSigmaPar(int i, double par);

  /// \brief performs a requested number of global iterations
  int GlobalFit(double* par = nullptr, double* error = nullptr, double* pull = nullptr);

  /// \brief perform global parameters fit once all the local equations have been fitted
  int GlobalFitIteration();

  /// \brief solve global matrix equation MatCGlob*X=VecBGlo and store the result in the VecBGlo
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

  /// \brief return error for parameter iPar
  double GetParError(int iPar) const;

  /// \brief print the final results into the logfile
  int PrintGlobalParameters() const;

  /// \brief set the list of runs to be rejected
  void SetRejRunList(const int* runs, const int nruns);

  /// \brief set the list of runs to be selected
  void SetAccRunList(const int* runs, const int nruns, const float* wghList = nullptr);

  /// \brief validate record according run lists set by the user
  bool IsRecordAcceptable();

  /// \brief Number of iterations is calculated from lChi2CutFac
  int SetIterations(const double lChi2CutFac);

  // constraints

  /// \brief define a constraint equation
  void SetGlobalConstraint(const std::vector<double>& dergb,
                           const double val, const double sigma = 0,
                           const bool doPrint = false);

  /// \brief define a constraint equation
  void SetGlobalConstraint(const std::vector<int>& indgb,
                           const std::vector<double>& dergb,
                           const int ngb, const double val,
                           double sigma = 0, const bool doPrint = false);

  /// \brief assing derivs of loc.eq.
  void SetLocalEquation(std::vector<double>& dergb, std::vector<double>& derlc,
                        const double lMeas, const double lSigma);

  /// \brief write data of single measurement.
  ///        Note: the records ignore regrouping, store direct parameters
  void SetLocalEquation(std::vector<int>& indgb, std::vector<double>& dergb,
                        int ngb, std::vector<int>& indlc,
                        std::vector<double>& derlc, const int nlc,
                        const double lMeas, const double lSigma);

  /// \brief return file name where is stored chi2 from LocalFit()
  const char* GetRecChi2FName() const { return fRecChi2FName.Data(); }

  /// \brief initialize the file and tree to store chi2 from LocalFit()
  bool InitChi2Storage(const int nEntriesAutoSave = 10000);

  /// \brief write tree and close file where are stored chi2 from LocalFit()
  void EndChi2Storage();

  o2::mft::MillePedeRecord* GetRecord() const { return fRecord; }
  long GetSelFirst() const { return fSelFirst; }
  long GetSelLast() const { return fSelLast; }
  void SetSelFirst(long v) { fSelFirst = v; }
  void SetSelLast(long v) { fSelLast = v; }

  void SetRecord(o2::mft::MillePedeRecord* aRecord) { fRecord = aRecord; }
  void SetRecordWriter(o2::mft::MilleRecordWriter* myP) { fRecordWriter = myP; }
  void SetConstraintsRecWriter(o2::mft::MilleRecordWriter* myP) { fConstraintsRecWriter = myP; }
  void SetRecordReader(o2::mft::MilleRecordReader* myP) { fRecordReader = myP; }
  void SetConstraintsRecReader(o2::mft::MilleRecordReader* myP) { fConstraintsRecReader = myP; }

  /// \brief return the limit in chi^2/nd for n sigmas stdev authorized
  ///
  /// Only n=1, 2, and 3 are expected in input
  float Chi2DoFLim(int nSig, int nDoF) const;

  // aliases for compatibility with millipede1
  void SetParSigma(int i, double par) { SetSigmaPar(i, par); }
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
  /// \brief read data record (if any) at entry recID
  void ReadRecordData(const long recID, const bool doPrint = false);

  /// \brief read constraint record (if any) at entry id recID
  void ReadRecordConstraint(const long recID, const bool doPrint = false);

  /// \brief Perform local parameters fit once all the local equations have been set
  ///
  /// localParams = (if !=0) will contain the fitted track parameters and related errors
  int LocalFit(std::vector<double>& localParams);

  bool IsZero(const double v, const double eps = 1e-16) const { return TMath::Abs(v) < eps; }

 protected:
  int fNLocPar;    ///< number of local parameters
  int fNGloPar;    ///< number of global parameters
  int fNGloParIni; ///< number of global parameters before grouping
  int fNGloSize;   ///< final size of the global matrix (NGloPar+NConstraints)

  long fNLocEquations;       ///< Number of local equations
  int fIter;                 ///< Current iteration
  int fMaxIter;              ///< Maximum number of iterations
  int fNStdDev;              ///< Number of standard deviations for chi2 cut
  int fNGloConstraints;      ///< Number of constraint equations
  int fNLagrangeConstraints; ///< Number of constraint equations requiring Lagrange multiplier
  long fNLocFits;            ///< Number of local fits
  long fNLocFitsRejected;    ///< Number of local fits rejected
  int fNGloFix;              ///< Number of globals fixed by user
  int fGloSolveStatus;       ///< Status of global solver at current step

  float fChi2CutFactor; ///< Cut factor for chi2 cut to accept local fit
  float fChi2CutRef;    ///< Reference cut for chi2 cut to accept local fit
  float fResCutInit;    ///< Cut in residual for first iterartion
  float fResCut;        ///< Cut in residual for other iterartiona
  int fMinPntValid;     ///< min number of points for global to vary

  int fNGroupsSet;               ///< number of groups set
  std::vector<int> fParamGrID;   ///< [fNGloPar] group id for the every parameter
  std::vector<int> fProcPnt;     ///< [fNGloPar] N of processed points per global variable
  std::vector<double> fVecBLoc;  ///< [fNLocPar] Vector B local (parameters)
  std::vector<double> fDiagCGlo; ///< [fNGloPar] Initial diagonal elements of C global matrix
  std::vector<double> fVecBGlo;  //! Vector B global (parameters)

  std::vector<double> fInitPar;  ///< [fNGloPar] Initial global parameters
  std::vector<double> fDeltaPar; ///< [fNGloPar] Variation of global parameters
  std::vector<double> fSigmaPar; ///< [fNGloPar] Sigma of allowed variation of global parameter

  std::vector<bool> fIsLinear;   ///< [fNGloPar] Flag for linear parameters
  std::vector<bool> fConstrUsed; //! Flag for used constraints

  std::vector<int> fGlo2CGlo; ///< [fNGloPar] global ID to compressed ID buffer
  std::vector<int> fCGlo2Glo; ///< [fNGloPar] compressed ID to global ID buffer

  // Matrices
  o2::mft::SymMatrix* fMatCLoc;     ///< Matrix C local
  o2::mft::MatrixSq* fMatCGlo;      ///< Matrix C global
  o2::mft::RectMatrix* fMatCGloLoc; ///< Rectangular matrix C g*l
  std::vector<int> fFillIndex;      ///< [fNGloPar] auxilary index array for fast matrix fill
  std::vector<double> fFillValue;   ///< [fNGloPar] auxilary value array for fast matrix fill

  TFile* fRecChi2File;
  TString fRecChi2FName;
  TString fRecChi2TreeName; ///< Name of chi2 per record tree
  TTree* fTreeChi2;
  float fSumChi2;
  bool fIsChi2BelowLimit;
  int fRecNDoF;

  o2::mft::MillePedeRecord* fRecord; ///< Buffer of measurements records

  long fCurrRecDataID;        ///< ID of the current data record
  long fCurrRecConstrID;      ///< ID of the current constraint record
  bool fLocFitAdd;            ///< Add contribution of carrent track (and not eliminate it)
  bool fUseRecordWeight;      ///< force or ignore the record weight
  int fMinRecordLength;       ///< ignore shorter records
  int fSelFirst;              ///< event selection start
  int fSelLast;               ///< event selection end
  TArrayL* fRejRunList;       ///< list of runs to reject (if any)
  TArrayL* fAccRunList;       ///< list of runs to select (if any)
  TArrayF* fAccRunListWgh;    ///< optional weights for data of accepted runs (if any)
  double fRunWgh;             ///< run weight
  double fWghScl[2];          ///< optional rescaling for odd/even residual weights (see its usage in LocalFit)
  std::vector<int> fkReGroup; ///< optional regrouping of parameters wrt ID's from the records

  static bool fgInvChol;        ///< Invert global matrix in Cholesky solver
  static bool fgWeightSigma;    ///< weight parameter constraint by statistics
  static bool fgIsMatGloSparse; ///< Type of the global matrix (sparse ...)
  static int fgMinResCondType;  ///< Type of the preconditioner for MinRes method
  static double fgMinResTol;    ///< Tolerance for MinRes solution
  static int fgMinResMaxIter;   ///< Max number of iterations for the MinRes method
  static int fgIterSol;         ///< type of iterative solution: MinRes or FGMRES
  static int fgNKrylovV;        ///< size of Krylov vectors buffer in FGMRES

  // processed data record bufferization
  o2::mft::MilleRecordWriter* fRecordWriter;         ///< data record writer
  o2::mft::MilleRecordWriter* fConstraintsRecWriter; ///< constraints record writer
  o2::mft::MilleRecordReader* fRecordReader;         ///< data record reader
  o2::mft::MilleRecordReader* fConstraintsRecReader; ///< constraints record reader

  ClassDef(MillePede2, 0);
};

} // namespace mft
} // namespace o2

#endif
