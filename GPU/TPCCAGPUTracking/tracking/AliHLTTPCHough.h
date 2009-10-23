//-*- Mode: C++ -*-
// $Id$
// origin hough/AliL3Hough.h,v 1.31 Fri Feb 25 07:32:13 2005 UTC by cvetan

#ifndef ALIHLTTPCHOUGH_H
#define ALIHLTTPCHOUGH_H

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

/** @file   AliHLTTPCHough.h
    @author Anders Vestbo, Cvetan Cheshkov
    @date   
    @brief  Steering for HLT TPC hough transform tracking algorithms. */

#include "AliHLTTPCRootTypes.h"

class AliHLTTPCHoughMaxFinder;
class AliHLTTPCHoughTransformer;
class AliHLTTPCHistogram;
class AliHLTTPCMemHandler;
class AliHLTTPCFileHandler;
class AliHLTTPCHoughEval;
class AliHLTTPCTrackArray;
class AliHLTTPCHoughMerger;
class AliHLTTPCHoughIntMerger;
class AliHLTTPCHoughGlobalMerger;
class AliHLTTPCBenchmark;

#ifdef HAVE_THREAD
class TThread;
#endif // HAVE_THREAD
class AliRunLoader;
class AliRawEvent;
class AliESDEvent;
class AliESDHLTtrack;

/** 
 * @class AliHLTTPCHough
 * Interface class for the HLT TPC Hough transform tracking algorithms
 *
 * Example how to use:
 *<pre>
 * AliHLTTPCHough *hough = new AliHLTTPCHough(path,kTRUE,NumberOfEtaSegments);
 * hough->ReadData(slice);
 * hough->Transform();
 * hough->FindTrackCandidates();
 * 
 * AliHLTTPCTrackArray *tracks = hough->GetTracks(patch);
 *
 *</pre>
*/
class AliHLTTPCHough {
 public:
  
  AliHLTTPCHough(); 
  AliHLTTPCHough(Char_t *path,Bool_t binary,Int_t netasegments=100,Bool_t bit8=kFALSE,Int_t tv=0,Char_t *infile=0,Char_t *ptr=0);
  virtual ~AliHLTTPCHough();

  void SetRunLoader(AliRunLoader *runloader) {fRunLoader = runloader;}

  void Init(Int_t netasegments,Int_t tv,AliRawEvent *rawevent,Float_t zvertex=0.0);
  void Init(Char_t *path,Bool_t binary,Int_t netasegments=100,Bool_t bit8=kFALSE,Int_t tv=0,Char_t *infile=0,Char_t *ptr=0,Float_t zvertex=0.0);
  void Init(Bool_t doit=kFALSE, Bool_t addhists=kFALSE);

  void Process(Int_t minslice,Int_t maxslice);
  void ReadData(Int_t slice,Int_t eventnr=0);
  void Transform(Int_t *rowrange = 0);
  void ProcessSliceIter();
  void ProcessPatchIter(Int_t patch);
  void MergePatches();
  void MergeInternally();
  void MergeEtaSlices();

  void FindTrackCandidates();
  void FindTrackCandidatesRow();
  void AddAllHistograms();
  void AddAllHistogramsRows();
  void PrepareForNextPatch(Int_t nextpatch);
  Int_t Evaluate(Int_t roadwidth=1,Int_t nrowstomiss=1);
  void EvaluatePatch(Int_t i,Int_t roadwidth,Int_t nrowstomiss);
  void WriteTracks(Int_t slice,Char_t *path="./");
  void WriteTracks(Char_t *path);
  Int_t FillESD(AliESDEvent *esd);
  void WriteDigits(Char_t *outfile="output_digits.root");
  void InitEvaluate();
  void DoBench(Char_t *filename);
  void AddTracks();
  
  //Setters
  void SetNEtaSegments(Int_t i) {fNEtaSegments = i;}
  void SetAddHistograms() {fAddHistograms = kTRUE;}
  void DoIterative() {fDoIterative = kTRUE;}
  void SetWriteDigits() {fWriteDigits = kTRUE;}
  void SetTransformerParams(Float_t ptres=0,Float_t ptmin=0,Float_t ptmax=0,Int_t ny=0,Int_t patch=-1);
  //{fPtRes=ptres;fNBinY=ny;fLowPt=ptmin;fUpperPt=ptmax;fPhi=psi;}
  void SetTransformerParams(Int_t nx,Int_t ny,Float_t lpt,Int_t patch);
  void CalcTransformerParams(Float_t lpt);
  void SetTransformerParams(Int_t nx,Int_t ny,Float_t lpt,Float_t phi);
  //{fNBinX=nx;fNBinY=ny;fLowPt=lpt;fPhi=phi;}
  void SetThreshold(Int_t t=3,Int_t patch=-1);
  void SetNSaveIterations(Int_t t=10) {fNSaveIterations=t;}
  void SetPeakThreshold(Int_t threshold=0,Int_t patch=-1);
  
  void SetPeakParameters(Int_t kspread,Float_t pratio) {fKappaSpread=kspread; fPeakRatio=pratio;}
  
  //Getters
  AliHLTTPCHoughTransformer *GetTransformer(Int_t i) const {if(!fHoughTransformer[i]) return 0; return fHoughTransformer[i];}
  AliHLTTPCTrackArray *GetTracks(Int_t i) const {if(!fTracks[i]) return 0; return fTracks[i];}
  AliHLTTPCHoughEval *GetEval(Int_t i) const {if(!fEval[i]) return 0; return fEval[i];}
  AliHLTTPCHoughMerger *GetMerger() const{if(!fMerger) return 0; return fMerger;}
  AliHLTTPCHoughIntMerger *GetInterMerger() const {if(!fInterMerger) return 0; return fInterMerger;}
  AliHLTTPCMemHandler *GetMemHandler(Int_t i) const {if(!fMemHandler[i]) return 0; return fMemHandler[i];}
  AliHLTTPCHoughMaxFinder *GetMaxFinder() const {return fPeakFinder;}

  //Special methods for executing Hough Transform as a thread
  static void *ProcessInThread(void *args);
  void StartProcessInThread(Int_t minslice,Int_t maxslice);
  Int_t WaitForThreadFinish();
  void SetMinMaxSlices(Int_t minslice,Int_t maxslice) {fMinSlice = minslice; fMaxSlice = maxslice;} 
  Int_t GetMinSlice() const {return fMinSlice;}
  Int_t GetMaxSlice() const {return fMaxSlice;}
  
 private:
  /// copy constructor not permitted
  AliHLTTPCHough(const AliHLTTPCHough);
  /// assignment operator not permitted
  AliHLTTPCHough& operator=(const AliHLTTPCHough);
  
  Char_t *fInputFile;//!
  Char_t *fInputPtr;//!
  AliRawEvent *fRawEvent;//!
  Char_t fPath[1024]; // Path to the files
  Bool_t fBinary; // Is input binary
  Bool_t fAddHistograms; // Add all patch histograms at the end or not
  Bool_t fDoIterative; // Iterative or not
  Bool_t fWriteDigits; // Write Digits or not
  Bool_t fUse8bits; // Use 8 bits or not
  Int_t fNEtaSegments; // Number of eta slices
  Int_t fNPatches; // Number of patches
  Int_t fLastPatch; //The index of the last processed patch
  Int_t fVersion; //which HoughTransformer to use
  Int_t fCurrentSlice; // Current TPC slice (sector)
  Int_t fEvent; // Current event number

  Int_t fPeakThreshold[6]; // Threshold for the peak finder
  Float_t fLowPt[6]; // Lower limit on Pt
  Float_t fUpperPt[6]; // Upper limit on Pt
  Float_t fPtRes[6]; // Desired Pt resolution
  Float_t fPhi[6]; // Limit on the emission angle
  Int_t fNBinX[6]; // Number of bins in the Hough space
  Int_t fNBinY[6]; // Number of bins in the Hough space
  Int_t fThreshold[6]; // Threshold for digits
  Int_t fNSaveIterations; //for HoughtransformerVhdl
  
  //parameters for the peak finder:
  Int_t fKappaSpread; // Kappa spread
  Float_t fPeakRatio; // Peak ratio

  Float_t fZVertex; // Z position of the primary vertex

  Int_t fMinSlice; // First TPC slice (sector) to process while running in a thread
  Int_t fMaxSlice; // Last TPC slice (sector) to process while running in a thread

  AliHLTTPCMemHandler **fMemHandler; //!
  AliHLTTPCHoughTransformer **fHoughTransformer; //!
  AliHLTTPCHoughEval **fEval; //!
  AliHLTTPCHoughMaxFinder *fPeakFinder; //!
  AliHLTTPCTrackArray **fTracks; //!
  AliHLTTPCTrackArray *fGlobalTracks; //!
  AliHLTTPCHoughMerger *fMerger; //!
  AliHLTTPCHoughIntMerger *fInterMerger; //!
  AliHLTTPCHoughGlobalMerger *fGlobalMerger; //!
  AliHLTTPCBenchmark *fBenchmark; //!

  AliRunLoader *fRunLoader; // Run Loader

  void CleanUp();
  Double_t GetCpuTime();

#ifdef HAVE_THREAD
  TThread *fThread; //! Pointer to the TThread object in case of running in a thread
#endif // HAVE_THREAD

  ClassDef(AliHLTTPCHough,1) //Hough transform base class
};

#endif
