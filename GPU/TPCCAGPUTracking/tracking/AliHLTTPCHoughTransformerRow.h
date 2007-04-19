// @(#) $Id$
// origin hough/AliL3HoughTransformerRow.h,v 1.15 Sun Apr 30 16:37:32 2006 UTC by hristov 

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/** @file   AliHLTTPCHoughTransformerRow.h
    @author Cvetan Cheshkov
    @date   
    @brief  Implementation of fast HLT TPC hough transform tracking. */

#ifndef ALIHLTTPCHOUGHTRANSFORMERROW_H
#define ALIHLTTPCHOUGHTRANSFORMERROW_H

#include "AliHLTTPCRootTypes.h"
#include "AliHLTTPCHoughTransformer.h"

#define MAX_N_GAPS 5
#define MIN_TRACK_LENGTH 70

class AliHLTTPCDigitData;
class AliHLTTPCHistogram;

class AliHLTTPCHoughTransformerRow : public AliHLTTPCHoughTransformer {

 public:
  /** standard constructor */
  AliHLTTPCHoughTransformerRow(); 
  /** constructor */
  AliHLTTPCHoughTransformerRow(Int_t slice,Int_t patch,Int_t netasegments,Bool_t DoMC=kFALSE,Float_t zvertex=0.0);
  /** standard destructor */
  virtual ~AliHLTTPCHoughTransformerRow();

  struct AliHLTEtaRow {
    UChar_t fStartPad; //First pad in the cluster
    UChar_t fEndPad; //Last pad in the cluster
    Bool_t fIsFound; //Is the cluster already found
#ifdef do_mc
    Int_t fMcLabels[MaxTrack]; //Array to store mc labels inside cluster
#endif
  };

  struct AliHLTPadHoughParams {
    // Parameters which represent given pad in the hough space
    // Used in order to avoid as much as possible floating
    // point operations during the hough transform
    Float_t fAlpha; // Starting value for the hough parameter alpha1
    Float_t fDeltaAlpha; // Slope of alpha1
    Int_t fFirstBin; // First alpha2 bin to be filled 
    Int_t fLastBin; // Last alpha2 bin to be filled
  };

  struct AliHLTTrackLength {
    // Structure is used for temporarely storage of the LUT
    // which contains the track lengths associated to each hough
    // space bin
    Bool_t fIsFilled; // Is bin already filled?
    UInt_t fFirstRow; // First TPC row crossed by the track
    UInt_t fLastRow; // Last TPC row crossed by the track
    Float_t fTrackPt; // Pt of the track
  };



  void CreateHistograms(Float_t ptmin,Float_t ptmax,Float_t pres,Int_t nybin,Float_t psi) {
    AliHLTTPCHoughTransformer::CreateHistograms(ptmin,ptmax,pres,nybin,psi);
  }
  void CreateHistograms(Int_t /*nxbin*/,Float_t /*ptmin*/,Int_t /*nybin*/,Float_t /*phimin*/,Float_t /*phimax*/)
  {STDCERR<<"This method for creation of parameter space histograms is not supported for this Transformer!"<<STDENDL;}
  void CreateHistograms(Int_t nxbin,Float_t xmin,Float_t xmax,
			Int_t nybin,Float_t ymin,Float_t ymax);
  void Reset();
  void TransformCircle();
  void TransformCircle(Int_t *rowRange,Int_t every) {
    AliHLTTPCHoughTransformer::TransformCircle(rowRange,every);
  }

  Int_t GetEtaIndex(Double_t eta) const;
  AliHLTTPCHistogram *GetHistogram(Int_t etaindex);
  Double_t GetEta(Int_t etaindex,Int_t slice) const;
  Int_t GetTrackID(Int_t etaindex,Double_t alpha1,Double_t alpha2) const;
  Int_t GetTrackLength(Double_t alpha1,Double_t alpha2,Int_t *rows) const;
  UChar_t *GetGapCount(Int_t etaindex) const { return fGapCount[etaindex]; }
  UChar_t *GetCurrentRowCount(Int_t etaindex) const { return fCurrentRowCount[etaindex]; }
  UChar_t *GetPrevBin(Int_t etaindex) const { return fPrevBin[etaindex]; }
  UChar_t *GetNextBin(Int_t etaindex) const { return fNextBin[etaindex]; }
  UChar_t *GetNextRow(Int_t etaindex) const { return fNextRow[etaindex]; }
  UChar_t *GetTrackNRows() const { return fTrackNRows; }
  UChar_t *GetTrackFirstRow() const { return fTrackFirstRow; }
  UChar_t *GetTrackLastRow() const { return fTrackLastRow; }
  static Float_t GetBeta1() {return fgBeta1;}
  static Float_t GetBeta2() {return fgBeta2;}
  static Float_t GetDAlpha() {return fgDAlpha;}
  static Float_t GetDEta() {return fgDEta;}
  static Double_t GetEtaCalcParam1() {return fgEtaCalcParam1;}
  static Double_t GetEtaCalcParam2() {return fgEtaCalcParam2;}
  static Double_t GetEtaCalcParam3() {return fgEtaCalcParam3;}

  void SetTPCRawStream(AliTPCRawStream *rawstream) {fTPCRawStream=rawstream;}

 private:
  /** copy constructor prohibited */
  AliHLTTPCHoughTransformerRow(const AliHLTTPCHoughTransformerRow&);
  /** assignment operator prohibited */
  AliHLTTPCHoughTransformerRow& operator=(const AliHLTTPCHoughTransformerRow&);

  UChar_t **fGapCount; //!
  UChar_t **fCurrentRowCount; //!
#ifdef do_mc
  AliHLTTrackIndex **fTrackID; //!
#endif

  UChar_t *fTrackNRows; //!
  UChar_t *fTrackFirstRow; //!
  UChar_t *fTrackLastRow; //!
  UChar_t *fInitialGapCount; //!

  UChar_t **fPrevBin; //!
  UChar_t **fNextBin; //!
  UChar_t **fNextRow; //!

  AliHLTPadHoughParams **fStartPadParams; //!
  AliHLTPadHoughParams **fEndPadParams; //!
  Float_t **fLUTr; //!

  Float_t *fLUTforwardZ; //!
  Float_t *fLUTbackwardZ; //!

  AliHLTTPCHistogram **fParamSpace; //!

  void TransformCircleFromDigitArray();
  void TransformCircleFromRawStream();

  void DeleteHistograms(); //Method to clean up the histograms containing Hough space

  inline void FillClusterRow(UChar_t i,Int_t binx1,Int_t binx2,UChar_t *ngaps2,UChar_t *currentrow2,UChar_t *lastrow2
#ifdef do_mc
			     ,AliHLTEtaRow etaclust,AliHLTTrackIndex *trackid
#endif
			     );
  inline void FillCluster(UChar_t i,Int_t etaindex,AliHLTEtaRow *etaclust,Int_t ilastpatch,Int_t firstbinx,Int_t lastbinx,Int_t nbinx,Int_t firstbiny);
#ifdef do_mc
  inline void FillClusterMCLabels(AliHLTDigitData digpt,AliHLTEtaRow *etaclust);
#endif

  void SetTransformerArrays(AliHLTTPCHoughTransformerRow *tr);

  static Float_t fgBeta1,fgBeta2; // Two curves which define the Hough space
  static Float_t fgDAlpha, fgDEta; // Correlation factor between Hough space bin size and resolution
  static Double_t fgEtaCalcParam1, fgEtaCalcParam2; // Parameters used for fast calculation of eta during the binning of Hough space
  static Double_t fgEtaCalcParam3; // Parameter used during the eta binning of the Hough Space in order to account for finite track radii

  AliTPCRawStream *fTPCRawStream; // Pointer to the raw stream in case of fast reading of the raw data (fast_raw flag)

  ClassDef(AliHLTTPCHoughTransformerRow,1) //TPC Rows Hough transformation class

};

#endif




