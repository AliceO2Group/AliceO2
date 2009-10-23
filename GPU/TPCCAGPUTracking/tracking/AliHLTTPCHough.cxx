// $Id$
// origin: hough/AliL3Hough.cxx,v 1.50 Tue Mar 28 18:05:12 2006 UTC by alibrary

//**************************************************************************
//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//*                                                                        *
//* Primary Authors: Anders Vestbo, Cvetan Cheshkov                        *
//*                  for The ALICE HLT Project.                            *
//*                                                                        *
//* Permission to use, copy, modify and distribute this software and its   *
//* documentation strictly for non-commercial purposes is hereby granted   *
//* without fee, provided that the above copyright notice appears in all   *
//* copies and that both the copyright notice and this permission notice   *
//* appear in the supporting documentation. The authors make no claims     *
//* about the suitability of this software for any purpose. It is          *
//* provided "as is" without express or implied warranty.                  *
//**************************************************************************

/** @file   AliHLTTPCHough.cxx
    @author Anders Vestbo, Cvetan Cheshkov
    @date   
    @brief  Steering for HLT TPC hough transform tracking algorithms. */


#include "AliHLTStdIncludes.h"
#include "AliLog.h"
#include <sys/time.h>

#include "AliHLTTPCLogging.h"

#ifdef HAVE_ALIHLTHOUGHMERGER
#include "AliHLTHoughMerger.h"
#endif //HAVE_ALIHLTHOUGHMERGER

#ifdef HAVE_ALIHLTHOUGHINTMERGER
#include "AliHLTHoughIntMerger.h"
#endif //HAVE_ALIHLTHOUGHINTMERGER

#ifdef HAVE_ALIHLTHOUGHGLOBALMERGER
#include "AliHLTHoughGlobalMerger.h"
#endif //HAVE_ALIHLTHOUGHGLOBALMERGER

#include "AliHLTTPCHistogram.h"
#include "AliHLTTPCHough.h"

#ifdef HAVE_ALIHLTHOUGHTRANSFORMERDEFAULT
// the original AliHLTHoughBaseTransformer has been renamed to
// AliHLTTPCHoughTransformer and AliHLTHoughTransformer to
// AliHLTHoughTransformerDefault, but the latter is not yet
// migrated
#include "AliHLTHoughTransformer.h"
#endif // HAVE_ALIHLTHOUGHTRANSFORMERDEFAULT

#ifdef HAVE_ALIHLTHOUGHCLUSTERTRANSFORMER
#include "AliHLTHoughClusterTransformer.h"
#endif // HAVE_ALIHLTHOUGHCLUSTERTRANSFORMER

#ifdef HAVE_ALIHLTHOUGHTRANSFORMERLUT
#include "AliHLTHoughTransformerLUT.h"
#endif // HAVE_ALIHLTHOUGHTRANSFORMERLUT

#ifdef HAVE_ALIHLTHOUGHTRANSFORMERVHDL
#include "AliHLTHoughTransformerVhdl.h"
#endif // HAVE_ALIHLTHOUGHTRANSFORMERVHDL

#include "AliHLTTPCHoughTransformerRow.h"

#include "AliHLTTPCHoughMaxFinder.h"
#include "AliHLTTPCBenchmark.h"
#include "AliHLTTPCFileHandler.h"

#ifdef HAVE_ALIHLTDATAHANDLER
#include "AliHLTDataHandler.h"
#endif // HAVE_ALIHLTDATAHANDLER

//#include "AliHLTDigitData.h"
#include "AliHLTTPCHoughEval.h"
#include "AliHLTTPCTransform.h"
#include "AliHLTTPCTrackArray.h"
#include "AliHLTTPCHoughTrack.h"

#ifdef HAVE_ALIHLTDDLDATAFILEHANDLER
#include "AliHLTDDLDataFileHandler.h"
#endif // HAVE_ALIHLTDDLDATAFILEHANDLER

#include "AliHLTTPCHoughKalmanTrack.h"

#ifdef HAVE_THREAD
#include "TThread.h"
#endif // HAVE_THREAD
#include <AliRunLoader.h>
#include <AliRawEvent.h>
#include <AliESDEvent.h>
#include <AliESDtrack.h>
#include <AliESDHLTtrack.h>

#if __GNUC__ >= 3
using namespace std;
#endif

/** ROOT macro for the implementation of ROOT specific class methods */
ClassImp(AliHLTTPCHough);

AliHLTTPCHough::AliHLTTPCHough()
{
  //Constructor
  
  fBinary        = kFALSE;
  fAddHistograms = kFALSE;
  fDoIterative   = kFALSE; 
  fWriteDigits   = kFALSE;
  fUse8bits      = kFALSE;

  fMemHandler       = 0;
  fHoughTransformer = 0;
  fEval             = 0;
  fPeakFinder       = 0;
  fTracks           = 0;
  fGlobalTracks     = 0;
  fMerger           = 0;
  fInterMerger      = 0;
  fGlobalMerger     = 0;
  fBenchmark        = 0;
  
  fNEtaSegments     = 0;
  fNPatches         = 0;
  fLastPatch        =-1;
  fVersion          = 0;
  fCurrentSlice     = 0;
  fEvent            = 0;
  
  fKappaSpread = 6;
  fPeakRatio   = 0.5;
  fInputFile   = 0;
  fInputPtr    = 0;
  fRawEvent    = 0;
  
  SetTransformerParams();
  SetThreshold();
  SetNSaveIterations();
  SetPeakThreshold();
  //just be sure that index is empty for new event
    AliHLTTPCFileHandler::CleanStaticIndex(); 
    fRunLoader = 0;
#ifdef HAVE_THREAD
  fThread = 0;
#endif // HAVE_THREAD
}

AliHLTTPCHough::AliHLTTPCHough(Char_t *path,Bool_t binary,Int_t netasegments,Bool_t bit8,Int_t tv,Char_t *infile,Char_t *ptr)
{
  //Normal constructor
  fBinary = binary;
  strcpy(fPath,path);
  fNEtaSegments  = netasegments;
  fAddHistograms = kFALSE;
  fDoIterative   = kFALSE; 
  fWriteDigits   = kFALSE;
  fUse8bits      = bit8;
  fVersion       = tv;
  fKappaSpread=6;
  fPeakRatio=0.5;
  if(!fBinary) {
    if(infile) {
      fInputFile = infile;
      fInputPtr = 0;
    }
    else {
      fInputFile = 0;
      fInputPtr = ptr;
    }
  }
  else {
    fInputFile = 0;
    fInputPtr = 0;
  }
  //just be sure that index is empty for new event
    AliHLTTPCFileHandler::CleanStaticIndex(); 
    fRunLoader = 0;
#ifdef HAVE_THREAD
  fThread = 0;
#endif // HAVE_THREAD
}

AliHLTTPCHough::~AliHLTTPCHough()
{
  //dtor

  CleanUp();

#ifdef HAVE_ALIHLTHOUGHMERGER
  if(fMerger)
    delete fMerger;
#endif //HAVE_ALIHLTHOUGHMERGER
  //cout << "Cleaned class merger " << endl;
#ifdef HAVE_ALIHLTHOUGHINTMERGER
  if(fInterMerger)
    delete fInterMerger;
#endif //HAVE_ALIHLTHOUGHINTMERGER
  //cout << "Cleaned class inter " << endl;
  if(fPeakFinder)
    delete fPeakFinder;
  //cout << "Cleaned class peak " << endl;
#ifdef HAVE_ALIHLTHOUGHGLOBALMERGER
  if(fGlobalMerger)
    delete fGlobalMerger;
#endif //HAVE_ALIHLTHOUGHGLOBALMERGER
  //cout << "Cleaned class global " << endl;
  if(fBenchmark)
    delete fBenchmark;
  //cout << "Cleaned class bench " << endl;
  if(fGlobalTracks)
    delete fGlobalTracks;
  //cout << "Cleaned class globaltracks " << endl;
#ifdef HAVE_THREAD
  if(fThread) {
    //    fThread->Delete();
    delete fThread;
    fThread = 0;
  }
#endif // HAVE_THREAD
}

void AliHLTTPCHough::CleanUp()
{
  //Cleanup memory
  
  for(Int_t i=0; i<fNPatches; i++)
    {
      if(fTracks[i]) delete fTracks[i];
      //cout << "Cleaned tracks " << i << endl;
      if(fEval[i]) delete fEval[i];
      //cout << "Cleaned eval " << i << endl;
      if(fHoughTransformer[i]) delete fHoughTransformer[i];
      //cout << "Cleaned traf " << i << endl;
      if(fMemHandler[i]) delete fMemHandler[i];
      //cout << "Cleaned mem " << i << endl;
    }
  
  if(fTracks) delete [] fTracks;
  //cout << "Cleaned class tracks " << endl;
  if(fEval) delete [] fEval;
  //cout << "Cleaned class eval " << endl;
  if(fHoughTransformer) delete [] fHoughTransformer;
  //cout << "Cleaned cleass trafo " << endl;
  if(fMemHandler) delete [] fMemHandler;
  //cout << "Cleaned class mem " << endl;
}

void AliHLTTPCHough::Init(Int_t netasegments,Int_t tv,AliRawEvent *rawevent,Float_t zvertex)
{
  //Normal constructor
  fNEtaSegments  = netasegments;
  fVersion       = tv;
  fRawEvent      = rawevent;
  fZVertex       = zvertex;

  Init();
}

void AliHLTTPCHough::Init(Char_t *path,Bool_t binary,Int_t netasegments,Bool_t bit8,Int_t tv,Char_t *infile,Char_t *ptr,Float_t zvertex)
{
  //Normal init of the AliHLTTPCHough
  fBinary = binary;
  strcpy(fPath,path);
  fNEtaSegments = netasegments;
  fWriteDigits  = kFALSE;
  fUse8bits     = bit8;
  fVersion      = tv;
  if(!fBinary) {
    if(infile) {
      fInputFile = infile;
      fInputPtr = 0;
    }
    else {
      fInputFile = 0;
      fInputPtr = ptr;
    }
  }
  else {
    fInputFile = 0;
    fInputPtr = 0;
  }
  fZVertex = zvertex;

  Init(); //do the rest
}

void AliHLTTPCHough::Init(Bool_t doit, Bool_t addhists)
{
  // Init
  fDoIterative   = doit; 
  fAddHistograms = addhists;

  fNPatches = AliHLTTPCTransform::GetNPatches();
  fHoughTransformer = new AliHLTTPCHoughTransformer*[fNPatches];
  fMemHandler = new AliHLTTPCMemHandler*[fNPatches];

  fTracks = new AliHLTTPCTrackArray*[fNPatches];
  fEval = new AliHLTTPCHoughEval*[fNPatches];
  
  fGlobalTracks = new AliHLTTPCTrackArray("AliHLTTPCHoughTrack");
  
  AliHLTTPCHoughTransformer *lasttransformer = 0;

  for(Int_t i=0; i<fNPatches; i++)
    {
      switch (fVersion){ //choose Transformer
      case 1: 
#ifdef HAVE_ALIHLTHOUGHTRANSFORMERLUT
	fHoughTransformer[i] = new AliHLTHoughTransformerLUT(0,i,fNEtaSegments);
#else
	AliErrorClassStream() << "AliHLTHoughTransformerLUT not compiled" << endl;
#endif // HAVE_ALIHLTHOUGHTRANSFORMERLUT
	break;
      case 2:
#ifdef HAVE_ALIHLTHOUGHCLUSTERTRANSFORMER
	fHoughTransformer[i] = new AliHLTHoughClusterTransformer(0,i,fNEtaSegments);
#else
	AliErrorClassStream() << "AliHLTHoughClusterTransformer not compiled" << endl;
#endif // HAVE_ALIHLTHOUGHCLUSTERTRANSFORMER
	break;
      case 3:
#ifdef HAVE_ALIHLTHOUGHTRANSFORMERVHDL
	fHoughTransformer[i] = new AliHLTHoughTransformerVhdl(0,i,fNEtaSegments,fNSaveIterations);
#else
	AliErrorClassStream() << "AliHLTHoughTransformerVhdl not compiled" << endl;
#endif // HAVE_ALIHLTHOUGHTRANSFORMERVHDL
	break;
      case 4:
	fHoughTransformer[i] = new AliHLTTPCHoughTransformerRow(0,i,fNEtaSegments,kFALSE,fZVertex);
	break;
      default:
#ifdef HAVE_ALIHLTHOUGHTRANSFORMERDEFAULT
	fHoughTransformer[i] = new AliHLTHoughTransformerDefault(0,i,fNEtaSegments,kFALSE,kFALSE);
#else
	AliErrorClassStream() << "AliHLTHoughTransformerDefault not compiled" << endl;
#endif // HAVE_ALIHLTHOUGHTRANSFORMERDEFAULT
      }

      fHoughTransformer[i]->SetLastTransformer(lasttransformer);
      lasttransformer = fHoughTransformer[i];
      //      fHoughTransformer[i]->CreateHistograms(fNBinX[i],fLowPt[i],fNBinY[i],-fPhi[i],fPhi[i]);
      fHoughTransformer[i]->CreateHistograms(fNBinX[i],-fLowPt[i],fLowPt[i],fNBinY[i],-fPhi[i],fPhi[i]);
      //fHoughTransformer[i]->CreateHistograms(fLowPt[i],fUpperPt[i],fPtRes[i],fNBinY[i],fPhi[i]);

      fHoughTransformer[i]->SetLowerThreshold(fThreshold[i]);
      fHoughTransformer[i]->SetUpperThreshold(100);

      LOG(AliHLTTPCLog::kInformational,"AliHLTTPCHough::Init","Version")
	<<"Initializing Hough transformer version "<<fVersion<<ENDLOG;
      
      fEval[i] = new AliHLTTPCHoughEval();
      fTracks[i] = new AliHLTTPCTrackArray("AliHLTTPCHoughTrack");
      if(fUse8bits) {
#ifdef HAVE_ALIHLTDATAHANDLER
	fMemHandler[i] = new AliHLTDataHandler();
#else //!HAVE_ALIHLTDATAHANDLER
	AliErrorClassStream() << "AliHLTDataHandler not compiled" << endl;
#endif // HAVE_ALIHLTDATAHANDLER
      } else
      	{
	  if(!fRawEvent) {
	    if(!fInputFile) {
	      if(!fInputPtr) {
		/* In case of reading digits file */
		fMemHandler[i] = new AliHLTTPCFileHandler(kTRUE); //use static index
		if(!fBinary) {
		  if(!fRunLoader) {
		    Char_t filename[1024];
		    sprintf(filename,"%s/digitfile.root",fPath);
		    fMemHandler[i]->SetAliInput(filename);
		  }
		  else {
		    fMemHandler[i]->SetAliInput(fRunLoader);
		  }
		}
	      }
	      else {
		/* In case of reading from DATE */
#ifdef HAVE_ALIHLTDDLDATAFILEHANDLER
		fMemHandler[i] = new AliHLTDDLDataFileHandler();
		fMemHandler[i]->SetReaderInput(fInputPtr,-1);
#else //!HAVE_ALIHLTDDLDATAFILEHANDLER
		AliErrorClassStream() << "AliHLTDDLDataFileHandler not compiled" << endl;
#endif //HAVE_ALIHLTDDLDATAFILEHANDLER
	      }
	    }
	    else {
	      /* In case of reading rawdata from ROOT file */
#ifdef HAVE_ALIHLTDDLDATAFILEHANDLER
	      fMemHandler[i] = new AliHLTDDLDataFileHandler();
	      fMemHandler[i]->SetReaderInput(fInputFile);
#else //!HAVE_ALIHLTDDLDATAFILEHANDLER
	      AliErrorClassStream() << "AliHLTDDLDataFileHandler not compiled" << endl;
#endif //HAVE_ALIHLTDDLDATAFILEHANDLER
	    }
	  }
	  else {
	    /* In case of reading rawdata using AliRawEvent */
#ifdef HAVE_ALIHLTDDLDATAFILEHANDLER
	    fMemHandler[i] = new AliHLTDDLDataFileHandler();
	    fMemHandler[i]->SetReaderInput(fRawEvent);
#else //!HAVE_ALIHLTDDLDATAFILEHANDLER
	    AliErrorClassStream() << "AliHLTDDLDataFileHandler not compiled" << endl;
#endif //HAVE_ALIHLTDDLDATAFILEHANDLER
	  }
	}
    }

  fPeakFinder = new AliHLTTPCHoughMaxFinder("KappaPhi",50000);
  if(fVersion!=4) {
#ifdef HAVE_ALIHLTHOUGHMERGER
    fMerger = new AliHLTHoughMerger(fNPatches);
#else 
    AliErrorClassStream() << "AliHLTHoughMerger not compiled" << endl;
#endif //HAVE_ALIHLTHOUGHMERGER
#ifdef HAVE_ALIHLTHOUGHINTMERGER
    fInterMerger = new AliHLTHoughIntMerger();
#else 
    AliErrorClassStream() << "AliHLTHoughIntMerger not compiled" << endl;
#endif //HAVE_ALIHLTHOUGHINTMERGER
  }
  else {
    fMerger = 0;
    fInterMerger = 0;
  }
  fGlobalMerger = 0;
  fBenchmark = new AliHLTTPCBenchmark();
}

void AliHLTTPCHough::SetTransformerParams(Float_t ptres,Float_t ptmin,Float_t ptmax,Int_t ny,Int_t patch)
{
  // Setup the parameters for the Hough Transformer
  // This includes the bin size and limits for
  // the parameter space histograms

  Int_t mrow;
  Float_t psi=0;
  if(patch==-1)
    mrow = 80;
  else
    mrow = AliHLTTPCTransform::GetLastRow(patch);
  if(ptmin)
    {
      Double_t lineradius = sqrt(pow(AliHLTTPCTransform::Row2X(mrow),2) + pow(AliHLTTPCTransform::GetMaxY(mrow),2));
      Double_t kappa = -1*AliHLTTPCTransform::GetBField()*AliHLTTPCTransform::GetBFact()/ptmin;
      psi = AliHLTTPCTransform::Deg2Rad(10) - asin(lineradius*kappa/2);
      cout<<"Calculated psi range "<<psi<<" in patch "<<patch<<endl;
    }

  if(patch==-1)
    {
      Int_t i=0;
      while(i < 6)
	{
	  fPtRes[i] = ptres;
	  fLowPt[i] = ptmin;
	  fUpperPt[i] = ptmax;
	  fNBinY[i] = ny;
	  fPhi[i] = psi;
	  fNBinX[i]=0;
	  i++;
	}
      return;
    }

  fPtRes[patch] = ptres;
  fLowPt[patch] = ptmin;
  fUpperPt[patch] = ptmax;
  fNBinY[patch] = ny;
  fPhi[patch] = psi;
}
/*
void AliHLTTPCHough::SetTransformerParams(Int_t nx,Int_t ny,Float_t ptmin,Int_t patch)
{
  // Setup the parameters for the Hough Transformer

  Int_t mrow=80;
  Double_t lineradius = sqrt(pow(AliHLTTPCTransform::Row2X(mrow),2) + pow(AliHLTTPCTransform::GetMaxY(mrow),2));
  Double_t kappa = -1*AliHLTTPCTransform::GetBField()*AliHLTTPCTransform::GetBFact()/ptmin;
  Double_t psi = AliHLTTPCTransform::Deg2Rad(10) - asin(lineradius*kappa/2);
  cout<<"Calculated psi range "<<psi<<" in patch "<<patch<<endl;
  
  Int_t i=0;
  while(i < 6)
    {
      fLowPt[i] = ptmin;
      fNBinY[i] = ny;
      fNBinX[i] = nx;
      fPhi[i] = psi;
      i++;
    }
}
*/
void AliHLTTPCHough::SetTransformerParams(Int_t nx,Int_t ny,Float_t ptmin,Int_t /*patch*/)
{
  // Setup the parameters for the Hough Transformer

  Double_t lineradius = 1.0/(AliHLTTPCHoughTransformerRow::GetBeta1()*sqrt(1.0+tan(AliHLTTPCTransform::Pi()*10/180)*tan(AliHLTTPCTransform::Pi()*10/180)));
  Double_t alpha1 = AliHLTTPCHoughTransformerRow::GetBeta1()*tan(AliHLTTPCTransform::Pi()*10/180);
  Double_t kappa = 1*AliHLTTPCTransform::GetBField()*AliHLTTPCTransform::GetBFact()/(ptmin*0.9);
  Double_t psi = AliHLTTPCTransform::Deg2Rad(10) - asin(lineradius*kappa/2);
  //  cout<<"Calculated psi range "<<psi<<" in patch "<<patch<<endl;
  Double_t alpha2 = alpha1 - (AliHLTTPCHoughTransformerRow::GetBeta1()-AliHLTTPCHoughTransformerRow::GetBeta2())*tan(psi);
  //  cout<<"Calculated alphas range "<<alpha1<<" "<<alpha2<<" in patch "<<patch<<endl;

  Int_t i=0;
  while(i < 6)
    {
      fLowPt[i] = 1.1*alpha1;
      fNBinY[i] = ny;
      fNBinX[i] = nx;
      fPhi[i] = alpha2;
      i++;
    }
}

void AliHLTTPCHough::CalcTransformerParams(Float_t ptmin)
{
  // Setup the parameters for the Row Hough Transformer
  // Automatically adjusts the number of bins in X and Y in a way
  // that the size of the hough bin is 2x (in X) and 2.5 (in Y) the
  // size of the tpc pads

  Double_t lineradius = 1.0/(AliHLTTPCHoughTransformerRow::GetBeta1()*sqrt(1.0+tan(AliHLTTPCTransform::Pi()*10/180)*tan(AliHLTTPCTransform::Pi()*10/180)));
  Double_t alpha1 = AliHLTTPCHoughTransformerRow::GetBeta1()*tan(AliHLTTPCTransform::Pi()*10/180);
  Double_t kappa = 1*AliHLTTPCTransform::GetBField()*AliHLTTPCTransform::GetBFact()/(ptmin*0.9);
  Double_t psi = AliHLTTPCTransform::Deg2Rad(10) - asin(lineradius*kappa/2);
  //  cout<<"Calculated psi range "<<psi<<endl;
  Double_t alpha2 = alpha1 - (AliHLTTPCHoughTransformerRow::GetBeta1()-AliHLTTPCHoughTransformerRow::GetBeta2())*tan(psi);
  alpha1 *= 1.1;
  //  cout<<"Calculated alphas range "<<alpha1<<" "<<alpha2<<endl;

  Double_t sizex = 2.0*AliHLTTPCTransform::GetPadPitchWidthLow()*AliHLTTPCHoughTransformerRow::GetBeta1()*AliHLTTPCHoughTransformerRow::GetBeta1();
  Double_t sizey = 2.5*AliHLTTPCTransform::GetPadPitchWidthUp()*AliHLTTPCHoughTransformerRow::GetBeta2()*AliHLTTPCHoughTransformerRow::GetBeta2();

  Int_t nx = 2*(Int_t)(alpha1/sizex)+1;
  Int_t ny = 2*(Int_t)(alpha2/sizey)+1;
  //  cout<<"Calculated number of bins "<<nx<<" "<<ny<<endl;

  Int_t i=0;
  while(i < 6)
    {
      fLowPt[i] = alpha1;
      fNBinY[i] = ny;
      fNBinX[i] = nx;
      fPhi[i] = alpha2;
      i++;
    }
}

void AliHLTTPCHough::SetTransformerParams(Int_t nx,Int_t ny,Float_t lpt,Float_t phi)
{
  // SetTransformerParams

  Int_t i=0;
  while(i < 6)
    {
      fLowPt[i] = lpt;
      fNBinY[i] = ny;
      fNBinX[i] = nx;
      fPhi[i] = phi;
      i++;
    }
}

void AliHLTTPCHough::SetThreshold(Int_t t3,Int_t patch)
{
  // Set digits threshold
  if(patch==-1)
    {
      Int_t i=0;
      while(i < 6)
	fThreshold[i++]=t3;
      return;
    }
  fThreshold[patch]=t3;
}

void AliHLTTPCHough::SetPeakThreshold(Int_t threshold,Int_t patch)
{
  // Set Peak Finder threshold
  if(patch==-1)
    {
      Int_t i=0;
      while(i < 6)
	fPeakThreshold[i++]=threshold;
      return;
    }
  fPeakThreshold[patch]=threshold;
}

void AliHLTTPCHough::DoBench(Char_t *name)
{
  fBenchmark->Analyze(name);
}

void AliHLTTPCHough::Process(Int_t minslice,Int_t maxslice)
{
  //Process all slices [minslice,maxslice].
#ifdef HAVE_ALIHLTHOUGHGLOBALMERGER
  fGlobalMerger = new AliHLTHoughGlobalMerger(minslice,maxslice);
#else
  return;
#endif //HAVE_ALIHLTHOUGHGLOBALMERGER

  for(Int_t i=minslice; i<=maxslice; i++)
    {
      ReadData(i);
      Transform();
      if(fAddHistograms) {
	if(fVersion != 4)
	  AddAllHistograms();
	else
	  AddAllHistogramsRows();
      }
      FindTrackCandidates();
      //Evaluate();
      //fGlobalMerger->FillTracks(fTracks[0],i);
    }
}

void AliHLTTPCHough::ReadData(Int_t slice,Int_t eventnr)
{
  //Read data from files, binary or root.
  
  if(fEvent!=eventnr) //just be sure that index is empty for new event
    AliHLTTPCFileHandler::CleanStaticIndex(); 
  fCurrentSlice = slice;

  for(Int_t i=0; i<fNPatches; i++)
    {
      fMemHandler[i]->Free();
      UInt_t ndigits=0;
      AliHLTTPCDigitRowData *digits =0;
      Char_t name[256];
      fMemHandler[i]->Init(slice,i);
      if(fBinary)//take input data from binary files
	{
	  if(fUse8bits)
	    sprintf(name,"%s/binaries/digits_c8_%d_%d_%d.raw",fPath,eventnr,slice,i);
	  else
	    sprintf(name,"%s/binaries/digits_%d_%d_%d.raw",fPath,eventnr,slice,i);

	  fMemHandler[i]->SetBinaryInput(name);
	  digits = (AliHLTTPCDigitRowData *)fMemHandler[i]->CompBinary2Memory(ndigits);
	  fMemHandler[i]->CloseBinaryInput();
	}
      else //read data from root file
	{
	  if(fEvent!=eventnr)
	    fMemHandler[i]->FreeDigitsTree();//or else the new event is not loaded
	  digits=(AliHLTTPCDigitRowData *)fMemHandler[i]->AliAltroDigits2Memory(ndigits,eventnr);
	}

      //Set the pointer to the TPCRawStream in case of fast raw data reading
      fHoughTransformer[i]->SetTPCRawStream(fMemHandler[i]->GetTPCRawStream());

      //set input data and init transformer
      fHoughTransformer[i]->SetInputData(ndigits,digits);
      fHoughTransformer[i]->Init(slice,i,fNEtaSegments);
    }

  fEvent=eventnr;
}

void AliHLTTPCHough::Transform(Int_t *rowrange)
{
  //Transform all data given to the transformer within the given slice
  //(after ReadData(slice))
  
  Double_t initTime,cpuTime;
  initTime = GetCpuTime();
  Int_t patchorder[6] = {5,2,0,1,3,4}; //The order in which patches are processed
  //  Int_t patchorder[6] = {0,1,2,3,4,5}; //The order in which patches are processed
  //  Int_t patchorder[6] = {5,4,3,2,1,0}; //The order in which patches are processed
  //  Int_t patchorder[6] = {5,2,4,3,1,0}; //The order in which patches are processed
  fLastPatch=-1;
  for(Int_t i=0; i<fNPatches; i++)
    {
      // In case of Row transformer reset the arrays only once
      if((fVersion != 4) || (i == 0)) {
	fBenchmark->Start("Hough Reset");
	fHoughTransformer[0]->Reset();//Reset the histograms
	fBenchmark->Stop("Hough Reset");
      }
      fBenchmark->Start("Hough Transform");
      PrepareForNextPatch(patchorder[i]);
      if(!rowrange) {
	char buf[256];
	sprintf(buf,"Patch %d",patchorder[i]);
	fBenchmark->Start(buf);
	fHoughTransformer[patchorder[i]]->SetLastPatch(fLastPatch);
	fHoughTransformer[patchorder[i]]->TransformCircle();
	fBenchmark->Stop(buf);
      }
      else
	fHoughTransformer[i]->TransformCircleC(rowrange,1);
      fBenchmark->Stop("Hough Transform");
      fLastPatch=patchorder[i];
    }
  cpuTime = GetCpuTime() - initTime;
  LOG(AliHLTTPCLog::kInformational,"AliHLTTPCHough::Transform()","Timing")
    <<"Transform done in average per patch of "<<cpuTime*1000/fNPatches<<" ms"<<ENDLOG;
}

void AliHLTTPCHough::MergePatches()
{
  // Merge patches if they are not summed
  if(fAddHistograms) //Nothing to merge here
    return;
  if (fMerger==NULL) return;
#ifdef HAVE_ALIHLTHOUGHMERGER
  fMerger->MergePatches(kTRUE);
#endif // HAVE_ALIHLTHOUGHMERGER
}

void AliHLTTPCHough::MergeInternally()
{
  // Merge patches internally
  if (fMerger==NULL) return;
#ifdef HAVE_ALIHLTHOUGHINTMERGER
  if(fAddHistograms)
    fInterMerger->FillTracks(fTracks[0]);
  else {
#ifdef HAVE_ALIHLTHOUGHMERGER
    fInterMerger->FillTracks(fMerger->GetOutTracks());
#endif // HAVE_ALIHLTHOUGHMERGER
  }
  
  fInterMerger->MMerge();
#endif // HAVE_ALIHLTHOUGHINTMERGER
}

void AliHLTTPCHough::ProcessSliceIter()
{
  //Process current slice (after ReadData(slice)) iteratively.
  
  if(!fAddHistograms)
    {
      if (fMerger==NULL) return;
      for(Int_t i=0; i<fNPatches; i++)
	{
	  ProcessPatchIter(i);
#ifdef HAVE_ALIHLTHOUGHMERGER
	  fMerger->FillTracks(fTracks[i],i); //Copy tracks to merger
#endif // HAVE_ALIHLTHOUGHMERGER
	}
    }
  else
    {
      for(Int_t i=0; i<10; i++)
	{
	  Transform();
	  AddAllHistograms();
	  InitEvaluate();
	  AliHLTTPCHoughTransformer *tr = fHoughTransformer[0];
	  for(Int_t j=0; j<fNEtaSegments; j++)
	    {
	      AliHLTTPCHistogram *hist = tr->GetHistogram(j);
	      if(hist->GetNEntries()==0) continue;
	      fPeakFinder->Reset();
	      fPeakFinder->SetHistogram(hist);
	      fPeakFinder->FindAbsMaxima();
	      AliHLTTPCHoughTrack *track = (AliHLTTPCHoughTrack*)fTracks[0]->NextTrack();
	      track->SetTrackParameters(fPeakFinder->GetXPeak(0),fPeakFinder->GetYPeak(0),fPeakFinder->GetWeight(0));
	      track->SetEtaIndex(j);
	      track->SetEta(tr->GetEta(j,fCurrentSlice));
	      for(Int_t k=0; k<fNPatches; k++)
		{
		  fEval[i]->SetNumOfPadsToLook(2);
		  fEval[i]->SetNumOfRowsToMiss(2);
		  fEval[i]->RemoveFoundTracks();
		  /*
		  Int_t nrows=0;
		  if(!fEval[i]->LookInsideRoad(track,nrows))
		    {
		      fTracks[0]->Remove(fTracks[0]->GetNTracks()-1);
		      fTracks[0]->Compress();
		    }
		  */
		}
	    }
	  
	}
      
    }
}

void AliHLTTPCHough::ProcessPatchIter(Int_t patch)
{
  //Process patch in a iterative way. 
  //transform + peakfinding + evaluation + transform +...

  Int_t numoftries = 5;
  AliHLTTPCHoughTransformer *tr = fHoughTransformer[patch];
  AliHLTTPCTrackArray *tracks = fTracks[patch];
  tracks->Reset();
  AliHLTTPCHoughEval *ev = fEval[patch];
  ev->InitTransformer(tr);
  //ev->RemoveFoundTracks();
  ev->SetNumOfRowsToMiss(3);
  ev->SetNumOfPadsToLook(2);
  AliHLTTPCHistogram *hist;
  for(Int_t t=0; t<numoftries; t++)
    {
      tr->Reset();
      tr->TransformCircle();
      for(Int_t i=0; i<fNEtaSegments; i++)
	{
	  hist = tr->GetHistogram(i);
	  if(hist->GetNEntries()==0) continue;
	  fPeakFinder->Reset();
	  fPeakFinder->SetHistogram(hist);
	  fPeakFinder->FindAbsMaxima();
	  //fPeakFinder->FindPeak1();
	  AliHLTTPCHoughTrack *track = (AliHLTTPCHoughTrack*)tracks->NextTrack();
	  track->SetTrackParameters(fPeakFinder->GetXPeak(0),fPeakFinder->GetYPeak(0),fPeakFinder->GetWeight(0));
	  track->SetEtaIndex(i);
	  track->SetEta(tr->GetEta(i,fCurrentSlice));
	  /*
	  Int_t nrows=0;
	  if(!ev->LookInsideRoad(track,nrows))
	    {	
	      tracks->Remove(tracks->GetNTracks()-1);
	      tracks->Compress();
	    }
	  */
	}
    }
  fTracks[0]->QSort();
  LOG(AliHLTTPCLog::kInformational,"AliHLTTPCHough::ProcessPatch","NTracks")
    <<AliHLTTPCLog::kDec<<"Found "<<tracks->GetNTracks()<<" tracks in patch "<<patch<<ENDLOG;
}

void AliHLTTPCHough::AddAllHistograms()
{
  //Add the histograms within one etaslice.
  //Resulting histogram are in patch=0.

  Double_t initTime,cpuTime;
  initTime = GetCpuTime();
  fBenchmark->Start("Add Histograms");
  for(Int_t i=0; i<fNEtaSegments; i++)
    {
      AliHLTTPCHistogram *hist0 = fHoughTransformer[0]->GetHistogram(i);
      for(Int_t j=1; j<fNPatches; j++)
	{
	  AliHLTTPCHistogram *hist = fHoughTransformer[j]->GetHistogram(i);
	  hist0->Add(hist);
	}
    }
  fBenchmark->Stop("Add Histograms");
  fAddHistograms = kTRUE;
  cpuTime = GetCpuTime() - initTime;
  LOG(AliHLTTPCLog::kInformational,"AliHLTTPCHough::AddAllHistograms()","Timing")
    <<"Adding histograms in "<<cpuTime*1000<<" ms"<<ENDLOG;
}

void AliHLTTPCHough::AddAllHistogramsRows()
{
  //Add the histograms within one etaslice.
  //Resulting histogram are in patch=0.

  Double_t initTime,cpuTime;
  initTime = GetCpuTime();
  fBenchmark->Start("Add HistogramsRows");

  UChar_t lastpatchlastrow = AliHLTTPCTransform::GetLastRowOnDDL(fLastPatch)+1;

  UChar_t *tracklastrow = ((AliHLTTPCHoughTransformerRow *)fHoughTransformer[0])->GetTrackLastRow();

  for(Int_t i=0; i<fNEtaSegments; i++)
    {
      UChar_t *gapcount = ((AliHLTTPCHoughTransformerRow *)fHoughTransformer[0])->GetGapCount(i);
      UChar_t *currentrowcount = ((AliHLTTPCHoughTransformerRow *)fHoughTransformer[0])->GetCurrentRowCount(i);

      AliHLTTPCHistogram *hist = fHoughTransformer[0]->GetHistogram(i);
      Int_t xmin = hist->GetFirstXbin();
      Int_t xmax = hist->GetLastXbin();
      Int_t ymin = hist->GetFirstYbin();
      Int_t ymax = hist->GetLastYbin();
      Int_t nxbins = hist->GetNbinsX()+2;

      for(Int_t ybin=ymin; ybin<=ymax; ybin++)
	{
	  for(Int_t xbin=xmin; xbin<=xmax; xbin++)
	    {
	      Int_t bin = xbin + ybin*nxbins; //Int_t bin = hist->GetBin(xbin,ybin);
	      if(gapcount[bin] < MAX_N_GAPS) {
		if(tracklastrow[bin] > lastpatchlastrow) {
		  if(lastpatchlastrow > currentrowcount[bin])
		    gapcount[bin] += (lastpatchlastrow-currentrowcount[bin]-1);
		}
		else {
		  if(tracklastrow[bin] > currentrowcount[bin])
		    gapcount[bin] += (tracklastrow[bin]-currentrowcount[bin]-1);
		}
		if(gapcount[bin] < MAX_N_GAPS)
		  hist->AddBinContent(bin,(159-gapcount[bin]));
	      }
	    }
	}
    }

  fBenchmark->Stop("Add HistogramsRows");
  fAddHistograms = kTRUE;
  cpuTime = GetCpuTime() - initTime;
  LOG(AliHLTTPCLog::kInformational,"AliHLTTPCHough::AddAllHistogramsRows()","Timing")
    <<"Adding histograms in "<<cpuTime*1000<<" ms"<<ENDLOG;
}

void AliHLTTPCHough::PrepareForNextPatch(Int_t nextpatch)
{
  // Prepare the parameter space for the processing of
  // the next read patch. According to the already
  // accumulated number of gaps in parameter space
  // bins, the routine updates the dynamic
  // pointers used in order to jump rapidly during the
  // filling of the parameter space.

  char buf[256];
  sprintf(buf,"Prepare For Patch %d",nextpatch);
  fBenchmark->Start(buf);

  UChar_t lastpatchlastrow;
  if(fLastPatch == -1)
    lastpatchlastrow = 0;
  else
    lastpatchlastrow = AliHLTTPCTransform::GetLastRowOnDDL(fLastPatch)+1;
  UChar_t nextpatchfirstrow;
  if(nextpatch==0)
    nextpatchfirstrow = 0;
  else
    nextpatchfirstrow = AliHLTTPCTransform::GetFirstRowOnDDL(nextpatch)-1;

  UChar_t *trackfirstrow = ((AliHLTTPCHoughTransformerRow *)fHoughTransformer[0])->GetTrackFirstRow();
  UChar_t *tracklastrow = ((AliHLTTPCHoughTransformerRow *)fHoughTransformer[0])->GetTrackLastRow();

  for(Int_t i=0; i<fNEtaSegments; i++)
    {
      UChar_t *gapcount = ((AliHLTTPCHoughTransformerRow *)fHoughTransformer[0])->GetGapCount(i);
      UChar_t *currentrowcount = ((AliHLTTPCHoughTransformerRow *)fHoughTransformer[0])->GetCurrentRowCount(i);
      UChar_t *prevbin = ((AliHLTTPCHoughTransformerRow *)fHoughTransformer[0])->GetPrevBin(i);
      UChar_t *nextbin = ((AliHLTTPCHoughTransformerRow *)fHoughTransformer[0])->GetNextBin(i);
      UChar_t *nextrow = ((AliHLTTPCHoughTransformerRow *)fHoughTransformer[0])->GetNextRow(i);

      AliHLTTPCHistogram *hist = fHoughTransformer[0]->GetHistogram(i);
      Int_t xmin = hist->GetFirstXbin();
      Int_t xmax = hist->GetLastXbin();
      Int_t ymin = hist->GetFirstYbin();
      Int_t ymax = hist->GetLastYbin();
      Int_t nxbins = hist->GetNbinsX()+2;

      if(fLastPatch != -1) {
	UChar_t lastyvalue = 0;
	Int_t endybin = ymin - 1;
	for(Int_t ybin=nextrow[ymin]; ybin<=ymax; ybin = nextrow[++ybin])
	  {
	    UChar_t lastxvalue = 0;
	    UChar_t maxvalue = 0;
	    Int_t endxbin = xmin - 1;
	    for(Int_t xbin=xmin; xbin<=xmax; xbin++)
	      {
		Int_t bin = xbin + ybin*nxbins;
		UChar_t value = 0;
		if(gapcount[bin] < MAX_N_GAPS) {
		  if(tracklastrow[bin] > lastpatchlastrow) {
		    if(lastpatchlastrow > currentrowcount[bin])
		      gapcount[bin] += (lastpatchlastrow-currentrowcount[bin]-1);
		  }
		  else {
		    if(tracklastrow[bin] > currentrowcount[bin])
		      gapcount[bin] += (tracklastrow[bin]-currentrowcount[bin]-1);
		  }
		  if(gapcount[bin] < MAX_N_GAPS) {
		    value = 1;
		    maxvalue = 1;
		    if(trackfirstrow[bin] < nextpatchfirstrow)
		      currentrowcount[bin] = nextpatchfirstrow;
		    else
		      currentrowcount[bin] = trackfirstrow[bin];
		  }
		}
		if(value > 0)
		  {
		    nextbin[xbin + ybin*nxbins] = (UChar_t)xbin;
		    prevbin[xbin + ybin*nxbins] = (UChar_t)xbin;
		    if(value > lastxvalue)
		      {
			UChar_t *tempnextbin = nextbin + endxbin + 1 + ybin*nxbins;
			memset(tempnextbin,(UChar_t)xbin,xbin-endxbin-1);
		      }
		    endxbin = xbin;
		  }
		else
		  {
		    prevbin[xbin + ybin*nxbins] = (UChar_t)endxbin;
		  }
		lastxvalue = value;
	      }
	    UChar_t *tempnextbin = nextbin + endxbin + 1 + ybin*nxbins;
	    memset(tempnextbin,(UChar_t)(xmax+1),xmax-endxbin);
	    if(maxvalue > 0)
	      {
		nextrow[ybin] = (UChar_t)ybin;
		if(maxvalue > lastyvalue)
		  {
		    UChar_t *tempnextrow = nextrow + endybin + 1;
		    memset(tempnextrow,(UChar_t)ybin,ybin-endybin-1);
		  }
		endybin = ybin;
	      }
	    lastyvalue = maxvalue;
	  }
	UChar_t *tempnextrow = nextrow + endybin + 1;
	memset(tempnextrow,(UChar_t)(ymax+1),ymax-endybin+1);
      }
      else {
	UChar_t lastyvalue = 0;
	Int_t endybin = ymin - 1;
	for(Int_t ybin=ymin; ybin<=ymax; ybin++)
	  {
	    UChar_t maxvalue = 0;
	    for(Int_t xbin=xmin; xbin<=xmax; xbin++)
	      {
		Int_t bin = xbin + ybin*nxbins;
		if(gapcount[bin] < MAX_N_GAPS) {
		  maxvalue = 1;
		  if(trackfirstrow[bin] < nextpatchfirstrow)
		    currentrowcount[bin] = nextpatchfirstrow;
		  else
		    currentrowcount[bin] = trackfirstrow[bin];
		}
	      }
	    if(maxvalue > 0)
	      {
		nextrow[ybin] = (UChar_t)ybin;
		if(maxvalue > lastyvalue)
		  {
		    UChar_t *tempnextrow = nextrow + endybin + 1;
		    memset(tempnextrow,(UChar_t)ybin,ybin-endybin-1);
		  }
		endybin = ybin;
	      }
	    lastyvalue = maxvalue;
	  }
	UChar_t *tempnextrow = nextrow + endybin + 1;
	memset(tempnextrow,(UChar_t)(ymax+1),ymax-endybin+1);
      }
    }

  fBenchmark->Stop(buf);
}

void AliHLTTPCHough::AddTracks()
{
  // Add current slice slice tracks to the global list of found tracks
  if(!fTracks[0])
    {
      cerr<<"AliHLTTPCHough::AddTracks : No tracks"<<endl;
      return;
    }
  AliHLTTPCTrackArray *tracks = fTracks[0];
  for(Int_t i=0; i<tracks->GetNTracks(); i++)
    {
      AliHLTTPCTrack *track = tracks->GetCheckedTrack(i);
      if(!track) continue;
      if(track->GetNHits()!=1) cerr<<"NHITS "<<track->GetNHits()<<endl;
      UInt_t *ids = track->GetHitNumbers();
      ids[0] = (fCurrentSlice&0x7f)<<25;
    }
  
  fGlobalTracks->AddTracks(fTracks[0],0,fCurrentSlice);
}

void AliHLTTPCHough::FindTrackCandidatesRow()
{
  // Find AliHLTTPCHoughTransformerRow track candidates
  if(fVersion != 4) {
    LOG(AliHLTTPCLog::kError,"AliHLTTPCHough::FindTrackCandidatesRow()","")
      <<"Incompatible Peak Finder version!"<<ENDLOG;
    return;
  }

  //Look for peaks in histograms, and find the track candidates
  Int_t npatches;
  if(fAddHistograms)
    npatches = 1; //Histograms have been added.
  else
    npatches = fNPatches;
  
  Double_t initTime,cpuTime;
  initTime = GetCpuTime();
  fBenchmark->Start("Find Maxima");
  for(Int_t i=0; i<npatches; i++)
    {
      AliHLTTPCHoughTransformer *tr = fHoughTransformer[i];
      AliHLTTPCHistogram *h = tr->GetHistogram(0);
      Float_t deltax = h->GetBinWidthX()*AliHLTTPCHoughTransformerRow::GetDAlpha();
      Float_t deltay = h->GetBinWidthY()*AliHLTTPCHoughTransformerRow::GetDAlpha();
      Float_t deltaeta = (tr->GetEtaMax()-tr->GetEtaMin())/tr->GetNEtaSegments()*AliHLTTPCHoughTransformerRow::GetDEta();
      Float_t zvertex = tr->GetZVertex();
      fTracks[i]->Reset();
      fPeakFinder->Reset();
      
      for(Int_t j=0; j<fNEtaSegments; j++)
	{
	  AliHLTTPCHistogram *hist = tr->GetHistogram(j);
	  if(hist->GetNEntries()==0) continue;
	  fPeakFinder->SetHistogram(hist);
	  fPeakFinder->SetEtaSlice(j);
	  fPeakFinder->SetTrackLUTs(((AliHLTTPCHoughTransformerRow *)tr)->GetTrackNRows(),((AliHLTTPCHoughTransformerRow *)tr)->GetTrackFirstRow(),((AliHLTTPCHoughTransformerRow *)tr)->GetTrackLastRow(),((AliHLTTPCHoughTransformerRow *)tr)->GetNextRow(j));
#ifdef do_mc
	  LOG(AliHLTTPCLog::kInformational,"AliHLTTPCHough::FindTrackCandidates()","")
	    <<"Starting "<<j<<" etaslice"<<ENDLOG;
#endif
	  fPeakFinder->SetThreshold(fPeakThreshold[i]);
	  fPeakFinder->FindAdaptedRowPeaks(1,0,0);//Maxima finder for HoughTransformerRow
	}
  
      for(Int_t k=0; k<fPeakFinder->GetEntries(); k++)
	{
	  //	  if(fPeakFinder->GetWeight(k) < 0) continue;
	  AliHLTTPCHoughTrack *track = (AliHLTTPCHoughTrack*)fTracks[i]->NextTrack();
	  Double_t starteta = tr->GetEta(fPeakFinder->GetStartEta(k),fCurrentSlice);
	  Double_t endeta = tr->GetEta(fPeakFinder->GetEndEta(k),fCurrentSlice);
	  Double_t eta = (starteta+endeta)/2.0;
	  track->SetTrackParametersRow(fPeakFinder->GetXPeak(k),fPeakFinder->GetYPeak(k),eta,fPeakFinder->GetWeight(k));
	  track->SetPterr(deltax); track->SetPsierr(deltay); track->SetTglerr(deltaeta);
	  track->SetBinXY(fPeakFinder->GetXPeak(k),fPeakFinder->GetYPeak(k),fPeakFinder->GetXPeakSize(k),fPeakFinder->GetYPeakSize(k));
	  track->SetZ0(zvertex);
	  Int_t etaindex = (fPeakFinder->GetStartEta(k)+fPeakFinder->GetEndEta(k))/2;
	  track->SetEtaIndex(etaindex);
	  Int_t rows[2];
	  ((AliHLTTPCHoughTransformerRow *)tr)->GetTrackLength(fPeakFinder->GetXPeak(k),fPeakFinder->GetYPeak(k),rows);
	  track->SetRowRange(rows[0],rows[1]);
	  track->SetSector(fCurrentSlice);
	  track->SetSlice(fCurrentSlice);
#ifdef do_mc
	  Int_t label = tr->GetTrackID(etaindex,fPeakFinder->GetXPeak(k),fPeakFinder->GetYPeak(k));
	  track->SetMCid(label);
#endif
	}
      LOG(AliHLTTPCLog::kInformational,"AliHLTTPCHough::FindTrackCandidates()","")
	<<"Found "<<fTracks[i]->GetNTracks()<<" tracks in slice "<<fCurrentSlice<<ENDLOG;
      fTracks[i]->QSort();
    }
  fBenchmark->Stop("Find Maxima");
  cpuTime = GetCpuTime() - initTime;
  LOG(AliHLTTPCLog::kInformational,"AliHLTTPCHough::FindTrackCandidates()","Timing")
    <<"Maxima finding done in "<<cpuTime*1000<<" ms"<<ENDLOG;
}

void AliHLTTPCHough::FindTrackCandidates()
{
  // Find AliHLTTPCHoughTransformer track candidates
  if(fVersion == 4) {
    LOG(AliHLTTPCLog::kError,"AliHLTTPCHough::FindTrackCandidatesRow()","")
      <<"Incompatible Peak Finder version!"<<ENDLOG;
    return;
  }

  Int_t npatches;
  if(fAddHistograms)
    npatches = 1; //Histograms have been added.
  else
    npatches = fNPatches;
  
  Double_t initTime,cpuTime;
  initTime = GetCpuTime();
  fBenchmark->Start("Find Maxima");
  for(Int_t i=0; i<npatches; i++)
    {
      AliHLTTPCHoughTransformer *tr = fHoughTransformer[i];
      fTracks[i]->Reset();
      
      for(Int_t j=0; j<fNEtaSegments; j++)
	{
	  AliHLTTPCHistogram *hist = tr->GetHistogram(j);
	  if(hist->GetNEntries()==0) continue;
	  fPeakFinder->Reset();
	  fPeakFinder->SetHistogram(hist);
#ifdef do_mc
	  cout<<"Starting "<<j<<" etaslice"<<endl;
#endif
	  fPeakFinder->SetThreshold(fPeakThreshold[i]);
	  fPeakFinder->FindAdaptedPeaks(fKappaSpread,fPeakRatio);
	  
	  for(Int_t k=0; k<fPeakFinder->GetEntries(); k++)
	    {
	      AliHLTTPCHoughTrack *track = (AliHLTTPCHoughTrack*)fTracks[i]->NextTrack();
	      track->SetTrackParameters(fPeakFinder->GetXPeak(k),fPeakFinder->GetYPeak(k),fPeakFinder->GetWeight(k));
	      track->SetEtaIndex(j);
	      track->SetEta(tr->GetEta(j,fCurrentSlice));
	      track->SetRowRange(AliHLTTPCTransform::GetFirstRow(0),AliHLTTPCTransform::GetLastRow(5));
	    }
	}
      cout<<"Found "<<fTracks[i]->GetNTracks()<<" tracks in patch "<<i<<endl;
      fTracks[i]->QSort();
    }
  fBenchmark->Stop("Find Maxima");
  cpuTime = GetCpuTime() - initTime;
  LOG(AliHLTTPCLog::kInformational,"AliHLTTPCHough::FindTrackCandidates()","Timing")
    <<"Maxima finding done in "<<cpuTime*1000<<" ms"<<ENDLOG;
}

void AliHLTTPCHough::InitEvaluate()
{
  //Pass the transformer objects to the AliHLTTPCHoughEval objects:
  //This will provide the evaluation objects with all the necessary
  //data and parameters it needs.
  
  for(Int_t i=0; i<fNPatches; i++) 
    fEval[i]->InitTransformer(fHoughTransformer[i]);
}

Int_t AliHLTTPCHough::Evaluate(Int_t roadwidth,Int_t nrowstomiss)
{
  //Evaluate the tracks, by looking along the road in the raw data.
  //If track does not cross all padrows - rows2miss, it is removed from the arrray.
  //If histograms were not added, the check is done locally in patch,
  //meaning that nrowstomiss is the number of padrows the road can miss with respect
  //to the number of rows in the patch.
  //If the histograms were added, the comparison is done globally in the _slice_, 
  //meaing that nrowstomiss is the number of padrows the road can miss with
  //respect to the total number of padrows in the slice.
  //
  //Return value = number of tracks which were removed (only in case of fAddHistograms)
  
  if(!fTracks[0])
    {
      LOG(AliHLTTPCLog::kError,"AliHLTTPCHough::Evaluate","Track Array")
	<<"No tracks to work with..."<<ENDLOG;
      return 0;
    }
  
  Int_t removedtracks=0;
  AliHLTTPCTrackArray *tracks=0;

  if(fAddHistograms)
    {
      tracks = fTracks[0];
      for(Int_t i=0; i<tracks->GetNTracks(); i++)
	{
	  AliHLTTPCTrack *track = tracks->GetCheckedTrack(i);
	  if(!track) continue;
	  track->SetNHits(0);
	}
    }
  
  for(Int_t i=0; i<fNPatches; i++)
    EvaluatePatch(i,roadwidth,nrowstomiss);
  
  //Here we check the tracks globally; 
  //how many good rows (padrows with signal) 
  //did it cross in the slice
  if(fAddHistograms) 
    {
      for(Int_t j=0; j<tracks->GetNTracks(); j++)
	{
	  AliHLTTPCHoughTrack *track = (AliHLTTPCHoughTrack*)tracks->GetCheckedTrack(j);
	  
	  if(track->GetNHits() < AliHLTTPCTransform::GetNRows() - nrowstomiss)
	    {
	      tracks->Remove(j);
	      removedtracks++;
	    }
	}
      tracks->Compress();
      tracks->QSort();
    }
    
  return removedtracks;
}

void AliHLTTPCHough::EvaluatePatch(Int_t i,Int_t roadwidth,Int_t nrowstomiss)
{
  //Evaluate patch i.
  
  fEval[i]->InitTransformer(fHoughTransformer[i]);
  fEval[i]->SetNumOfPadsToLook(roadwidth);
  fEval[i]->SetNumOfRowsToMiss(nrowstomiss);
  //fEval[i]->RemoveFoundTracks();
  
  AliHLTTPCTrackArray *tracks=0;
  
  if(!fAddHistograms)
    tracks = fTracks[i];
  else
    tracks = fTracks[0];
  
  Int_t nrows=0;
  for(Int_t j=0; j<tracks->GetNTracks(); j++)
    {
      AliHLTTPCHoughTrack *track = (AliHLTTPCHoughTrack*)tracks->GetCheckedTrack(j);
      if(!track)
	{
	  LOG(AliHLTTPCLog::kWarning,"AliHLTTPCHough::EvaluatePatch","Track array")
	    <<"Track object missing!"<<ENDLOG;
	  continue;
	} 
      nrows=0;
      Int_t rowrange[2] = {AliHLTTPCTransform::GetFirstRow(i),AliHLTTPCTransform::GetLastRow(i)};
      Bool_t result = fEval[i]->LookInsideRoad(track,nrows,rowrange);
      if(fAddHistograms)
	{
	  Int_t pre=track->GetNHits();
	  track->SetNHits(pre+nrows);
	}
      else//the track crossed too few good padrows (padrows with signal) in the patch, so remove it
	{
	  if(result == kFALSE)
	    tracks->Remove(j);
	}
    }
  
  tracks->Compress();

}

void AliHLTTPCHough::MergeEtaSlices()
{
  //Merge tracks found in neighbouring eta slices.
  //Removes the track with the lower weight.
  
  fBenchmark->Start("Merge Eta-slices");
  AliHLTTPCTrackArray *tracks = fTracks[0];
  if(!tracks)
    {
      cerr<<"AliHLTTPCHough::MergeEtaSlices : No tracks "<<endl;
      return;
    }
  for(Int_t j=0; j<tracks->GetNTracks(); j++)
    {
      AliHLTTPCHoughTrack *track1 = (AliHLTTPCHoughTrack*)tracks->GetCheckedTrack(j);
      if(!track1) continue;
      for(Int_t k=j+1; k<tracks->GetNTracks(); k++)
	{
	  AliHLTTPCHoughTrack *track2 = (AliHLTTPCHoughTrack*)tracks->GetCheckedTrack(k);
	  if(!track2) continue;
	  if(abs(track1->GetEtaIndex() - track2->GetEtaIndex()) != 1) continue;
	  if(fabs(track1->GetKappa()-track2->GetKappa()) < 0.006 && 
	     fabs(track1->GetPsi()- track2->GetPsi()) < 0.1)
	    {
	      //cout<<"Merging track in slices "<<track1->GetEtaIndex()<<" "<<track2->GetEtaIndex()<<endl;
	      if(track1->GetWeight() > track2->GetWeight())
		tracks->Remove(k);
	      else
		tracks->Remove(j);
	    }
	}
    }
  fBenchmark->Stop("Merge Eta-slices");
  tracks->Compress();
}

void AliHLTTPCHough::WriteTracks(Char_t *path)
{
  // Write found tracks into file
  //cout<<"AliHLTTPCHough::WriteTracks : Sorting the tracsk"<<endl;
  //fGlobalTracks->QSort();
  
  Char_t filename[1024];
  sprintf(filename,"%s/tracks_%d.raw",path,fEvent);
  AliHLTTPCMemHandler mem;
  mem.SetBinaryOutput(filename);
  mem.TrackArray2Binary(fGlobalTracks);
  mem.CloseBinaryOutput();
  fGlobalTracks->Reset();
}

void AliHLTTPCHough::WriteTracks(Int_t slice,Char_t *path)
{
  // Write found tracks slice by slice into file
  
  AliHLTTPCMemHandler mem;
  Char_t fname[100];
  if(fAddHistograms)
    {
      sprintf(fname,"%s/tracks_ho_%d_%d.raw",path,fEvent,slice);
      mem.SetBinaryOutput(fname);
      mem.TrackArray2Binary(fTracks[0]);
      mem.CloseBinaryOutput();
    }
  else 
    {
      for(Int_t i=0; i<fNPatches; i++)
	{
	  sprintf(fname,"%s/tracks_ho_%d_%d_%d.raw",path,fEvent,slice,i);
	  mem.SetBinaryOutput(fname);
	  mem.TrackArray2Binary(fTracks[i]);
	  mem.CloseBinaryOutput();
	}
    }
}

Int_t AliHLTTPCHough::FillESD(AliESDEvent *esd)
{
  // Fill the found hough transform tracks
  // into the ESD. The tracks are stored as
  // AliESDHLTtrack objects.

  // No TPC PID so far,assuming pions
  Double_t prob[AliPID::kSPECIES];
  for(Int_t i=0;i<AliPID::kSPECIES;i++) {
    if(i==AliPID::kPion) prob[i]=1.0;
    else prob[i]=0.1;
  }

  if(!fGlobalTracks) return 0;
  Int_t nglobaltracks = 0;
  for(Int_t i=0; i<fGlobalTracks->GetNTracks(); i++)
    {
      AliHLTTPCHoughTrack *tpt = (AliHLTTPCHoughTrack *)fGlobalTracks->GetCheckedTrack(i);
      if(!tpt) continue; 
      
      if(tpt->GetWeight()<0) continue;
      AliHLTTPCHoughKalmanTrack *tpctrack = new AliHLTTPCHoughKalmanTrack(*tpt);
      if(!tpctrack) continue;
      AliESDtrack *esdtrack2 = new AliESDtrack() ; 
      esdtrack2->UpdateTrackParams(tpctrack,AliESDtrack::kTPCin);
      esdtrack2->SetESDpid(prob);
      esd->AddTrack(esdtrack2);
      nglobaltracks++;
      delete esdtrack2;
      delete tpctrack;
    }
  return nglobaltracks;
}

void AliHLTTPCHough::WriteDigits(Char_t *outfile)
{
  //Write the current data to a new rootfile.
  for(Int_t i=0; i<fNPatches; i++)
    {
      AliHLTTPCDigitRowData *tempPt = (AliHLTTPCDigitRowData*)fHoughTransformer[i]->GetDataPointer();
      fMemHandler[i]->AliDigits2RootFile(tempPt,outfile);
    }
}

Double_t AliHLTTPCHough::GetCpuTime()
{
  //Return the Cputime in seconds.
 struct timeval tv;
 gettimeofday( &tv, NULL );
 return tv.tv_sec+(((Double_t)tv.tv_usec)/1000000.);
}

void *AliHLTTPCHough::ProcessInThread(void *args)
{
  // Called in case Hough transform tracking
  // is executed in a thread

  AliHLTTPCHough *instance = (AliHLTTPCHough *)args;
  Int_t minslice = instance->GetMinSlice();
  Int_t maxslice = instance->GetMaxSlice();
  for(Int_t i=minslice; i<=maxslice; i++)
    {
      instance->ReadData(i,0);
      instance->Transform();
      instance->AddAllHistogramsRows();
      instance->FindTrackCandidatesRow();
      instance->AddTracks();
    }
  return (void *)0;
}

void AliHLTTPCHough::StartProcessInThread(Int_t minslice,Int_t maxslice)
{
  // Starts the Hough transform tracking as a
  // separate thread. Takes as parameters the
  // range of TPC slices (sectors) to be reconstructed

#ifdef HAVE_THREAD
  if(!fThread) {
    char buf[255];
    sprintf(buf,"houghtrans_%d_%d",minslice,maxslice);
    SetMinMaxSlices(minslice,maxslice);
    //    fThread = new TThread(buf,(void (*) (void *))&ProcessInThread,(void *)this);
    fThread = new TThread(buf,&ProcessInThread,(void *)this);
    fThread->Run();
  }
#else  // HAVE_THREAD
  AliErrorClassStream() << "thread support not compiled" << endl;  
#endif // HAVE_THREAD
  return;
}

Int_t AliHLTTPCHough::WaitForThreadFinish()
{
  // Routine is used in case we run the
  // Hough transform tracking in several
  // threads and want to sync them before
  // writing the results to the ESD

#ifdef HAVE_THREAD
#if ROOT_VERSION_CODE < 262403
  return TThread::Join(fThread->GetId());
#else
  return fThread->Join(fThread->GetId());
#endif
#endif // HAVE_THREAD
  AliErrorClassStream() << "thread support not compiled" << endl;
  return 0;
}
