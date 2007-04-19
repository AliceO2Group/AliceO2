// @(#) $Id$
// origin: hough/AliL3Hough.cxx,v 1.50 Tue Mar 28 18:05:12 2006 UTC by alibrary

// Author: Anders Vestbo <mailto:vestbo@fi.uib.no>
//*-- Copyright &copy ALICE HLT Group

#include "AliHLTStandardIncludes.h"
#include <sys/time.h>

#include "AliHLTLogging.h"
#include "AliHLTHoughMerger.h"
#include "AliHLTHoughIntMerger.h"
#include "AliHLTHoughGlobalMerger.h"
#include "AliHLTHistogram.h"
#include "AliHLTHough.h"
#include "AliHLTHoughTransformer.h"
//#include "AliHLTHoughClusterTransformer.h"
#include "AliHLTHoughTransformerLUT.h"
#include "AliHLTHoughTransformerVhdl.h"
#include "AliHLTHoughTransformerRow.h"
#include "AliHLTHoughMaxFinder.h"
#include "AliHLTBenchmark.h"
#ifdef use_aliroot
#include "AliHLTFileHandler.h"
#else
#include "AliHLTMemHandler.h"
#endif
#include "AliHLTDataHandler.h"
#include "AliHLTDigitData.h"
#include "AliHLTHoughEval.h"
#include "AliHLTTransform.h"
#include "AliHLTTrackArray.h"
#include "AliHLTHoughTrack.h"
#include "AliHLTDDLDataFileHandler.h"
#include "AliHLTHoughKalmanTrack.h"

#include "TThread.h"

#if __GNUC__ >= 3
using namespace std;
#endif

/** /class AliHLTTPCHough
//<pre>
//_____________________________________________________________
// AliHLTTPCHough
//
// Interface class for the Hough transform
//
// Example how to use:
//
// AliHLTTPCHough *hough = new AliHLTTPCHough(path,kTRUE,NumberOfEtaSegments);
// hough->ReadData(slice);
// hough->Transform();
// hough->FindTrackCandidates();
// 
// AliHLTTrackArray *tracks = hough->GetTracks(patch);
//
//</pre>
*/

ClassImp(AliHLTTPCHough)

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
#ifdef use_aliroot
  //just be sure that index is empty for new event
    AliHLTFileHandler::CleanStaticIndex(); 
#ifdef use_newio
    fRunLoader = 0;
#endif
#endif
  fThread = 0;
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
#ifdef use_aliroot
  //just be sure that index is empty for new event
    AliHLTFileHandler::CleanStaticIndex(); 
#ifdef use_newio
    fRunLoader = 0;
#endif
#endif
  fThread = 0;
}

AliHLTTPCHough::~AliHLTTPCHough()
{
  //dtor

  CleanUp();
  if(fMerger)
    delete fMerger;
  //cout << "Cleaned class merger " << endl;
  if(fInterMerger)
    delete fInterMerger;
  //cout << "Cleaned class inter " << endl;
  if(fPeakFinder)
    delete fPeakFinder;
  //cout << "Cleaned class peak " << endl;
  if(fGlobalMerger)
    delete fGlobalMerger;
  //cout << "Cleaned class global " << endl;
  if(fBenchmark)
    delete fBenchmark;
  //cout << "Cleaned class bench " << endl;
  if(fGlobalTracks)
    delete fGlobalTracks;
  //cout << "Cleaned class globaltracks " << endl;
  if(fThread) {
    //    fThread->Delete();
    delete fThread;
    fThread = 0;
  }
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

  fNPatches = AliHLTTransform::GetNPatches();
  fHoughTransformer = new AliHLTHoughBaseTransformer*[fNPatches];
  fMemHandler = new AliHLTMemHandler*[fNPatches];

  fTracks = new AliHLTTrackArray*[fNPatches];
  fEval = new AliHLTHoughEval*[fNPatches];
  
  fGlobalTracks = new AliHLTTrackArray("AliHLTHoughTrack");
  
  AliHLTHoughBaseTransformer *lasttransformer = 0;

  for(Int_t i=0; i<fNPatches; i++)
    {
      switch (fVersion){ //choose Transformer
      case 1: 
	fHoughTransformer[i] = new AliHLTHoughTransformerLUT(0,i,fNEtaSegments);
	break;
      case 2:
	//fHoughTransformer[i] = new AliHLTHoughClusterTransformer(0,i,fNEtaSegments);
	break;
      case 3:
	fHoughTransformer[i] = new AliHLTHoughTransformerVhdl(0,i,fNEtaSegments,fNSaveIterations);
	break;
      case 4:
	fHoughTransformer[i] = new AliHLTHoughTransformerRow(0,i,fNEtaSegments,kFALSE,fZVertex);
	break;
      default:
	fHoughTransformer[i] = new AliHLTHoughTransformer(0,i,fNEtaSegments,kFALSE,kFALSE);
      }

      fHoughTransformer[i]->SetLastTransformer(lasttransformer);
      lasttransformer = fHoughTransformer[i];
      //      fHoughTransformer[i]->CreateHistograms(fNBinX[i],fLowPt[i],fNBinY[i],-fPhi[i],fPhi[i]);
      fHoughTransformer[i]->CreateHistograms(fNBinX[i],-fLowPt[i],fLowPt[i],fNBinY[i],-fPhi[i],fPhi[i]);
      //fHoughTransformer[i]->CreateHistograms(fLowPt[i],fUpperPt[i],fPtRes[i],fNBinY[i],fPhi[i]);

      fHoughTransformer[i]->SetLowerThreshold(fThreshold[i]);
      fHoughTransformer[i]->SetUpperThreshold(100);

      LOG(AliHLTLog::kInformational,"AliHLTTPCHough::Init","Version")
	<<"Initializing Hough transformer version "<<fVersion<<ENDLOG;
      
      fEval[i] = new AliHLTHoughEval();
      fTracks[i] = new AliHLTTrackArray("AliHLTHoughTrack");
      if(fUse8bits)
	fMemHandler[i] = new AliHLTDataHandler();
      else
#ifdef use_aliroot
      	{
	  if(!fRawEvent) {
	    if(!fInputFile) {
	      if(!fInputPtr) {
		/* In case of reading digits file */
		fMemHandler[i] = new AliHLTFileHandler(kTRUE); //use static index
		if(!fBinary) {
#if use_newio
		  if(!fRunLoader) {
#endif
		    Char_t filename[1024];
		    sprintf(filename,"%s/digitfile.root",fPath);
		    fMemHandler[i]->SetAliInput(filename);
#if use_newio
		  }
		  else {
		    fMemHandler[i]->SetAliInput(fRunLoader);
		  }
#endif
		}
	      }
	      else {
		/* In case of reading from DATE */
		fMemHandler[i] = new AliHLTDDLDataFileHandler();
		fMemHandler[i]->SetReaderInput(fInputPtr,-1);
	      }
	    }
	    else {
	      /* In case of reading rawdata from ROOT file */
	      fMemHandler[i] = new AliHLTDDLDataFileHandler();
	      fMemHandler[i]->SetReaderInput(fInputFile);
	    }
	  }
	  else {
	    /* In case of reading rawdata using AliRawEvent */
	    fMemHandler[i] = new AliHLTDDLDataFileHandler();
	    fMemHandler[i]->SetReaderInput(fRawEvent);
	  }
	}
#else
      fMemHandler[i] = new AliHLTMemHandler();
#endif
    }

  fPeakFinder = new AliHLTHoughMaxFinder("KappaPhi",50000);
  if(fVersion!=4) {
    fMerger = new AliHLTHoughMerger(fNPatches);
    fInterMerger = new AliHLTHoughIntMerger();
  }
  else {
    fMerger = 0;
    fInterMerger = 0;
  }
  fGlobalMerger = 0;
  fBenchmark = new AliHLTBenchmark();
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
    mrow = AliHLTTransform::GetLastRow(patch);
  if(ptmin)
    {
      Double_t lineradius = sqrt(pow(AliHLTTransform::Row2X(mrow),2) + pow(AliHLTTransform::GetMaxY(mrow),2));
      Double_t kappa = -1*AliHLTTransform::GetBField()*AliHLTTransform::GetBFact()/ptmin;
      psi = AliHLTTransform::Deg2Rad(10) - asin(lineradius*kappa/2);
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
  Double_t lineradius = sqrt(pow(AliHLTTransform::Row2X(mrow),2) + pow(AliHLTTransform::GetMaxY(mrow),2));
  Double_t kappa = -1*AliHLTTransform::GetBField()*AliHLTTransform::GetBFact()/ptmin;
  Double_t psi = AliHLTTransform::Deg2Rad(10) - asin(lineradius*kappa/2);
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

  Double_t lineradius = 1.0/(AliHLTHoughTransformerRow::GetBeta1()*sqrt(1.0+tan(AliHLTTransform::Pi()*10/180)*tan(AliHLTTransform::Pi()*10/180)));
  Double_t alpha1 = AliHLTHoughTransformerRow::GetBeta1()*tan(AliHLTTransform::Pi()*10/180);
  Double_t kappa = 1*AliHLTTransform::GetBField()*AliHLTTransform::GetBFact()/(ptmin*0.9);
  Double_t psi = AliHLTTransform::Deg2Rad(10) - asin(lineradius*kappa/2);
  //  cout<<"Calculated psi range "<<psi<<" in patch "<<patch<<endl;
  Double_t alpha2 = alpha1 - (AliHLTHoughTransformerRow::GetBeta1()-AliHLTHoughTransformerRow::GetBeta2())*tan(psi);
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

  Double_t lineradius = 1.0/(AliHLTHoughTransformerRow::GetBeta1()*sqrt(1.0+tan(AliHLTTransform::Pi()*10/180)*tan(AliHLTTransform::Pi()*10/180)));
  Double_t alpha1 = AliHLTHoughTransformerRow::GetBeta1()*tan(AliHLTTransform::Pi()*10/180);
  Double_t kappa = 1*AliHLTTransform::GetBField()*AliHLTTransform::GetBFact()/(ptmin*0.9);
  Double_t psi = AliHLTTransform::Deg2Rad(10) - asin(lineradius*kappa/2);
  //  cout<<"Calculated psi range "<<psi<<endl;
  Double_t alpha2 = alpha1 - (AliHLTHoughTransformerRow::GetBeta1()-AliHLTHoughTransformerRow::GetBeta2())*tan(psi);
  alpha1 *= 1.1;
  //  cout<<"Calculated alphas range "<<alpha1<<" "<<alpha2<<endl;

  Double_t sizex = 2.0*AliHLTTransform::GetPadPitchWidthLow()*AliHLTHoughTransformerRow::GetBeta1()*AliHLTHoughTransformerRow::GetBeta1();
  Double_t sizey = 2.5*AliHLTTransform::GetPadPitchWidthUp()*AliHLTHoughTransformerRow::GetBeta2()*AliHLTHoughTransformerRow::GetBeta2();

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
  fGlobalMerger = new AliHLTHoughGlobalMerger(minslice,maxslice);
  
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
  
#ifdef use_aliroot
  if(fEvent!=eventnr) //just be sure that index is empty for new event
    AliHLTFileHandler::CleanStaticIndex(); 
#endif
  fCurrentSlice = slice;

  for(Int_t i=0; i<fNPatches; i++)
    {
      fMemHandler[i]->Free();
      UInt_t ndigits=0;
      AliHLTDigitRowData *digits =0;
      Char_t name[256];
      fMemHandler[i]->Init(slice,i);
      if(fBinary)//take input data from binary files
	{
	  if(fUse8bits)
	    sprintf(name,"%s/binaries/digits_c8_%d_%d_%d.raw",fPath,eventnr,slice,i);
	  else
	    sprintf(name,"%s/binaries/digits_%d_%d_%d.raw",fPath,eventnr,slice,i);

	  fMemHandler[i]->SetBinaryInput(name);
	  digits = (AliHLTDigitRowData *)fMemHandler[i]->CompBinary2Memory(ndigits);
	  fMemHandler[i]->CloseBinaryInput();
	}
      else //read data from root file
	{
#ifdef use_aliroot
	  if(fEvent!=eventnr)
	    fMemHandler[i]->FreeDigitsTree();//or else the new event is not loaded
	  digits=(AliHLTDigitRowData *)fMemHandler[i]->AliAltroDigits2Memory(ndigits,eventnr);
#else
	  cerr<<"You cannot read from rootfile now"<<endl;
#endif
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
  LOG(AliHLTLog::kInformational,"AliHLTTPCHough::Transform()","Timing")
    <<"Transform done in average per patch of "<<cpuTime*1000/fNPatches<<" ms"<<ENDLOG;
}

void AliHLTTPCHough::MergePatches()
{
  // Merge patches if they are not summed
  if(fAddHistograms) //Nothing to merge here
    return;
  fMerger->MergePatches(kTRUE);
}

void AliHLTTPCHough::MergeInternally()
{
  // Merge patches internally
  if(fAddHistograms)
    fInterMerger->FillTracks(fTracks[0]);
  else
    fInterMerger->FillTracks(fMerger->GetOutTracks());
  
  fInterMerger->MMerge();
}

void AliHLTTPCHough::ProcessSliceIter()
{
  //Process current slice (after ReadData(slice)) iteratively.
  
  if(!fAddHistograms)
    {
      for(Int_t i=0; i<fNPatches; i++)
	{
	  ProcessPatchIter(i);
	  fMerger->FillTracks(fTracks[i],i); //Copy tracks to merger
	}
    }
  else
    {
      for(Int_t i=0; i<10; i++)
	{
	  Transform();
	  AddAllHistograms();
	  InitEvaluate();
	  AliHLTHoughBaseTransformer *tr = fHoughTransformer[0];
	  for(Int_t j=0; j<fNEtaSegments; j++)
	    {
	      AliHLTHistogram *hist = tr->GetHistogram(j);
	      if(hist->GetNEntries()==0) continue;
	      fPeakFinder->Reset();
	      fPeakFinder->SetHistogram(hist);
	      fPeakFinder->FindAbsMaxima();
	      AliHLTHoughTrack *track = (AliHLTHoughTrack*)fTracks[0]->NextTrack();
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
  AliHLTHoughBaseTransformer *tr = fHoughTransformer[patch];
  AliHLTTrackArray *tracks = fTracks[patch];
  tracks->Reset();
  AliHLTHoughEval *ev = fEval[patch];
  ev->InitTransformer(tr);
  //ev->RemoveFoundTracks();
  ev->SetNumOfRowsToMiss(3);
  ev->SetNumOfPadsToLook(2);
  AliHLTHistogram *hist;
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
	  AliHLTHoughTrack *track = (AliHLTHoughTrack*)tracks->NextTrack();
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
  LOG(AliHLTLog::kInformational,"AliHLTTPCHough::ProcessPatch","NTracks")
    <<AliHLTLog::kDec<<"Found "<<tracks->GetNTracks()<<" tracks in patch "<<patch<<ENDLOG;
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
      AliHLTHistogram *hist0 = fHoughTransformer[0]->GetHistogram(i);
      for(Int_t j=1; j<fNPatches; j++)
	{
	  AliHLTHistogram *hist = fHoughTransformer[j]->GetHistogram(i);
	  hist0->Add(hist);
	}
    }
  fBenchmark->Stop("Add Histograms");
  fAddHistograms = kTRUE;
  cpuTime = GetCpuTime() - initTime;
  LOG(AliHLTLog::kInformational,"AliHLTTPCHough::AddAllHistograms()","Timing")
    <<"Adding histograms in "<<cpuTime*1000<<" ms"<<ENDLOG;
}

void AliHLTTPCHough::AddAllHistogramsRows()
{
  //Add the histograms within one etaslice.
  //Resulting histogram are in patch=0.

  Double_t initTime,cpuTime;
  initTime = GetCpuTime();
  fBenchmark->Start("Add HistogramsRows");

  UChar_t lastpatchlastrow = AliHLTTransform::GetLastRowOnDDL(fLastPatch)+1;

  UChar_t *tracklastrow = ((AliHLTHoughTransformerRow *)fHoughTransformer[0])->GetTrackLastRow();

  for(Int_t i=0; i<fNEtaSegments; i++)
    {
      UChar_t *gapcount = ((AliHLTHoughTransformerRow *)fHoughTransformer[0])->GetGapCount(i);
      UChar_t *currentrowcount = ((AliHLTHoughTransformerRow *)fHoughTransformer[0])->GetCurrentRowCount(i);

      AliHLTHistogram *hist = fHoughTransformer[0]->GetHistogram(i);
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
  LOG(AliHLTLog::kInformational,"AliHLTTPCHough::AddAllHistogramsRows()","Timing")
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
    lastpatchlastrow = AliHLTTransform::GetLastRowOnDDL(fLastPatch)+1;
  UChar_t nextpatchfirstrow;
  if(nextpatch==0)
    nextpatchfirstrow = 0;
  else
    nextpatchfirstrow = AliHLTTransform::GetFirstRowOnDDL(nextpatch)-1;

  UChar_t *trackfirstrow = ((AliHLTHoughTransformerRow *)fHoughTransformer[0])->GetTrackFirstRow();
  UChar_t *tracklastrow = ((AliHLTHoughTransformerRow *)fHoughTransformer[0])->GetTrackLastRow();

  for(Int_t i=0; i<fNEtaSegments; i++)
    {
      UChar_t *gapcount = ((AliHLTHoughTransformerRow *)fHoughTransformer[0])->GetGapCount(i);
      UChar_t *currentrowcount = ((AliHLTHoughTransformerRow *)fHoughTransformer[0])->GetCurrentRowCount(i);
      UChar_t *prevbin = ((AliHLTHoughTransformerRow *)fHoughTransformer[0])->GetPrevBin(i);
      UChar_t *nextbin = ((AliHLTHoughTransformerRow *)fHoughTransformer[0])->GetNextBin(i);
      UChar_t *nextrow = ((AliHLTHoughTransformerRow *)fHoughTransformer[0])->GetNextRow(i);

      AliHLTHistogram *hist = fHoughTransformer[0]->GetHistogram(i);
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
  AliHLTTrackArray *tracks = fTracks[0];
  for(Int_t i=0; i<tracks->GetNTracks(); i++)
    {
      AliHLTTrack *track = tracks->GetCheckedTrack(i);
      if(!track) continue;
      if(track->GetNHits()!=1) cerr<<"NHITS "<<track->GetNHits()<<endl;
      UInt_t *ids = track->GetHitNumbers();
      ids[0] = (fCurrentSlice&0x7f)<<25;
    }
  
  fGlobalTracks->AddTracks(fTracks[0],0,fCurrentSlice);
}

void AliHLTTPCHough::FindTrackCandidatesRow()
{
  // Find AliHLTHoughTransformerRow track candidates
  if(fVersion != 4) {
    LOG(AliHLTLog::kError,"AliHLTTPCHough::FindTrackCandidatesRow()","")
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
      AliHLTHoughBaseTransformer *tr = fHoughTransformer[i];
      AliHLTHistogram *h = tr->GetHistogram(0);
      Float_t deltax = h->GetBinWidthX()*AliHLTHoughTransformerRow::GetDAlpha();
      Float_t deltay = h->GetBinWidthY()*AliHLTHoughTransformerRow::GetDAlpha();
      Float_t deltaeta = (tr->GetEtaMax()-tr->GetEtaMin())/tr->GetNEtaSegments()*AliHLTHoughTransformerRow::GetDEta();
      Float_t zvertex = tr->GetZVertex();
      fTracks[i]->Reset();
      fPeakFinder->Reset();
      
      for(Int_t j=0; j<fNEtaSegments; j++)
	{
	  AliHLTHistogram *hist = tr->GetHistogram(j);
	  if(hist->GetNEntries()==0) continue;
	  fPeakFinder->SetHistogram(hist);
	  fPeakFinder->SetEtaSlice(j);
	  fPeakFinder->SetTrackLUTs(((AliHLTHoughTransformerRow *)tr)->GetTrackNRows(),((AliHLTHoughTransformerRow *)tr)->GetTrackFirstRow(),((AliHLTHoughTransformerRow *)tr)->GetTrackLastRow(),((AliHLTHoughTransformerRow *)tr)->GetNextRow(j));
#ifdef do_mc
	  LOG(AliHLTLog::kInformational,"AliHLTTPCHough::FindTrackCandidates()","")
	    <<"Starting "<<j<<" etaslice"<<ENDLOG;
#endif
	  fPeakFinder->SetThreshold(fPeakThreshold[i]);
	  fPeakFinder->FindAdaptedRowPeaks(1,0,0);//Maxima finder for HoughTransformerRow
	}
  
      for(Int_t k=0; k<fPeakFinder->GetEntries(); k++)
	{
	  //	  if(fPeakFinder->GetWeight(k) < 0) continue;
	  AliHLTHoughTrack *track = (AliHLTHoughTrack*)fTracks[i]->NextTrack();
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
	  ((AliHLTHoughTransformerRow *)tr)->GetTrackLength(fPeakFinder->GetXPeak(k),fPeakFinder->GetYPeak(k),rows);
	  track->SetRowRange(rows[0],rows[1]);
	  track->SetSector(fCurrentSlice);
	  track->SetSlice(fCurrentSlice);
#ifdef do_mc
	  Int_t label = tr->GetTrackID(etaindex,fPeakFinder->GetXPeak(k),fPeakFinder->GetYPeak(k));
	  track->SetMCid(label);
#endif
	}
      LOG(AliHLTLog::kInformational,"AliHLTTPCHough::FindTrackCandidates()","")
	<<"Found "<<fTracks[i]->GetNTracks()<<" tracks in slice "<<fCurrentSlice<<ENDLOG;
      fTracks[i]->QSort();
    }
  fBenchmark->Stop("Find Maxima");
  cpuTime = GetCpuTime() - initTime;
  LOG(AliHLTLog::kInformational,"AliHLTTPCHough::FindTrackCandidates()","Timing")
    <<"Maxima finding done in "<<cpuTime*1000<<" ms"<<ENDLOG;
}

void AliHLTTPCHough::FindTrackCandidates()
{
  // Find AliHLTHoughTransformer track candidates
  if(fVersion == 4) {
    LOG(AliHLTLog::kError,"AliHLTTPCHough::FindTrackCandidatesRow()","")
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
      AliHLTHoughBaseTransformer *tr = fHoughTransformer[i];
      fTracks[i]->Reset();
      
      for(Int_t j=0; j<fNEtaSegments; j++)
	{
	  AliHLTHistogram *hist = tr->GetHistogram(j);
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
	      AliHLTHoughTrack *track = (AliHLTHoughTrack*)fTracks[i]->NextTrack();
	      track->SetTrackParameters(fPeakFinder->GetXPeak(k),fPeakFinder->GetYPeak(k),fPeakFinder->GetWeight(k));
	      track->SetEtaIndex(j);
	      track->SetEta(tr->GetEta(j,fCurrentSlice));
	      track->SetRowRange(AliHLTTransform::GetFirstRow(0),AliHLTTransform::GetLastRow(5));
	    }
	}
      cout<<"Found "<<fTracks[i]->GetNTracks()<<" tracks in patch "<<i<<endl;
      fTracks[i]->QSort();
    }
  fBenchmark->Stop("Find Maxima");
  cpuTime = GetCpuTime() - initTime;
  LOG(AliHLTLog::kInformational,"AliHLTTPCHough::FindTrackCandidates()","Timing")
    <<"Maxima finding done in "<<cpuTime*1000<<" ms"<<ENDLOG;
}

void AliHLTTPCHough::InitEvaluate()
{
  //Pass the transformer objects to the AliHLTHoughEval objects:
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
      LOG(AliHLTLog::kError,"AliHLTTPCHough::Evaluate","Track Array")
	<<"No tracks to work with..."<<ENDLOG;
      return 0;
    }
  
  Int_t removedtracks=0;
  AliHLTTrackArray *tracks=0;

  if(fAddHistograms)
    {
      tracks = fTracks[0];
      for(Int_t i=0; i<tracks->GetNTracks(); i++)
	{
	  AliHLTTrack *track = tracks->GetCheckedTrack(i);
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
	  AliHLTHoughTrack *track = (AliHLTHoughTrack*)tracks->GetCheckedTrack(j);
	  
	  if(track->GetNHits() < AliHLTTransform::GetNRows() - nrowstomiss)
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
  
  AliHLTTrackArray *tracks=0;
  
  if(!fAddHistograms)
    tracks = fTracks[i];
  else
    tracks = fTracks[0];
  
  Int_t nrows=0;
  for(Int_t j=0; j<tracks->GetNTracks(); j++)
    {
      AliHLTHoughTrack *track = (AliHLTHoughTrack*)tracks->GetCheckedTrack(j);
      if(!track)
	{
	  LOG(AliHLTLog::kWarning,"AliHLTTPCHough::EvaluatePatch","Track array")
	    <<"Track object missing!"<<ENDLOG;
	  continue;
	} 
      nrows=0;
      Int_t rowrange[2] = {AliHLTTransform::GetFirstRow(i),AliHLTTransform::GetLastRow(i)};
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
  AliHLTTrackArray *tracks = fTracks[0];
  if(!tracks)
    {
      cerr<<"AliHLTTPCHough::MergeEtaSlices : No tracks "<<endl;
      return;
    }
  for(Int_t j=0; j<tracks->GetNTracks(); j++)
    {
      AliHLTHoughTrack *track1 = (AliHLTHoughTrack*)tracks->GetCheckedTrack(j);
      if(!track1) continue;
      for(Int_t k=j+1; k<tracks->GetNTracks(); k++)
	{
	  AliHLTHoughTrack *track2 = (AliHLTHoughTrack*)tracks->GetCheckedTrack(k);
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
  AliHLTMemHandler mem;
  mem.SetBinaryOutput(filename);
  mem.TrackArray2Binary(fGlobalTracks);
  mem.CloseBinaryOutput();
  fGlobalTracks->Reset();
}

void AliHLTTPCHough::WriteTracks(Int_t slice,Char_t *path)
{
  // Write found tracks slice by slice into file
  
  AliHLTMemHandler mem;
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

#ifdef use_aliroot
Int_t AliHLTTPCHough::FillESD(AliESD *esd)
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
      AliHLTHoughTrack *tpt = (AliHLTHoughTrack *)fGlobalTracks->GetCheckedTrack(i);
      if(!tpt) continue; 
      
      if(tpt->GetWeight()<0) continue;
      AliHLTHoughKalmanTrack *tpctrack = new AliHLTHoughKalmanTrack(*tpt);
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
#endif

void AliHLTTPCHough::WriteDigits(Char_t *outfile)
{
  //Write the current data to a new rootfile.
#ifdef use_aliroot  

  for(Int_t i=0; i<fNPatches; i++)
    {
      AliHLTDigitRowData *tempPt = (AliHLTDigitRowData*)fHoughTransformer[i]->GetDataPointer();
      fMemHandler[i]->AliDigits2RootFile(tempPt,outfile);
    }
#else
  cerr<<"AliHLTTPCHough::WriteDigits : You need to compile with AliROOT!"<<endl;
  return;
#endif  
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

  AliHLTHough *instance = (AliHLTHough *)args;
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

  if(!fThread) {
    char buf[255];
    sprintf(buf,"houghtrans_%d_%d",minslice,maxslice);
    SetMinMaxSlices(minslice,maxslice);
    //    fThread = new TThread(buf,(void (*) (void *))&ProcessInThread,(void *)this);
    fThread = new TThread(buf,&ProcessInThread,(void *)this);
    fThread->Run();
  }
  return;
}

Int_t AliHLTTPCHough::WaitForThreadFinish()
{
  // Routine is used in case we run the
  // Hough transform tracking in several
  // threads and want to sync them before
  // writing the results to the ESD

#if ROOT_VERSION_CODE < 262403
  return TThread::Join(fThread->GetId());
#else
  return fThread->Join(fThread->GetId());
#endif
}
