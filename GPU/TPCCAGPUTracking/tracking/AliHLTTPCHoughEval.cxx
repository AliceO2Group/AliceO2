// @(#) $Id$
// origin: hough/AliL3HoughEval.cxx,v 1.28 Thu Jun 17 10:36:14 2004 UTC by cvetan

// Author: Anders Vestbo <mailto:vestbo@fi.uib.no>
//*-- Copyright &copy ALICE HLT Group

#include "AliHLTStdIncludes.h"

#include <TH1.h>
#include <TFile.h>

#include "AliHLTTPCLogging.h"
#include "AliHLTTPCHoughEval.h"
#include "AliHLTTPCMemHandler.h"
#include "AliHLTTPCTrackArray.h"
#include "AliHLTTPCHoughTransformer.h"
#include "AliHLTTPCDigitData.h"
#include "AliHLTTPCHoughTrack.h"
#include "AliHLTTPCTransform.h"
#include "AliHLTTPCHistogram.h"
#include "AliHLTTPCHistogram1D.h"

#if __GNUC__ == 3
using namespace std;
#endif

/** /class AliHLTTPCHoughEval
//<pre>
//_____________________________________________________________
// AliHLTTPCHoughEval
//
// Evaluation class for tracklets produced by the Hough transform.
//
</pre>
*/

ClassImp(AliHLTTPCHoughEval)

AliHLTTPCHoughEval::AliHLTTPCHoughEval()
{
  //default ctor  
  fRemoveFoundTracks = kFALSE;
  fNumOfPadsToLook = 1;
  fNumOfRowsToMiss = 1;
  fEtaHistos=0;
  fRowPointers = 0;
}


AliHLTTPCHoughEval::~AliHLTTPCHoughEval()
{
  //dtor
  fHoughTransformer = 0;
  if(fRowPointers)
    {
      for(Int_t i=0; i<fNrows; i++)
	fRowPointers[i] = 0;
      delete [] fRowPointers;
    }
}

void AliHLTTPCHoughEval::InitTransformer(AliHLTTPCHoughTransformer *transformer)
{
  //Init hough transformer
  fHoughTransformer = transformer;
  fSlice = fHoughTransformer->GetSlice();
  fPatch = fHoughTransformer->GetPatch();
  fNrows = AliHLTTPCTransform::GetLastRow(fPatch) - AliHLTTPCTransform::GetFirstRow(fPatch) + 1;
  fNEtaSegments = fHoughTransformer->GetNEtaSegments();
  fEtaMin = fHoughTransformer->GetEtaMin();
  fEtaMax = fHoughTransformer->GetEtaMax();
  fZVertex = fHoughTransformer->GetZVertex();
  GenerateLUT();
}

void AliHLTTPCHoughEval::GenerateLUT()
{
  //Generate a Look-up table, to limit the access to raw data
  
  if(!fRowPointers)
    fRowPointers = new AliHLTTPCDigitRowData*[fNrows];

  AliHLTTPCDigitRowData *tempPt = (AliHLTTPCDigitRowData*)fHoughTransformer->GetDataPointer();
  if(!tempPt)
    printf("\nAliHLTTPCHoughEval::GenerateLUT : Zero data pointer\n");
  
  for(Int_t i=AliHLTTPCTransform::GetFirstRow(fPatch); i<=AliHLTTPCTransform::GetLastRow(fPatch); i++)
    {
      Int_t prow = i - AliHLTTPCTransform::GetFirstRow(fPatch);
      fRowPointers[prow] = tempPt;
      AliHLTTPCMemHandler::UpdateRowPointer(tempPt);
    }
  
}

Bool_t AliHLTTPCHoughEval::LookInsideRoad(AliHLTTPCHoughTrack *track,Int_t &nrowscrossed,Int_t *rowrange,Bool_t remove)
{
  //Look at rawdata along the road specified by the track candidates.
  //If track is good, return true, if not return false.
  
  Int_t sector,row;
  
  Int_t nrow=0,npixs=0;//,rows_crossed=0;
  Float_t xyz[3];
  
  Int_t totalcharge=0;//total charge along the road
  
  //for(Int_t padrow = AliHLTTPCTransform::GetFirstRow(fPatch); padrow <= AliHLTTPCTransform::GetLastRow(fPatch); padrow++)
  for(Int_t padrow = rowrange[0]; padrow<=rowrange[1]; padrow++)
    {
      Int_t prow = padrow - AliHLTTPCTransform::GetFirstRow(fPatch);
      if(track->IsHelix())
	{
	  if(!track->GetCrossingPoint(padrow,xyz))  
	    {
	      continue;
	    }
	}
      else
	{
	  track->GetLineCrossingPoint(padrow,xyz);
	  xyz[0] += AliHLTTPCTransform::Row2X(track->GetFirstRow());
	  Float_t r = sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1]);
	  xyz[2] = r*track->GetTgl();
	}
      
      AliHLTTPCTransform::Slice2Sector(fSlice,padrow,sector,row);
      AliHLTTPCTransform::Local2Raw(xyz,sector,row);

      npixs=0;
      
      //Get the timebins for this pad
      AliHLTTPCDigitRowData *tempPt = fRowPointers[prow];
      if(!tempPt) 
	{
	  printf("AliHLTTPCHoughEval::LookInsideRoad : Zero data pointer\n");
	  continue;
	}
      
      //Look at both sides of the pad:
      for(Int_t p=(Int_t)rint(xyz[1])-fNumOfPadsToLook; p<=(Int_t)rint(xyz[1])+fNumOfPadsToLook; p++)
	{
	  AliHLTTPCDigitData *digPt = tempPt->fDigitData;
	  for(UInt_t j=0; j<tempPt->fNDigit; j++)
	    {
	      Int_t pad = digPt[j].fPad;
	      Int_t charge = digPt[j].fCharge;
	      if(charge <= fHoughTransformer->GetLowerThreshold()) continue;
	      if(pad < p) continue;
	      if(pad > p) break;
	      UShort_t time = digPt[j].fTime;
	      Double_t eta = AliHLTTPCTransform::GetEta(fSlice,padrow,pad,time);
	      Int_t pixelindex = fHoughTransformer->GetEtaIndex(eta);
	      if(pixelindex != track->GetEtaIndex()) continue;
	      totalcharge += digPt[j].fCharge;
	      if(remove)
		digPt[j].fCharge = 0; //Erease the track from image
	      npixs++;
	    }
	}
            
      if(npixs > 1)//At least 2 digits on this padrow
	{
	  nrow++;
	}	  
    }
  if(remove)
    return kTRUE;
  
  nrowscrossed += nrow; //Update the number of rows crossed.
  
  if(nrow >= rowrange[1]-rowrange[0]+1 - fNumOfRowsToMiss)//this was a good track
    {
      if(fRemoveFoundTracks)
	{
	  Int_t dummy=0;
	  LookInsideRoad(track,dummy,rowrange,kTRUE);
	}
      return kTRUE;
    }
  else
    return kFALSE;
}

void AliHLTTPCHoughEval::FindEta(AliHLTTPCTrackArray *tracks)
{
  //Find the corresponding eta slice hough space  
  Int_t sector,row;
  Float_t xyz[3];
  
  Int_t ntracks = tracks->GetNTracks();
  fEtaHistos = new AliHLTTPCHistogram1D*[ntracks];
  
  Char_t hname[100];
  for(Int_t i=0; i<ntracks; i++)
    {
      sprintf(hname,"etahist_%d",i);
      fEtaHistos[i] = new AliHLTTPCHistogram1D(hname,hname,100,0,1);
    }
  Double_t etaslice = (fEtaMax - fEtaMin)/fNEtaSegments;
  
  for(Int_t ntr=0; ntr<ntracks; ntr++)
    {
      AliHLTTPCHoughTrack *track = (AliHLTTPCHoughTrack*)tracks->GetCheckedTrack(ntr);
      if(!track) continue;
      for(Int_t padrow = AliHLTTPCTransform::GetFirstRow(fPatch); padrow <= AliHLTTPCTransform::GetLastRow(fPatch); padrow++)
	{
	  Int_t prow = padrow - AliHLTTPCTransform::GetFirstRow(fPatch);
	  
	  if(!track->GetCrossingPoint(padrow,xyz))  
	    {
	      printf("AliHLTTPCHoughEval::LookInsideRoad : Track does not cross line!!\n");
	      continue;
	    }
	  
	  AliHLTTPCTransform::Slice2Sector(fSlice,padrow,sector,row);
	  AliHLTTPCTransform::Local2Raw(xyz,sector,row);
	  
	  //Get the timebins for this pad
	  AliHLTTPCDigitRowData *tempPt = fRowPointers[prow];
	  if(!tempPt) 
	    {
	      printf("AliHLTTPCHoughEval::LookInsideRoad : Zero data pointer\n");
	      continue;
	    }
	  
	  //Look at both sides of the pad:
	  for(Int_t p=(Int_t)rint(xyz[1])-fNumOfPadsToLook; p<=(Int_t)rint(xyz[1])+fNumOfPadsToLook; p++)
	    {
	      AliHLTTPCDigitData *digPt = tempPt->fDigitData;
	      for(UInt_t j=0; j<tempPt->fNDigit; j++)
		{
		  UChar_t pad = digPt[j].fPad;
		  Int_t charge = digPt[j].fCharge;
		  if(charge <= fHoughTransformer->GetLowerThreshold()) continue;
		  if(pad < p) continue;
		  if(pad > p) break;
		  UShort_t time = digPt[j].fTime;
		  Double_t eta = AliHLTTPCTransform::GetEta(fSlice,padrow,pad,time);
		  Int_t pixelindex = (Int_t)(eta/etaslice);
		  if(pixelindex > track->GetEtaIndex()+1) continue;
		  if(pixelindex < track->GetEtaIndex()-1) break;
		  fEtaHistos[ntr]->Fill(eta,digPt[j].fCharge);
		}
	    }
	}
    }
  
  for(Int_t i=0; i<ntracks; i++)
    {
      AliHLTTPCHistogram1D *hist = fEtaHistos[i];
      Int_t maxbin = hist->GetMaximumBin();
      Double_t maxvalue = hist->GetBinContent(maxbin);
      AliHLTTPCHoughTrack *track = (AliHLTTPCHoughTrack*)tracks->GetCheckedTrack(i);
      if(!track) continue;
      if(hist->GetBinContent(maxbin-1)<maxvalue && hist->GetBinContent(maxbin+1)<maxvalue)
	{
	  track->SetWeight((Int_t)maxvalue,kTRUE); 
	  track->SetEta(hist->GetBinCenter(maxbin));
	  track->SetNHits(track->GetWeight());
	}
      else
	{
	  track->SetWeight(0);
	  tracks->Remove(i); //remove this track, because it was not a peak
	}    
    }
  tracks->Compress();
  
  //for(Int_t i=0; i<ntracks; i++)
  //delete fEtaHistos[i];
  //delete [] fEtaHistos;
}

void AliHLTTPCHoughEval::DisplayEtaSlice(Int_t etaindex,AliHLTTPCHistogram *hist)
{
  //Display the current raw data inside the (slice,patch)

  if(!hist)
    {
      printf("AliHLTTPCHoughEval::DisplayEtaSlice : No input histogram!\n");
      return;
    }
  
  for(Int_t padrow = AliHLTTPCTransform::GetFirstRow(fPatch); padrow <= AliHLTTPCTransform::GetLastRow(fPatch); padrow++)
    {
      Int_t prow = padrow - AliHLTTPCTransform::GetFirstRow(fPatch);
                  
      AliHLTTPCDigitRowData *tempPt = fRowPointers[prow];
      if(!tempPt) 
	{
	  printf("AliHLTTPCHoughEval::DisplayEtaSlice : Zero data pointer\n");
	  continue;
	}
      
      AliHLTTPCDigitData *digPt = tempPt->fDigitData;
      if((Int_t)tempPt->fRow != padrow)
	{
	  printf("\nAliHLTTPCHoughEval::DisplayEtaSlice : Mismatching padrows!!!\n");
	  return;
	}
      for(UInt_t j=0; j<tempPt->fNDigit; j++)
	{
	  UChar_t pad = digPt[j].fPad;
	  UChar_t charge = digPt[j].fCharge;
	  UShort_t time = digPt[j].fTime;
	  if((Int_t)charge <= fHoughTransformer->GetLowerThreshold() || (Int_t)charge >= fHoughTransformer->GetUpperThreshold()) continue;
	  Float_t xyz[3];
	  Int_t sector,row;
	  AliHLTTPCTransform::Slice2Sector(fSlice,padrow,sector,row);
	  AliHLTTPCTransform::Raw2Local(xyz,sector,row,pad,time);
	  xyz[2] -= fZVertex;
	  Double_t eta = AliHLTTPCTransform::GetEta(xyz);
	  Int_t pixelindex = fHoughTransformer->GetEtaIndex(eta);//(Int_t)(eta/etaslice);
	  if(pixelindex != etaindex) continue;
	  hist->Fill(xyz[0],xyz[1],charge);
	}
    }
  
}

void AliHLTTPCHoughEval::CompareMC(AliHLTTPCTrackArray */*tracks*/,Char_t */*trackfile*/,Int_t /*threshold*/)
{
  /*  
  struct GoodTrack goodtracks[15000];
  Int_t nt=0;
  ifstream in(trackfile);
  if(in)
    {
      printf("Reading good tracks from file %s\n",trackfile);
      while (in>>goodtracks[nt].label>>goodtracks[nt].code>>
	     goodtracks[nt].px>>goodtracks[nt].py>>goodtracks[nt].pz>>
	     goodtracks[nt].pt>>goodtracks[nt].eta>>goodtracks[nt].nhits) 
	{
	  nt++;
	  if (nt==15000) 
	    {
	      cerr<<"Too many good tracks"<<endl;
	      break;
	    }
	}
      if (!in.eof())
	{
	  LOG(AliHLTTPCLog::kError,"AliHLTTPCHoughEval::CompareMC","Input file")
	    <<"Error in file reading"<<ENDLOG;
	  return;
	}
    }
  else
    {
      LOG(AliHLTTPCLog::kError,"AliHLTTPCHoughEval::CompareMC","Input")
	<<"No input trackfile "<<trackfile<<ENDLOG;
    }
  
  Int_t *particles = new Int_t[fNEtaSegments];
  Int_t *ftracks = new Int_t[fNEtaSegments];
  for(Int_t i=0; i<fNEtaSegments; i++)
    {
      particles[i]=0;
      ftracks[i]=0;
    }
  
  TH1F *ptgood = new TH1F("ptgood","ptgood",5,0,2);
  TH1F *ptfound = new TH1F("ptfound","ptgood",5,0,2);
  TH1F *pteff = new TH1F("pteff","pteff",5,0,2);
  TH1F *etafound = new TH1F("etafound","etafound",5,0,1);
  TH1F *etagood = new TH1F("etagood","etagood",5,0,1);
  TH1F *etaeff = new TH1F("etaeff","etaeff",5,0,1);
  
  Double_t etaslice = (fEtaMax - fEtaMin)/fNEtaSegments;
  for(Int_t i=0; i<tracks->GetNTracks(); i++)
    {
      AliHLTTPCHoughTrack *tr = (AliHLTTPCHoughTrack*)tracks->GetCheckedTrack(i);
      if(!tr) continue;
      if(tr->GetWeight()<threshold) continue;
      Int_t trackindex = tr->GetEtaIndex();
      if(trackindex <0 || trackindex >= fNEtaSegments) continue;
      ftracks[trackindex]++;
      ptfound->Fill(tr->GetPt());
      etafound->Fill(tr->GetEta());
    }
  for(Int_t i=0; i<nt; i++)
    {
      if(goodtracks[i].nhits < 174) continue;
      if(goodtracks[i].pt < 0.2) continue;
      Int_t particleindex = (Int_t)(goodtracks[i].eta/etaslice);
      if(particleindex < 0 || particleindex >= fNEtaSegments) continue;
      particles[particleindex]++;
      ptgood->Fill(goodtracks[i].pt);
      etagood->Fill(goodtracks[i].eta);
    }
  
  Double_t found=0;
  Double_t good =0;
  for(Int_t i=0; i<fNEtaSegments; i++)
    {
      //printf("Slice %d : Found tracks %d, good tracks %d\n",i,ftracks[i],particles[i]);
      found += ftracks[i];
      good += particles[i];
    }
  printf("And the total efficiency was: %f\n",found/good);

  ptgood->Sumw2(); ptfound->Sumw2();
  etagood->Sumw2(); etafound->Sumw2();
  pteff->Divide(ptfound,ptgood,1,1,"b");
  etaeff->Divide(etafound,etagood,1,1,"b");
  TFile *file = TFile::Open("eff.root","RECREATE");
  ptgood->Write();
  ptfound->Write();
  pteff->Write();
  etafound->Write();
  etagood->Write();
  etaeff->Write();
  file->Close();
  
  delete [] particles;
  delete [] ftracks;
  */  
}
