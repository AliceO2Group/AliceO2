// @(#) $Id$
// origin: hough/AliL3HoughMaxFinder.cxx,v 1.13 Tue Mar 28 18:05:12 2006 UTC by alibrary 

// Author: Anders Vestbo <mailto:vestbo@fi.uib.no>
//*-- Copyright &copy ALICE HLT Group

#include <strings.h>

#include <TNtuple.h>
#include <TFile.h>

#include "AliHLTTPCLogging.h"
#include "AliHLTTPCHoughMaxFinder.h"
#include "AliHLTTPCHistogram.h"
#include "AliHLTTPCTrackArray.h"
#include "AliHLTTPCHoughTrack.h"
#include "AliHLTTPCTransform.h"
#include "AliHLTTPCHoughTransformerRow.h"

#if __GNUC__ >= 3
using namespace std;
#endif

/** \class AliHLTTPCHoughMaxFinder
<pre>
//_____________________________________________________________
// AliHLTTPCHoughMaxFinder
//
// Maximum finder
//
</pre>
*/

ClassImp(AliHLTTPCHoughMaxFinder)

  
AliHLTTPCHoughMaxFinder::AliHLTTPCHoughMaxFinder()
{
  //Default constructor
  fThreshold = 0;
  fCurrentEtaSlice = -1;
  fHistoType=0;
  fXPeaks=0;
  fYPeaks=0;
  fWeight=0;
  fNPeaks=0;
  fN1PeaksPrevEtaSlice=0;
  fN2PeaksPrevEtaSlice=0;
  fSTARTXPeaks=0;
  fSTARTYPeaks=0;
  fENDXPeaks=0;
  fENDYPeaks=0;
  fSTARTETAPeaks=0;
  fENDETAPeaks=0;
  fNMax=0;
  fGradX=1;
  fGradY=1;
#ifndef no_root
  fNtuppel = 0;
#endif
}

AliHLTTPCHoughMaxFinder::AliHLTTPCHoughMaxFinder(Char_t *histotype,Int_t nmax,AliHLTTPCHistogram *hist)
{
  //Constructor

  //fTracks = new AliHLTTPCTrackArray("AliHLTTPCHoughTrack");
  if(strcmp(histotype,"KappaPhi")==0) fHistoType='c';
  if(strcmp(histotype,"DPsi")==0) fHistoType='l';

  fCurrentEtaSlice = -1;
  
  if(hist)
    fCurrentHisto = hist;
  
  fGradX=1;
  fGradY=1;
  fNMax=nmax;
  fXPeaks = new Float_t[fNMax];
  fYPeaks = new Float_t[fNMax];
  fWeight = new Int_t[fNMax];
  fSTARTXPeaks = new Int_t[fNMax];
  fSTARTYPeaks = new Int_t[fNMax];
  fENDXPeaks = new Int_t[fNMax];
  fENDYPeaks = new Int_t[fNMax];
  fSTARTETAPeaks = new Int_t[fNMax];
  fENDETAPeaks = new Int_t[fNMax];
#ifndef no_root
  fNtuppel = 0;
#endif
  fThreshold=0;
}

AliHLTTPCHoughMaxFinder::~AliHLTTPCHoughMaxFinder()
{
  //Destructor
  if(fXPeaks)
    delete [] fXPeaks;
  if(fYPeaks)
    delete [] fYPeaks;
  if(fWeight)
    delete [] fWeight;
  if(fSTARTXPeaks)
    delete [] fSTARTXPeaks;
  if(fSTARTYPeaks)
    delete [] fSTARTYPeaks;
  if(fENDXPeaks)
    delete [] fENDXPeaks;
  if(fENDYPeaks)
    delete [] fENDYPeaks;
  if(fSTARTETAPeaks)
    delete [] fSTARTETAPeaks;
  if(fENDETAPeaks)
    delete [] fENDETAPeaks;
#ifndef no_root
  if(fNtuppel)
    delete fNtuppel;
#endif
}

void AliHLTTPCHoughMaxFinder::Reset()
{
  // Method to reinit the Peak Finder
  for(Int_t i=0; i<fNMax; i++)
    {
      fXPeaks[i]=0;
      fYPeaks[i]=0;
      fWeight[i]=0;
      fSTARTXPeaks[i]=0;
      fSTARTYPeaks[i]=0;
      fENDXPeaks[i]=0;
      fENDYPeaks[i]=0;
      fSTARTETAPeaks[i]=0;
      fENDETAPeaks[i]=0;
    }
  fNPeaks=0;
  fN1PeaksPrevEtaSlice=0;
  fN2PeaksPrevEtaSlice=0;
}

void AliHLTTPCHoughMaxFinder::CreateNtuppel()
{
  // Fill a NTuple with the peak parameters
#ifndef no_root
  //content#; neighbouring bins of the peak.
  fNtuppel = new TNtuple("ntuppel","Peak charateristics","kappa:phi0:weigth:content3:content5:content1:content7");
  fNtuppel->SetDirectory(0);
#endif  
}

void AliHLTTPCHoughMaxFinder::WriteNtuppel(Char_t *filename)
{
  // Write the NTuple with the peak parameters
#ifndef no_root
  TFile *file = TFile::Open(filename,"RECREATE");
  if(!file)
    {
      cerr<<"AliHLTTPCHoughMaxFinder::WriteNtuppel : Error opening file "<<filename<<endl;
      return;
    }
  fNtuppel->Write();
  file->Close();
#endif
}

void AliHLTTPCHoughMaxFinder::FindAbsMaxima()
{
  // Simple Peak Finder in the Hough space
  if(!fCurrentHisto)
    {
      cerr<<"AliHLTTPCHoughMaxFinder::FindAbsMaxima : No histogram"<<endl;
      return;
    }
  AliHLTTPCHistogram *hist = fCurrentHisto;
  
  if(hist->GetNEntries() == 0)
    return;
  
  Int_t xmin = hist->GetFirstXbin();
  Int_t xmax = hist->GetLastXbin();
  Int_t ymin = hist->GetFirstYbin();
  Int_t ymax = hist->GetLastYbin();  
  Int_t bin;
  Double_t value,maxvalue=0;
  
  Int_t maxxbin=0,maxybin=0;
  for(Int_t xbin=xmin; xbin<=xmax; xbin++)
    {
      for(Int_t ybin=ymin; ybin<=ymax; ybin++)
	{
	  bin = hist->GetBin(xbin,ybin);
	  value = hist->GetBinContent(bin);
	  if(value>maxvalue)
	    {
	      maxvalue = value;
	      maxxbin = xbin;
	      maxybin = ybin;
	    }
	}
    }
  
  if(maxvalue == 0)
    return;
  
  if(fNPeaks > fNMax)
    {
      cerr<<"AliHLTTPCHoughMaxFinder::FindAbsMaxima : Array out of range : "<<fNPeaks<<endl;
      return;
    }
  
  Double_t maxx = hist->GetBinCenterX(maxxbin);
  Double_t maxy = hist->GetBinCenterY(maxybin);
  fXPeaks[fNPeaks] = maxx;
  fYPeaks[fNPeaks] = maxy;
  fWeight[fNPeaks] = (Int_t)maxvalue;

  fNPeaks++;
#ifndef no_root
  if(fNtuppel)
    {
      Int_t bin3 = hist->GetBin(maxxbin-1,maxybin);
      Int_t bin5 = hist->GetBin(maxxbin+1,maxybin);
      Int_t bin1 = hist->GetBin(maxxbin,maxybin-1);
      Int_t bin7 = hist->GetBin(maxxbin,maxybin+1);
      
      fNtuppel->Fill(maxx,maxy,maxvalue,hist->GetBinContent(bin3),hist->GetBinContent(bin5),hist->GetBinContent(bin1),hist->GetBinContent(bin7));
    }
#endif  
}

void AliHLTTPCHoughMaxFinder::FindBigMaxima()
{
  // Another Peak finder  
  AliHLTTPCHistogram *hist = fCurrentHisto;
  
  if(hist->GetNEntries() == 0)
    return;
  
  Int_t xmin = hist->GetFirstXbin();
  Int_t xmax = hist->GetLastXbin();
  Int_t ymin = hist->GetFirstYbin();
  Int_t ymax = hist->GetLastYbin();
  Int_t bin[25],binindex;
  Double_t value[25];
  
  for(Int_t xbin=xmin+2; xbin<xmax-3; xbin++)
    {
      for(Int_t ybin=ymin+2; ybin<ymax-3; ybin++)
	{
	  binindex=0;
	  for(Int_t xb=xbin-2; xb<=xbin+2; xb++)
	    {
	      for(Int_t yb=ybin-2; yb<=ybin+2; yb++)
		{
		  bin[binindex]=hist->GetBin(xb,yb);
		  value[binindex]=hist->GetBinContent(bin[binindex]);
		  binindex++;
		}
	    }
	  if(value[12]==0) continue;
	  Int_t b=0;
	  while(1)
	    {
	      if(value[b] > value[12] || b==binindex) break;
	      b++;
	      //printf("b %d\n",b);
	    }
	  if(b == binindex)
	    {
	      //Found maxima
	      if(fNPeaks > fNMax)
		{
		  cerr<<"AliHLTTPCHoughMaxFinder::FindBigMaxima : Array out of range "<<fNPeaks<<endl;
		  return;
		}
	      
	      Double_t maxx = hist->GetBinCenterX(xbin);
	      Double_t maxy = hist->GetBinCenterY(ybin);
	      fXPeaks[fNPeaks] = maxx;
	      fYPeaks[fNPeaks] = maxy;
	      fNPeaks++;
	    }
	}
    }
}

void AliHLTTPCHoughMaxFinder::FindMaxima(Int_t threshold)
{
  //Locate all the maxima in input histogram.
  //Maxima is defined as bins with more entries than the
  //immediately neighbouring bins. 
  
  if(fCurrentHisto->GetNEntries() == 0)
    return;
  
  Int_t xmin = fCurrentHisto->GetFirstXbin();
  Int_t xmax = fCurrentHisto->GetLastXbin();
  Int_t ymin = fCurrentHisto->GetFirstYbin();
  Int_t ymax = fCurrentHisto->GetLastYbin();
  Int_t bin[9];
  Double_t value[9];
  
  //Float_t max_kappa = 0.001;
  //Float_t max_phi0 = 0.08;

  for(Int_t xbin=xmin+1; xbin<=xmax-1; xbin++)
    {
      for(Int_t ybin=ymin+1; ybin<=ymax-1; ybin++)
	{
	  bin[0] = fCurrentHisto->GetBin(xbin-1,ybin-1);
	  bin[1] = fCurrentHisto->GetBin(xbin,ybin-1);
	  bin[2] = fCurrentHisto->GetBin(xbin+1,ybin-1);
	  bin[3] = fCurrentHisto->GetBin(xbin-1,ybin);
	  bin[4] = fCurrentHisto->GetBin(xbin,ybin);
	  bin[5] = fCurrentHisto->GetBin(xbin+1,ybin);
	  bin[6] = fCurrentHisto->GetBin(xbin-1,ybin+1);
	  bin[7] = fCurrentHisto->GetBin(xbin,ybin+1);
	  bin[8] = fCurrentHisto->GetBin(xbin+1,ybin+1);
	  value[0] = fCurrentHisto->GetBinContent(bin[0]);
	  value[1] = fCurrentHisto->GetBinContent(bin[1]);
	  value[2] = fCurrentHisto->GetBinContent(bin[2]);
	  value[3] = fCurrentHisto->GetBinContent(bin[3]);
	  value[4] = fCurrentHisto->GetBinContent(bin[4]);
	  value[5] = fCurrentHisto->GetBinContent(bin[5]);
	  value[6] = fCurrentHisto->GetBinContent(bin[6]);
	  value[7] = fCurrentHisto->GetBinContent(bin[7]);
	  value[8] = fCurrentHisto->GetBinContent(bin[8]);
	  
	  
	  if(value[4]>value[0] && value[4]>value[1] && value[4]>value[2]
	     && value[4]>value[3] && value[4]>value[5] && value[4]>value[6]
	     && value[4]>value[7] && value[4]>value[8])
	    {
	      //Found a local maxima
	      Float_t maxx = fCurrentHisto->GetBinCenterX(xbin);
	      Float_t maxy = fCurrentHisto->GetBinCenterY(ybin);
	      
	      if((Int_t)value[4] <= threshold) continue;//central bin below threshold
	      if(fNPeaks >= fNMax)
		{
		  cout<<"AliHLTTPCHoughMaxFinder::FindMaxima : Array out of range "<<fNPeaks<<endl;
		  return;
		}
	      
	      //Check the gradient:
	      if(value[3]/value[4] > fGradX && value[5]/value[4] > fGradX)
		continue;

	      if(value[1]/value[4] > fGradY && value[7]/value[4] > fGradY)
		continue;

	      fXPeaks[fNPeaks] = maxx;
	      fYPeaks[fNPeaks] = maxy;
	      fWeight[fNPeaks] = (Int_t)value[4];
	      fNPeaks++;

	      /*
	      //Check if the peak is overlapping with a previous:
	      Bool_t bigger = kFALSE;
	      for(Int_t p=0; p<fNPeaks; p++)
	        {
		  if(fabs(maxx - fXPeaks[p]) < max_kappa && fabs(maxy - fYPeaks[p]) < max_phi0)
		    {
		      bigger = kTRUE;
		      if(value[4] > fWeight[p]) //this peak is bigger.
			{
		 	  fXPeaks[p] = maxx;
			  fYPeaks[p] = maxy;
			  fWeight[p] = (Int_t)value[4];
			}
		      else
			continue; //previous peak is bigger.
		    }
		}
	      if(!bigger) //there were no overlapping peaks.
		{
		  fXPeaks[fNPeaks] = maxx;
		  fYPeaks[fNPeaks] = maxy;
		  fWeight[fNPeaks] = (Int_t)value[4];
		  fNPeaks++;
		}
	      */
	    }
	}
    }
  
}

struct AliHLTTPCWindow 
{
  Int_t fStart; // Start
  Int_t fSum; // Sum
};

void AliHLTTPCHoughMaxFinder::FindAdaptedPeaks(Int_t kappawindow,Float_t cutratio)
{
  //Peak finder which looks for peaks with a certain shape.
  //The first step involves a pre-peak finder, which looks for peaks
  //in windows (size controlled by kappawindow) summing over each psi-bin.
  //These pre-preaks are then matched between neighbouring kappa-bins to
  //look for real 2D peaks exhbiting the typical cross-shape in the Hough circle transform.
  //The maximum bin within this region is marked as the peak itself, and
  //a few checks is performed to avoid the clear fake peaks (asymmetry check etc.)
  
  
  AliHLTTPCHistogram *hist = fCurrentHisto;
  
  if(!hist)
    {
      cerr<<"AliHLTTPCHoughMaxFinder : No histogram!"<<endl;
      return;
    }
  
  if(hist->GetNEntries() == 0)
    return;

  Int_t xmin = hist->GetFirstXbin();
  Int_t xmax = hist->GetLastXbin();
  Int_t ymin = hist->GetFirstYbin();
  Int_t ymax = hist->GetLastYbin();


  //Start by looking for pre-peaks:
  
  AliHLTTPCWindow **localmaxima = new AliHLTTPCWindow*[hist->GetNbinsY()];
  
  Short_t *nmaxs = new Short_t[hist->GetNbinsY()];
  Int_t n,lastsum,sum;
  Bool_t sumwasrising;
  for(Int_t ybin=ymin; ybin<=ymax; ybin++)
    {
      localmaxima[ybin-ymin] = new AliHLTTPCWindow[hist->GetNbinsX()];
      nmaxs[ybin-ymin] = 0;
      sumwasrising=0;
      lastsum=0;
      n=0;
      for(Int_t xbin=xmin; xbin<=xmax-kappawindow; xbin++)
	{
	  sum=0;
	  for(Int_t lbin=xbin; lbin<xbin+kappawindow; lbin++)
	    sum += hist->GetBinContent(hist->GetBin(lbin,ybin));
	  
	  if(sum < lastsum)
	    {
	      if(sum > fThreshold)
		if(sumwasrising)//Previous sum was a local maxima
		  {
		    localmaxima[ybin-ymin][nmaxs[ybin-ymin]].fStart = xbin-1;
		    localmaxima[ybin-ymin][nmaxs[ybin-ymin]].fSum = lastsum;
		    nmaxs[ybin-ymin]++;
		  }
	      
	      sumwasrising=0;
	    }
	  else if(sum > 0) 
	    sumwasrising=1;
	  lastsum=sum;
	}
    }
  
  Int_t match=0;
  Int_t *starts = new Int_t[hist->GetNbinsY()+1];
  Int_t *maxs = new Int_t[hist->GetNbinsY()+1];
  
  for(Int_t ybin=ymax; ybin >= ymin+1; ybin--)
    {
      for(Int_t i=0; i<nmaxs[ybin-ymin]; i++)
	{
	  Int_t lw = localmaxima[ybin-ymin][i].fSum;

	  if(lw<0)
	    continue; //already used

	  Int_t maxvalue=0,maxybin=0,maxxbin=0,maxwindow=0;
	  for(Int_t k=localmaxima[ybin-ymin][i].fStart; k<localmaxima[ybin-ymin][i].fStart + kappawindow; k++)
	    if(hist->GetBinContent(hist->GetBin(k,ybin)) > maxvalue)
	      {
		maxvalue = hist->GetBinContent(hist->GetBin(k,ybin));
		maxybin = ybin;
		maxxbin = k;
	      }
	  
	  //start expanding in the psi-direction:

	  Int_t lb = localmaxima[ybin-ymin][i].fStart;
	  //Int_t ystart=ybin;
	  starts[ybin] = localmaxima[ybin-ymin][i].fStart;
	  maxs[ybin] = maxxbin;
	  Int_t yl=ybin-1,nybins=1;
	  
	  //cout<<"Starting search at ybin "<<ybin<<" start "<<lb<<" with sum "<<localmaxima[ybin-ymin][i].sum<<endl;
	  while(yl >= ymin)
	    {
	      Bool_t found=0;
	      for(Int_t j=0; j<nmaxs[yl-ymin]; j++)
		{
		  if( localmaxima[yl-ymin][j].fStart - lb < 0) continue;
		  if( localmaxima[yl-ymin][j].fStart < lb + kappawindow + match &&
		      localmaxima[yl-ymin][j].fStart >= lb && localmaxima[yl-ymin][j].fSum > 0)
		    {
		      
		      //cout<<"match at ybin "<<yl<<" yvalue "<<hist->GetBinCenterY(yl)<<" start "<<localmaxima[yl-ymin][j].start<<" sum "<<localmaxima[yl-ymin][j].sum<<endl;
		      
		      Int_t lmaxvalue=0,lmaxxbin=0;
		      for(Int_t k=localmaxima[yl-ymin][j].fStart; k<localmaxima[yl-ymin][j].fStart + kappawindow; k++)
			{
			  if(hist->GetBinContent(hist->GetBin(k,yl)) > maxvalue)
			    {
			      maxvalue = hist->GetBinContent(hist->GetBin(k,yl));
			      maxxbin = k;
			      maxybin = yl;
			      maxwindow = j;
			    }
			  if(hist->GetBinContent(hist->GetBin(k,yl)) > lmaxvalue)//local maxima value
			    {
			      lmaxvalue=hist->GetBinContent(hist->GetBin(k,yl));
			      lmaxxbin=k;
			    }
			}
		      nybins++;
		      starts[yl] = localmaxima[yl-ymin][j].fStart;
		      maxs[yl] = lmaxxbin;
		      localmaxima[yl-ymin][j].fSum=-1; //Mark as used
		      found=1;
		      lb = localmaxima[yl-ymin][j].fStart;
		      break;//Since we found a match in this bin, we dont have to search it anymore, goto next bin.
		    }
		}
	      if(!found || yl == ymin)//no more local maximas to be matched, so write the final peak and break the expansion:
		{
		  if(nybins > 4)
		    {
		      //cout<<"ystart "<<ystart<<" and nybins "<<nybins<<endl;

		      Bool_t truepeak=kTRUE;
		      
		      //cout<<"Maxima found at xbin "<<maxxbin<<" ybin "<<maxybin<<" value "<<maxvalue<<endl;
		      //cout<<"Starting to sum at xbin "<<starts[maxybin-ymin]<<endl;
		      
		      
		      //Look in a window on both sides to probe the asymmetry
		      Float_t right=0,left=0;
		      for(Int_t w=maxxbin+1; w<=maxxbin+3; w++)
			{
			  for(Int_t r=maxybin+1; r<=maxybin+3; r++)
			    {
			      right += (Float_t)hist->GetBinContent(hist->GetBin(w,r));
			    }
			}
		      
		      for(Int_t w=maxxbin-1; w>=maxxbin-3; w--)
			{
			  for(Int_t r=maxybin+1; r<=maxybin+3; r++)
			    {
			      left += (Float_t)hist->GetBinContent(hist->GetBin(w,r));
			    }
			}
		      
		      //cout<<"ratio "<<right/left<<endl;
		      
		      Float_t upperratio=1,lowerratio=1;
		      if(left)
			upperratio = right/left;
		      
		      right=left=0;
		      for(Int_t w=maxxbin+1; w<=maxxbin+3; w++)
			{
			  for(Int_t r=maxybin-1; r>=maxybin-3; r--)
			    {
			      right += (Float_t)hist->GetBinContent(hist->GetBin(w,r));
			    }
			}
		      
		      for(Int_t w=maxxbin-1; w>=maxxbin-3; w--)
			{
			  for(Int_t r=maxybin-1; r>=maxybin-3; r--)
			    {
			      left += (Float_t)hist->GetBinContent(hist->GetBin(w,r));
			    }
			}
		      
		      //cout<<"ratio "<<left/right<<endl;
		      
		      if(right)
			lowerratio = left/right;
		      
		      if(upperratio > cutratio || lowerratio > cutratio)
 			truepeak=kFALSE;
		      
		      if(truepeak)
			{
			  
			  fXPeaks[fNPeaks] = hist->GetBinCenterX(maxxbin);
			  fYPeaks[fNPeaks] = hist->GetBinCenterY(maxybin);
			  fWeight[fNPeaks] = maxvalue;
			  fNPeaks++;
			  
			  /*
			  //Calculate the peak using weigthed means:
			  Float_t sum=0;
			  fYPeaks[fNPeaks]=0;
			  for(Int_t k=maxybin-1; k<=maxybin+1; k++)
			    {
			      Float_t lsum = 0;
			      for(Int_t l=starts[k]; l<starts[k]+kappawindow; l++)
				{
				  lsum += (Float_t)hist->GetBinContent(hist->GetBin(l,k));
				  sum += (Float_t)hist->GetBinContent(hist->GetBin(l,k));
				}
			      fYPeaks[fNPeaks] += lsum*hist->GetBinCenterY(k);
			    }
			  fYPeaks[fNPeaks] /= sum;
			  Int_t ybin1,ybin2;
			  if(fYPeaks[fNPeaks] < hist->GetBinCenterY(hist->FindYbin(fYPeaks[fNPeaks])))
			    {
			      ybin1 = hist->FindYbin(fYPeaks[fNPeaks])-1;
			      ybin2 = ybin1+1;
			    }
			  else
			    {
			      ybin1 = hist->FindYbin(fYPeaks[fNPeaks]);
			      ybin2 = ybin1+1;
			    }

			  Float_t kappa1=0,kappa2=0;
			  sum=0;
			  for(Int_t k=starts[ybin1]; k<starts[ybin1] + kappawindow; k++)
			    {
			      kappa1 += hist->GetBinCenterX(k)*hist->GetBinContent(hist->GetBin(k,ybin1));
			      sum += (Float_t)hist->GetBinContent(hist->GetBin(k,ybin1));
			    }
			  kappa1 /= sum;
			  sum=0;
			  for(Int_t k=starts[ybin2]; k<starts[ybin2] + kappawindow; k++)
			    {
			      kappa2 += hist->GetBinCenterX(k)*hist->GetBinContent(hist->GetBin(k,ybin2));
			      sum += (Float_t)hist->GetBinContent(hist->GetBin(k,ybin2));
			    }
			  kappa2 /= sum;
			  
			  fXPeaks[fNPeaks] = ( kappa1*( hist->GetBinCenterY(ybin2) - fYPeaks[fNPeaks] ) + 
					       kappa2*( fYPeaks[fNPeaks] - hist->GetBinCenterY(ybin1) ) )  / 
 			    (hist->GetBinCenterY(ybin2) - hist->GetBinCenterY(ybin1));

			  fNPeaks++;
			  */
			}
		    }
		  break;
		}
	      else
		yl--;//Search continues...
	    }
	}
    }

  for(Int_t i=0; i<hist->GetNbinsY(); i++)
    delete localmaxima[i];

  delete [] localmaxima;
  delete [] nmaxs;
  delete [] starts;
  delete [] maxs;
}

struct AliHLTTPCPreYPeak 
{
  Int_t fStartPosition; // Start position in X
  Int_t fEndPosition; // End position in X
  Int_t fMinValue; // Minimum value inside the prepeak
  Int_t fMaxValue; // Maximum value inside the prepeak
  Int_t fPrevValue; // Neighbour values
  Int_t fLeftValue; // Neighbour values
  Int_t fRightValue; // Neighbour values
};

void AliHLTTPCHoughMaxFinder::FindAdaptedRowPeaks(Int_t kappawindow,Int_t xsize,Int_t ysize)
{
  // Peak finder which is working over the Hough Space provided by the AliHLTTPCHoughTransformerRow class
  AliHLTTPCHistogram *hist = fCurrentHisto;
  
  if(!hist)
    {
      cerr<<"AliHLTTPCHoughMaxFinder : No histogram!"<<endl;
      return;
    }
  
  if(hist->GetNEntries() == 0)
    return;
  
  Int_t xmin = hist->GetFirstXbin();
  Int_t xmax = hist->GetLastXbin();
  Int_t ymin = hist->GetFirstYbin();
  Int_t ymax = hist->GetLastYbin();
  Int_t nxbins = hist->GetNbinsX()+2;
  Int_t *content = hist->GetContentArray();

  //Start by looking for pre-peaks:
  
  AliHLTTPCPreYPeak **localmaxima = new AliHLTTPCPreYPeak*[hist->GetNbinsY()];
  
  Short_t *nmaxs = new Short_t[hist->GetNbinsY()];
  memset(nmaxs,0,hist->GetNbinsY()*sizeof(Short_t));
  Int_t lastvalue=0,value=0;
  for(Int_t ybin=fNextRow[ymin]; ybin<=ymax; ybin = fNextRow[ybin+1])
    {
      localmaxima[ybin-ymin] = new AliHLTTPCPreYPeak[nxbins-2];
      lastvalue = 0;
      Bool_t found = 0;
      for(Int_t xbin=xmin; xbin<=xmax; xbin++)
	{
	  value = content[xbin + nxbins*ybin]; //value = hist->GetBinContent(xbin + nxbins*ybin); //value = hist->GetBinContent(hist->GetBin(xbin,ybin));
	  if(value > 0)
	    {
	      if((value - lastvalue) > 1)
		{
		  localmaxima[ybin-ymin][nmaxs[ybin-ymin]].fStartPosition = xbin;
		  localmaxima[ybin-ymin][nmaxs[ybin-ymin]].fEndPosition = xbin;
		  localmaxima[ybin-ymin][nmaxs[ybin-ymin]].fMinValue = value;
		  localmaxima[ybin-ymin][nmaxs[ybin-ymin]].fMaxValue = value;
		  localmaxima[ybin-ymin][nmaxs[ybin-ymin]].fPrevValue = 0;
		  localmaxima[ybin-ymin][nmaxs[ybin-ymin]].fLeftValue = lastvalue;
		  found = 1;
		}
	      if(abs(value - lastvalue) <= 1)
		{
		  if(found) {
		    localmaxima[ybin-ymin][nmaxs[ybin-ymin]].fEndPosition = xbin;
		    if(value>localmaxima[ybin-ymin][nmaxs[ybin-ymin]].fMaxValue)
		      localmaxima[ybin-ymin][nmaxs[ybin-ymin]].fMaxValue = value;
		    if(value<localmaxima[ybin-ymin][nmaxs[ybin-ymin]].fMinValue)
		      localmaxima[ybin-ymin][nmaxs[ybin-ymin]].fMinValue = value;
		  }
		}
	      if((value - lastvalue) < -1)
		{
		  if(found) {
		    localmaxima[ybin-ymin][nmaxs[ybin-ymin]].fRightValue = value;
		    nmaxs[ybin-ymin]++;
		    found = 0;
		  }
		}
	    }
	  else
	    {
	      if(found) {
		localmaxima[ybin-ymin][nmaxs[ybin-ymin]].fRightValue = value;
		nmaxs[ybin-ymin]++;
		found = 0;
	      }
	    }
	  lastvalue = value;
	      
	}
      if(found) {
	localmaxima[ybin-ymin][nmaxs[ybin-ymin]].fRightValue = 0;
	nmaxs[ybin-ymin]++;
      }
    }
  
  AliHLTTPCPre2DPeak maxima[500];
  Int_t nmaxima = 0;

  for(Int_t ybin=ymax; ybin >= ymin; ybin--)
    {
      for(Int_t i=0; i<nmaxs[ybin-ymin]; i++)
	{
	  Int_t localminvalue = localmaxima[ybin-ymin][i].fMinValue;
	  Int_t localmaxvalue = localmaxima[ybin-ymin][i].fMaxValue;
	  Int_t localprevvalue = localmaxima[ybin-ymin][i].fPrevValue;
	  Int_t localnextvalue = 0;
	  Int_t localleftvalue = localmaxima[ybin-ymin][i].fLeftValue;
	  Int_t localrightvalue = localmaxima[ybin-ymin][i].fRightValue;

	  if(localminvalue<0)
	    continue; //already used

	  //start expanding in the psi-direction:

	  Int_t localxstart = localmaxima[ybin-ymin][i].fStartPosition;
	  Int_t localxend = localmaxima[ybin-ymin][i].fEndPosition;
	  Int_t tempxstart = localmaxima[ybin-ymin][i].fStartPosition;
	  Int_t tempxend = localmaxima[ybin-ymin][i].fEndPosition;

	  Int_t localy=ybin-1,nybins=1;

	  while(localy >= ymin)
	    {
	      Bool_t found=0;
	      for(Int_t j=0; j<nmaxs[localy-ymin]; j++)
		{
		  if( (localmaxima[localy-ymin][j].fStartPosition <= (tempxend + kappawindow)) && (localmaxima[localy-ymin][j].fEndPosition >= (tempxstart - kappawindow))) 
		    {
		      if((localmaxima[localy-ymin][j].fMinValue <= localmaxvalue) && (localmaxima[localy-ymin][j].fMaxValue >= localminvalue))
			{
			  if(localmaxima[localy-ymin][j].fEndPosition > localxend)
			    localxend = localmaxima[localy-ymin][j].fEndPosition;
			  if(localmaxima[localy-ymin][j].fStartPosition < localxstart)
			    localxstart = localmaxima[localy-ymin][j].fStartPosition;
			  tempxstart = localmaxima[localy-ymin][j].fStartPosition;
			  tempxend = localmaxima[localy-ymin][j].fEndPosition;
			  if(localmaxima[localy-ymin][j].fMinValue < localminvalue)
			    localminvalue = localmaxima[localy-ymin][j].fMinValue;
			  if(localmaxima[localy-ymin][j].fMaxValue > localmaxvalue)
			    localmaxvalue = localmaxima[localy-ymin][j].fMaxValue;
			  if(localmaxima[localy-ymin][j].fRightValue > localrightvalue)
			    localrightvalue = localmaxima[localy-ymin][j].fRightValue;
			  if(localmaxima[localy-ymin][j].fLeftValue > localleftvalue)
			    localleftvalue = localmaxima[localy-ymin][j].fLeftValue;
			  localmaxima[localy-ymin][j].fMinValue = -1;
			  found = 1;
			  nybins++;
			  break;
			}
		      else
			{
			  if(localmaxvalue > localmaxima[localy-ymin][j].fPrevValue)
			    localmaxima[localy-ymin][j].fPrevValue = localmaxvalue;
			  if(localmaxima[localy-ymin][j].fMaxValue > localnextvalue)
			    localnextvalue = localmaxima[localy-ymin][j].fMaxValue;
			}
		    }
		}
	      if(!found || localy == ymin)//no more local maximas to be matched, so write the final peak and break the expansion:
		{
		  if((nybins > ysize) && ((localxend-localxstart+1) > xsize) && (localmaxvalue > localprevvalue) && (localmaxvalue > localnextvalue) && (localmaxvalue > localleftvalue) && (localmaxvalue > localrightvalue))
		  //		  if((nybins > ysize) && ((localxend-localxstart+1) > xsize))
		    {
		      maxima[nmaxima].fX = ((Float_t)localxstart+(Float_t)localxend)/2.0;
		      maxima[nmaxima].fY = ((Float_t)ybin+(Float_t)(localy+1))/2.0;
		      maxima[nmaxima].fSizeX = localxend-localxstart+1;
		      maxima[nmaxima].fSizeY = nybins;
		      maxima[nmaxima].fWeight = (localminvalue+localmaxvalue)/2;
		      maxima[nmaxima].fStartX = localxstart;
		      maxima[nmaxima].fEndX = localxend;
		      maxima[nmaxima].fStartY = localy +1;
		      maxima[nmaxima].fEndY = ybin;
#ifdef do_mc
		      //		      cout<<"Peak found at: "<<((Float_t)localxstart+(Float_t)localxend)/2.0<<" "<<((Float_t)ybin+(Float_t)(localy+1))/2.0<<" "<<localminvalue<<" "<<localmaxvalue<<" "<<" with weight "<<(localminvalue+localmaxvalue)/2<<" and size "<<localxend-localxstart+1<<" by "<<nybins<<endl;
#endif
		      nmaxima++;
		    }
		  break;
		}
	      else
		localy--;//Search continues...
	    }
	}
    }

  //remove fake tracks
  
  for(Int_t i = 0; i < (nmaxima - 1); i++)
    {
      //      if(maxima[i].fWeight < 0) continue;
      for(Int_t j = i + 1; j < nmaxima; j++)
	{
	  //	  if(maxima[j].fWeight < 0) continue;
	  MergeRowPeaks(&maxima[i],&maxima[j],5.0);
	}
    }

  //merge tracks in neighbour eta slices
  Int_t currentnpeaks = fNPeaks;
  for(Int_t i = 0; i < nmaxima; i++) {
    if(maxima[i].fWeight < 0) continue;
    Bool_t merged = kFALSE;
    for(Int_t j = fN1PeaksPrevEtaSlice; j < fN2PeaksPrevEtaSlice; j++) {
      //      if(fWeight[j] < 0) continue;
      // Merge only peaks with limited size in eta
      if((fENDETAPeaks[j]-fSTARTETAPeaks[j]) >= 2) continue;
      if((maxima[i].fStartX <= fENDXPeaks[j]+1) && (maxima[i].fEndX >= fSTARTXPeaks[j]-1) && (maxima[i].fStartY <= fENDYPeaks[j]+1) && (maxima[i].fEndY >= fSTARTYPeaks[j]-1)){
	//merge
	merged = kTRUE;
	if(fWeight[j] > 0) {
	  fXPeaks[fNPeaks] = (hist->GetPreciseBinCenterX(maxima[i].fX)+(fENDETAPeaks[j]-fSTARTETAPeaks[j]+1)*fXPeaks[j])/(fENDETAPeaks[j]-fSTARTETAPeaks[j]+2);
	  fYPeaks[fNPeaks] = (hist->GetPreciseBinCenterY(maxima[i].fY)+(fENDETAPeaks[j]-fSTARTETAPeaks[j]+1)*fYPeaks[j])/(fENDETAPeaks[j]-fSTARTETAPeaks[j]+2);
	  fSTARTXPeaks[fNPeaks] = maxima[i].fStartX;
	  fSTARTYPeaks[fNPeaks] = maxima[i].fStartY;
	  fENDXPeaks[fNPeaks] = maxima[i].fEndX;
	  fENDYPeaks[fNPeaks] = maxima[i].fEndY;

	  fWeight[fNPeaks] = abs((Int_t)maxima[i].fWeight) + abs(fWeight[j]);
	  fSTARTETAPeaks[fNPeaks] = fSTARTETAPeaks[j];
	  fENDETAPeaks[fNPeaks] = fCurrentEtaSlice;
	  fNPeaks++;
	}
	fWeight[j] = -abs(fWeight[j]);
      }
    }
    fXPeaks[fNPeaks] = hist->GetPreciseBinCenterX(maxima[i].fX);
    fYPeaks[fNPeaks] = hist->GetPreciseBinCenterY(maxima[i].fY);
    if(!merged)
      fWeight[fNPeaks] = abs((Int_t)maxima[i].fWeight);
    else
      fWeight[fNPeaks] = -abs((Int_t)maxima[i].fWeight);
    fSTARTXPeaks[fNPeaks] = maxima[i].fStartX;
    fSTARTYPeaks[fNPeaks] = maxima[i].fStartY;
    fENDXPeaks[fNPeaks] = maxima[i].fEndX;
    fENDYPeaks[fNPeaks] = maxima[i].fEndY;
    fSTARTETAPeaks[fNPeaks] = fCurrentEtaSlice;
    fENDETAPeaks[fNPeaks] = fCurrentEtaSlice;
    fNPeaks++;
  }

  fN1PeaksPrevEtaSlice = currentnpeaks;    
  fN2PeaksPrevEtaSlice = fNPeaks;

  for(Int_t ybin=fNextRow[ymin]; ybin<=ymax; ybin = fNextRow[ybin+1])
    delete [] localmaxima[ybin-ymin];

  delete [] localmaxima;
  delete [] nmaxs;
}

void AliHLTTPCHoughMaxFinder::FindPeak1(Int_t ywindow,Int_t xbinsides)
{
  //Testing mutliple peakfinding.
  //The algorithm searches the histogram for prepreaks by looking in windows
  //for each bin on the xaxis. The size of these windows is controlled by ywindow.
  //Then the prepreaks are sorted according to their weight (sum inside window),
  //and the peak positions are calculated by taking the weighted mean in both
  //x and y direction. The size of the peak in x-direction is controlled by xbinsides.

  if(!fCurrentHisto)
    {
      printf("AliHLTTPCHoughMaxFinder::FindPeak1 : No input histogram\n");
      return;
    }  
  if(fCurrentHisto->GetNEntries()==0)
    return;
  
  //Int_t ywindow=2;
  //Int_t xbinsides=1;
  
  //Float_t max_kappa = 0.001;
  //Float_t max_phi0 = 0.08;
  
  Int_t maxsum=0;
  
  Int_t xmin = fCurrentHisto->GetFirstXbin();
  Int_t xmax = fCurrentHisto->GetLastXbin();
  Int_t ymin = fCurrentHisto->GetFirstYbin();
  Int_t ymax = fCurrentHisto->GetLastYbin();
  Int_t nbinsx = fCurrentHisto->GetNbinsX()+1;
  
  AliHLTTPCAxisWindow **windowPt = new AliHLTTPCAxisWindow*[nbinsx];
  AliHLTTPCAxisWindow **anotherPt = new AliHLTTPCAxisWindow*[nbinsx];
  
  for(Int_t i=0; i<nbinsx; i++)
    {
      windowPt[i] = new AliHLTTPCAxisWindow;
#if defined(__DECCXX)
      bzero((char *)windowPt[i],sizeof(AliHLTTPCAxisWindow));
#else
      bzero((void*)windowPt[i],sizeof(AliHLTTPCAxisWindow));
#endif
      anotherPt[i] = windowPt[i];
    }
  
  for(Int_t xbin=xmin; xbin<=xmax; xbin++)
    {
      maxsum = 0;
      for(Int_t ybin=ymin; ybin<=ymax-ywindow; ybin++)
	{
	  Int_t suminwindow=0;
	  for(Int_t b=ybin; b<ybin+ywindow; b++)
	    {
	      //inside window
	      Int_t bin = fCurrentHisto->GetBin(xbin,b);
	      suminwindow += (Int_t)fCurrentHisto->GetBinContent(bin);
	    }
	  
	  if(suminwindow > maxsum)
	    {
	      maxsum = suminwindow;
	      windowPt[xbin]->fYmin = ybin;
	      windowPt[xbin]->fYmax = ybin + ywindow;
	      windowPt[xbin]->fWeight = suminwindow;
	      windowPt[xbin]->fXbin = xbin;
	    }
	}
    }

  //Sort the windows according to the weight
  SortPeaks(windowPt,0,nbinsx);
  
  Float_t top,butt;
  for(Int_t i=0; i<nbinsx; i++)
    {
      top=butt=0;
      Int_t xbin = windowPt[i]->fXbin;
      
      if(xbin<xmin || xbin > xmax-1) continue;
      
      //Check if this is really a local maxima
      if(anotherPt[xbin-1]->fWeight > anotherPt[xbin]->fWeight ||
	 anotherPt[xbin+1]->fWeight > anotherPt[xbin]->fWeight)
	continue;

      for(Int_t j=windowPt[i]->fYmin; j<windowPt[i]->fYmax; j++)
	{
	  //Calculate the mean in y direction:
	  Int_t bin = fCurrentHisto->GetBin(windowPt[i]->fXbin,j);
	  top += (fCurrentHisto->GetBinCenterY(j))*(fCurrentHisto->GetBinContent(bin));
	  butt += fCurrentHisto->GetBinContent(bin);
	}
      
      if(butt < fThreshold)
      	continue;
      
      fXPeaks[fNPeaks] = fCurrentHisto->GetBinCenterX(windowPt[i]->fXbin);
      fYPeaks[fNPeaks] = top/butt;
      fWeight[fNPeaks] = (Int_t)butt;
      //cout<<"mean in y "<<ypeaks[n]<<" on x "<<windowPt[i]->xbin<<" content "<<butt<<endl;
      fNPeaks++;
      if(fNPeaks==fNMax) 
	{
	  cerr<<"AliHLTTPCHoughMaxFinder::FindPeak1 : Peak array out of range!!!"<<endl;
	  break;
	}
    }

  
  //Improve the peaks by including the region around in x.
  Float_t ytop,ybutt;
  Int_t prev;
  Int_t w;
  for(Int_t i=0; i<fNPeaks; i++)
    {
      Int_t xbin = fCurrentHisto->FindXbin(fXPeaks[i]);
      if(xbin - xbinsides < xmin || xbin + xbinsides > xmax) continue;
      top=butt=0;
      ytop=0,ybutt=0;	  
      w=0;
      prev = xbin - xbinsides+1;
      for(Int_t j=xbin-xbinsides; j<=xbin+xbinsides; j++)
	{
	  /*
	  //Check if the windows are overlapping:
	  if(anotherPt[j]->ymin > anotherPt[prev]->ymax) {prev=j; continue;}
	  if(anotherPt[j]->ymax < anotherPt[prev]->ymin) {prev=j; continue;}
	  prev = j;
	  */
	  
	  top += fCurrentHisto->GetBinCenterX(j)*anotherPt[j]->fWeight;
	  butt += anotherPt[j]->fWeight;
	  
	  for(Int_t k=anotherPt[j]->fYmin; k<anotherPt[j]->fYmax; k++)
	    {
	      Int_t bin = fCurrentHisto->GetBin(j,k);
	      ytop += (fCurrentHisto->GetBinCenterY(k))*(fCurrentHisto->GetBinContent(bin));
	      ybutt += fCurrentHisto->GetBinContent(bin);
	      w+=(Int_t)fCurrentHisto->GetBinContent(bin);
	    }
	}
      
      fXPeaks[i] = top/butt;
      fYPeaks[i] = ytop/ybutt;
      fWeight[i] = w;
      //cout<<"Setting weight "<<w<<" kappa "<<fXPeaks[i]<<" phi0 "<<fYPeaks[i]<<endl;
      
      /*
      //Check if this peak is overlapping with a previous:
      for(Int_t p=0; p<i; p++)
	{
	  //cout<<fabs(fXPeaks[p] - fXPeaks[i])<<" "<<fabs(fYPeaks[p] - fYPeaks[i])<<endl;
	  if(fabs(fXPeaks[p] - fXPeaks[i]) < max_kappa &&
	     fabs(fYPeaks[p] - fYPeaks[i]) < max_phi0)
	    {
	      fWeight[i]=0;
	      //break;
	    }
	}
      */
    }
  
  for(Int_t i=0; i<nbinsx; i++)
    delete windowPt[i];
  delete [] windowPt;
  delete [] anotherPt;
}

void AliHLTTPCHoughMaxFinder::SortPeaks(struct AliHLTTPCAxisWindow **a,Int_t first,Int_t last)
{
  //General sorting routine
  //Sort according to PeakCompare()

  static struct AliHLTTPCAxisWindow *tmp;
  static int i;           // "static" to save stack space
  int j;
  
  while (last - first > 1) {
    i = first;
    j = last;
    for (;;) {
      while (++i < last && PeakCompare(a[i], a[first]) < 0)
	;
      while (--j > first && PeakCompare(a[j], a[first]) > 0)
	;
      if (i >= j)
	break;
      
      tmp  = a[i];
      a[i] = a[j];
      a[j] = tmp;
    }
    if (j == first) {
      ++first;
      continue;
    }
    tmp = a[first];
    a[first] = a[j];
    a[j] = tmp;
    if (j - first < last - (j + 1)) {
      SortPeaks(a, first, j);
      first = j + 1;   // QSort(j + 1, last);
    } else {
      SortPeaks(a, j + 1, last);
      last = j;        // QSort(first, j);
    }
  }
  
}

Int_t AliHLTTPCHoughMaxFinder::PeakCompare(struct AliHLTTPCAxisWindow *a,struct AliHLTTPCAxisWindow *b) const
{
  // Peak comparison based on peaks weight
  if(a->fWeight < b->fWeight) return 1;
  if(a->fWeight > b->fWeight) return -1;
  return 0;
}

void AliHLTTPCHoughMaxFinder::FindPeak(Int_t t1,Double_t t2,Int_t t3)
{
  //Attempt of a more sophisticated peak finder.
  //Finds the best peak in the histogram, and returns the corresponding
  //track object.

  if(!fCurrentHisto)
    {
      printf("AliHLTTPCHoughMaxFinder::FindPeak : No histogram!!\n");
      return;
    }
  AliHLTTPCHistogram *hist = fCurrentHisto;
  if(hist->GetNEntries()==0)
    return;

  Int_t xmin = hist->GetFirstXbin();
  Int_t xmax = hist->GetLastXbin();
  Int_t ymin = hist->GetFirstYbin();
  Int_t ymax = hist->GetLastYbin();
  Int_t nbinsx = hist->GetNbinsX()+1;
  
  Int_t *m = new Int_t[nbinsx];
  Int_t *mlow = new Int_t[nbinsx];
  Int_t *mup = new Int_t[nbinsx];
  
  
 recompute:  //this is a goto.
  
  for(Int_t i=0; i<nbinsx; i++)
    {
      m[i]=0;
      mlow[i]=0;
      mup[i]=0;
    }

  Int_t maxx=0,sum=0,maxxbin=0,bin=0;

  for(Int_t xbin=xmin; xbin<=xmax; xbin++)
    {
      for(Int_t ybin=ymin; ybin <= ymax - t1; ybin++)
	{
	  sum = 0;
	  for(Int_t y=ybin; y <= ybin+t1; y++)
	    {
	      if(y>ymax) break;
	      //Inside window
	      bin = hist->GetBin(xbin,y);
	      sum += (Int_t)hist->GetBinContent(bin);
	      
	    }
	  if(sum > m[xbin]) //Max value locally in this xbin
	    {
	      m[xbin]=sum;
	      mlow[xbin]=ybin;
	      mup[xbin]=ybin + t1;
	    }
	  
	}
      
      if(m[xbin] > maxx) //Max value globally in x-direction
	{
	  maxxbin = xbin;
	  maxx = m[xbin];//sum;
	}
    }
  //printf("maxxbin %d maxx %d mlow %d mup %d\n",maxxbin,maxx,mlow[maxxbin],mup[maxxbin]);
  //printf("ylow %f yup %f\n",hist->GetBinCenterY(mlow[maxxbin]),hist->GetBinCenterY(mup[maxxbin]));

  //Determine a width in the x-direction
  Int_t xlow=0,xup=0;
  
  for(Int_t xbin=maxxbin-1; xbin >= xmin; xbin--)
    {
      if(m[xbin] < maxx*t2)
	{
	  xlow = xbin+1;
	  break;
	}
    }
  for(Int_t xbin = maxxbin+1; xbin <=xmax; xbin++)
    {
      if(m[xbin] < maxx*t2)
	{
	  xup = xbin-1;
	  break;
	}
    }
  
  Double_t top=0,butt=0,value,xpeak;
  if(xup - xlow + 1 > t3)
    {
      t1 -= 1;
      printf("\nxrange out if limit xup %d xlow %d t1 %d\n\n",xlow,xup,t1);
      if(t1 > 1)
	goto recompute;
      else
	{
	  xpeak = hist->GetBinCenterX(maxxbin);
	  goto moveon;
	}
    }
  
  //printf("xlow %f xup %f\n",hist->GetBinCenterX(xlow),hist->GetBinCenterX(xup));
  //printf("Spread in x %d\n",xup-xlow +1);

  //Now, calculate the center of mass in x-direction
  for(Int_t xbin=xlow; xbin <= xup; xbin++)
    {
      value = hist->GetBinCenterX(xbin);
      top += value*m[xbin];
      butt += m[xbin];
    }
  xpeak = top/butt;
  
 moveon:
  
  //Find the peak in y direction:
  Int_t xl = hist->FindXbin(xpeak);
  if(hist->GetBinCenterX(xl) > xpeak)
    xl--;

  Int_t xu = xl + 1;
  
  if(hist->GetBinCenterX(xl) > xpeak || hist->GetBinCenterX(xu) <= xpeak)
    printf("\nAliHLTTPCHoughMaxFinder::FindPeak : Wrong xrange %f %f %f\n\n",hist->GetBinCenterX(xl),xpeak,hist->GetBinCenterX(xu));
    
    //printf("\nxlow %f xup %f\n",hist->GetBinCenterX(xl),hist->GetBinCenterX(xu));

  value=top=butt=0;
  
  //printf("ylow %f yup %f\n",hist->GetBinCenterY(mlow[xl]),hist->GetBinCenterY(mup[xl]));
  //printf("ylow %f yup %f\n",hist->GetBinCenterY(mlow[xu]),hist->GetBinCenterY(mup[xu]));
  
  for(Int_t ybin=mlow[xl]; ybin <= mup[xl]; ybin++)
    {
      value = hist->GetBinCenterY(ybin);
      bin = hist->GetBin(xl,ybin);
      top += value*hist->GetBinContent(bin);
      butt += hist->GetBinContent(bin);
    }
  Double_t ypeaklow = top/butt;
  
  //printf("ypeaklow %f\n",ypeaklow);

  value=top=butt=0;
  for(Int_t ybin=mlow[xu]; ybin <= mup[xu]; ybin++)
    {
      value = hist->GetBinCenterY(ybin);
      bin = hist->GetBin(xu,ybin);
      top += value*hist->GetBinContent(bin);
      butt += hist->GetBinContent(bin);
    }
  Double_t ypeakup = top/butt;
  
  //printf("ypeakup %f\n",ypeakup);

  Double_t xvalueup = hist->GetBinCenterX(xu);
  Double_t xvaluelow = hist->GetBinCenterX(xl);

  Double_t ypeak = (ypeaklow*(xvalueup - xpeak) + ypeakup*(xpeak - xvaluelow))/(xvalueup - xvaluelow);


  //Find the weight:
  //bin = hist->FindBin(xpeak,ypeak);
  //Int_t weight = (Int_t)hist->GetBinContent(bin);

  //AliHLTTPCHoughTrack *track = new AliHLTTPCHoughTrack();
  //track->SetTrackParameters(xpeak,ypeak,weight);
  fXPeaks[fNPeaks]=xpeak;
  fYPeaks[fNPeaks]=ypeak;
  fWeight[fNPeaks]=(Int_t)hist->GetBinContent(bin);
  fNPeaks++;
  
  delete [] m;
  delete [] mlow;
  delete [] mup;
  
  //return track;
}

Float_t AliHLTTPCHoughMaxFinder::GetXPeakSize(Int_t i) const
{
  // Get X size of a peak
  if(i<0 || i>fNMax)
    {
      STDCERR<<"AliHLTTPCHoughMaxFinder::GetXPeakSize : Invalid index "<<i<<STDENDL;
      return 0;
    }
  Float_t binwidth = fCurrentHisto->GetBinWidthX();
  return binwidth*(fENDXPeaks[i]-fSTARTXPeaks[i]+1);
}

Float_t AliHLTTPCHoughMaxFinder::GetYPeakSize(Int_t i) const
{
  // Get Y size of a peak
  if(i<0 || i>fNMax)
    {
      STDCERR<<"AliHLTTPCHoughMaxFinder::GetYPeak : Invalid index "<<i<<STDENDL;
      return 0;
    }
  Float_t binwidth = fCurrentHisto->GetBinWidthY();
  return binwidth*(fENDYPeaks[i]-fSTARTYPeaks[i]+1);
}

Bool_t AliHLTTPCHoughMaxFinder::MergeRowPeaks(AliHLTTPCPre2DPeak *maxima1, AliHLTTPCPre2DPeak *maxima2, Float_t distance)
{
  // Check the distance between tracks corresponding to given Hough space peaks and if the
  // distance is smaller than some threshold value marks the smaller peak as fake
  AliHLTTPCHistogram *hist = fCurrentHisto;
  Int_t nxbins = hist->GetNbinsX()+2;

  Int_t xtrack1=0,xtrack2=0,ytrack1=0,ytrack2=0;
  Int_t deltax = 9999;
  for(Int_t ix1 = maxima1->fStartX; ix1 <= maxima1->fEndX; ix1++) {
    for(Int_t ix2 = maxima2->fStartX; ix2 <= maxima2->fEndX; ix2++) {
      if(abs(ix1 - ix2) < deltax) {
	deltax = abs(ix1 - ix2);
	xtrack1 = ix1;
	xtrack2 = ix2;
      }
    }
  }
  Int_t deltay = 9999;
  for(Int_t iy1 = maxima1->fStartY; iy1 <= maxima1->fEndY; iy1++) {
    for(Int_t iy2 = maxima2->fStartY; iy2 <= maxima2->fEndY; iy2++) {
      if(abs(iy1 - iy2) < deltay) {
	deltay = abs(iy1 - iy2);
	ytrack1 = iy1;
	ytrack2 = iy2;
      }
    }
  }
  Int_t firstrow1 = fTrackFirstRow[xtrack1 + nxbins*ytrack1];
  Int_t lastrow1 = fTrackLastRow[xtrack1 + nxbins*ytrack1];
  Int_t firstrow2 = fTrackFirstRow[xtrack1 + nxbins*ytrack1];
  Int_t lastrow2 = fTrackLastRow[xtrack1 + nxbins*ytrack1];
  Int_t firstrow,lastrow;
  if(firstrow1 < firstrow2)
    firstrow = firstrow2;
  else
    firstrow = firstrow1;

  if(lastrow1 > lastrow2)
    lastrow = lastrow2;
  else
    lastrow = lastrow1;
	 
  AliHLTTPCHoughTrack track1;
  Float_t x1 = hist->GetPreciseBinCenterX(xtrack1);
  Float_t y1 = hist->GetPreciseBinCenterY(ytrack1);
  Float_t psi1 = atan((x1-y1)/(AliHLTTPCHoughTransformerRow::GetBeta1()-AliHLTTPCHoughTransformerRow::GetBeta2()));
  Float_t kappa1 = 2.0*(x1*cos(psi1)-AliHLTTPCHoughTransformerRow::GetBeta1()*sin(psi1));
  track1.SetTrackParameters(kappa1,psi1,1);
  Float_t firsthit1[3];
  if(!track1.GetCrossingPoint(firstrow,firsthit1)) return kFALSE;
  Float_t lasthit1[3];
  if(!track1.GetCrossingPoint(lastrow,lasthit1)) return kFALSE;

  AliHLTTPCHoughTrack track2;
  Float_t x2 = hist->GetPreciseBinCenterX(xtrack2);
  Float_t y2 = hist->GetPreciseBinCenterY(ytrack2);
  Float_t psi2 = atan((x2-y2)/(AliHLTTPCHoughTransformerRow::GetBeta1()-AliHLTTPCHoughTransformerRow::GetBeta2()));
  Float_t kappa2 = 2.0*(x2*cos(psi2)-AliHLTTPCHoughTransformerRow::GetBeta1()*sin(psi2));
  track2.SetTrackParameters(kappa2,psi2,1);
  Float_t firsthit2[3];
  if(!track2.GetCrossingPoint(firstrow,firsthit2)) return kFALSE;
  Float_t lasthit2[3];
  if(!track2.GetCrossingPoint(lastrow,lasthit2)) return kFALSE;
	  
  Float_t padpitchlow = AliHLTTPCTransform::GetPadPitchWidth(AliHLTTPCTransform::GetPatch(firstrow));
  Float_t padpitchup = AliHLTTPCTransform::GetPadPitchWidth(AliHLTTPCTransform::GetPatch(lastrow));
  // check the distance between tracks at the edges
  //  cout<<"Check "<<firsthit1[1]<<" "<<firsthit2[1]<<" "<<padpitchlow<<" "<<lasthit1[1]<<" "<<lasthit2[1]<<" "<<padpitchup<<" "<<xtrack1<<" "<<ytrack1<<" "<<xtrack2<<" "<<ytrack2<<endl;
  if((fabs(firsthit1[1]-firsthit2[1])/padpitchlow + fabs(lasthit1[1]-lasthit2[1])/padpitchup) < distance) {
    if(maxima1->fSizeX*maxima1->fSizeY > maxima2->fSizeX*maxima2->fSizeY)
      maxima2->fWeight = -fabs(maxima2->fWeight);
    if(maxima1->fSizeX*maxima1->fSizeY < maxima2->fSizeX*maxima2->fSizeY)
      maxima1->fWeight = -fabs(maxima1->fWeight);
    if(maxima1->fSizeX*maxima1->fSizeY == maxima2->fSizeX*maxima2->fSizeY) {
      if(maxima1->fStartX > maxima2->fStartX)
	maxima1->fStartX = maxima2->fStartX;
      if(maxima1->fStartY > maxima2->fStartY)
	maxima1->fStartY = maxima2->fStartY;
      if(maxima1->fEndX < maxima2->fEndX)
	maxima1->fEndX = maxima2->fEndX;
      if(maxima1->fEndY < maxima2->fEndY)
	maxima1->fEndY = maxima2->fEndY;
      maxima1->fX = ((Float_t)maxima1->fStartX + (Float_t)maxima1->fEndX)/2.0;
      maxima1->fY = ((Float_t)maxima1->fStartY + (Float_t)maxima1->fEndY)/2.0;
      maxima1->fSizeX = (maxima1->fEndX - maxima1->fStartX + 1);
      maxima1->fSizeY = (maxima1->fEndY - maxima1->fStartY + 1);
      maxima2->fWeight = -fabs(maxima2->fWeight);
    }
    return kTRUE;
  }
  return kFALSE;
}
