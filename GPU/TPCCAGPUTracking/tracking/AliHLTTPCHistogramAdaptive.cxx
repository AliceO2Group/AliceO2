// $Id$
// origin: hough/AliL3HistogramAdaptive.cxx,v 1.13 Thu Jun 23 17:46:54 2005 UTC by hristov

// Author: Anders Vestbo <mailto:vestbo@fi.uib.no>
//*-- Copyright &copy ALICE HLT Group

#include "AliHLTStdIncludes.h"
#include "AliHLTTPCLogging.h"
#include "AliHLTTPCHistogramAdaptive.h"
#include "AliHLTTPCTransform.h"
#include "AliHLTTPCTrack.h"

#if __GNUC__ >= 3
using namespace std;
#endif

//_____________________________________________________________
// AliHLTTPCHistogramAdaptive
//
// 2D histogram class adapted for kappa and psi as used in the Circle Hough Transform.
// The bins in kappa is not linear, but has a width which is specified by argument
// ptres in the constructor. This gives the relative pt resolution which should
// be kept throughout the kappa range. 

ClassImp(AliHLTTPCHistogramAdaptive)

AliHLTTPCHistogramAdaptive::AliHLTTPCHistogramAdaptive() : AliHLTTPCHistogram()
{
  //default ctor
  fKappaBins=0;
}

  
AliHLTTPCHistogramAdaptive::AliHLTTPCHistogramAdaptive(Char_t *name,Double_t minpt,Double_t maxpt,Double_t ptres,
					       Int_t nybins,Double_t ymin,Double_t ymax)
{
  //normal ctor
  strcpy(fName,name);
  
  fPtres = ptres;
  fXmin = -1*AliHLTTPCTransform::GetBFact()*AliHLTTPCTransform::GetBField()/minpt;
  fXmax = AliHLTTPCTransform::GetBFact()*AliHLTTPCTransform::GetBField()/minpt;

  fMinPt = minpt;
  fMaxPt = maxpt;
  fNxbins = InitKappaBins();
  fNybins = nybins;
  
  fYmin = ymin;
  fYmax = ymax;
  fFirstXbin=1;
  fFirstYbin=1;
  fLastXbin = fNxbins;
  fLastYbin = fNybins;
  fNcells = (fNxbins+2)*(fNybins+2);

  fThreshold=0;
  fContent = new Int_t[fNcells];
  Reset();
}

AliHLTTPCHistogramAdaptive::~AliHLTTPCHistogramAdaptive()
{
  //dtor
  if(fKappaBins)
    delete [] fKappaBins;
}

Int_t AliHLTTPCHistogramAdaptive::InitKappaBins()
{
  //Here a LUT for the kappa values created. This has to be done since
  //the binwidth in kappa is not constant, but change according to the
  //set relative resolution in pt.
  //Since the kappa values are symmetric about origo, the size of the
  //LUT is half of the total number of bins in kappa direction.
  
  Double_t pt = fMinPt,deltapt,localpt;
  Int_t bin=0;
  
  while(pt < fMaxPt)
    {
      localpt = pt;
      deltapt = fPtres*localpt*localpt;
      pt += 2*deltapt;
      bin++;
    }
  fKappaBins = new Double_t[bin+1];
  pt=fMinPt;
  bin=0;
  fKappaBins[bin] = AliHLTTPCTransform::GetBFact()*AliHLTTPCTransform::GetBField()/fMinPt; 
  while(pt < fMaxPt)
    {
      localpt = pt;
      deltapt = fPtres*localpt*localpt;
      pt += 2*deltapt;                      //*2 because pt +- 1/2*deltapt is one bin
      bin++;
      fKappaBins[bin] = AliHLTTPCTransform::GetBFact()*AliHLTTPCTransform::GetBField()/pt;
    }
  return (bin+1)*2; //Both negative and positive kappa.
}


void AliHLTTPCHistogramAdaptive::Fill(Double_t x,Double_t y,Int_t weight)
{
  //Fill a given bin in the histogram
  Int_t bin = FindBin(x,y);
  if(bin < 0)
    return;
  AddBinContent(bin,weight);

}

Int_t AliHLTTPCHistogramAdaptive::FindBin(Double_t x,Double_t y) const
{
  //Find a bin in the histogram  
  Int_t xbin = FindXbin(x);
  Int_t ybin = FindYbin(y);
  
  if(!xbin || !ybin) 
    return -1;
  return GetBin(xbin,ybin);
}

Int_t AliHLTTPCHistogramAdaptive::FindXbin(Double_t x) const
{
  //Find X bin in the histogram
  if(x < fXmin || x > fXmax || fabs(x) < fKappaBins[(fNxbins/2-1)])
    return 0;
  
  //Remember that kappa value is decreasing with bin number!
  //Also, the bin numbering starts at 1 and ends at fNxbins,
  //so the corresponding elements in the LUT is bin - 1.

  Int_t bin=0;
  while(bin < fNxbins/2)
    {
      if(fabs(x) <= fKappaBins[bin] && fabs(x) > fKappaBins[bin+1])
	break;
      bin++;
    }
  if(x < 0)
    return bin + 1;
  else 
    return fNxbins - bin;
  
}

Int_t AliHLTTPCHistogramAdaptive::FindYbin(Double_t y) const
{
  //Find Y bin in the histogram
  if(y < fYmin || y > fYmax)
    return 0;
  
  return 1 + (Int_t)(fNybins*(y-fYmin)/(fYmax-fYmin));
}

Double_t AliHLTTPCHistogramAdaptive::GetBinCenterX(Int_t xbin) const
{
  //Returns bin center in X
  if(xbin < fFirstXbin || xbin > fLastXbin)
    {
      LOG(AliHLTTPCLog::kWarning,"AliHLTTPCHistogramAdaptive::GetBinCenterX","Bin-value")
	<<"XBinvalue out of range "<<xbin<<ENDLOG;
      return 0;
    }
  
  //The bin numbers go from 1 to fNxbins, so the corresponding
  //element in the LUT is xbin - 1. This is the reason why we 
  //substract a 1 here:
  
  Int_t bin = xbin;
  bin -= 1;
  if(bin >= fNxbins/2)
    bin = fNxbins - 1 - bin;
  
  //Remember again that the kappa-values are _decreasing_ with bin number.
  
  Double_t binwidth = fKappaBins[bin] - fKappaBins[bin+1];
  Double_t kappa = fKappaBins[bin] - 0.5*binwidth;
  if(xbin < fNxbins/2)
    return -1.*kappa;
  else
    return kappa;

}

Double_t AliHLTTPCHistogramAdaptive::GetBinCenterY(Int_t ybin) const
{
  //Returns bin center in Y
  if(ybin < fFirstYbin || ybin > fLastYbin)
    {
      LOG(AliHLTTPCLog::kError,"AliHLTTPCHistogramAdaptive::GetBinCenterY","ybin")
	<<"Bin-value out of range "<<ybin<<ENDLOG;
      return -1;
    }
  Double_t binwidth = (fYmax - fYmin) / fNybins;
  return fYmin + (ybin-0.5) * binwidth;

}


void AliHLTTPCHistogramAdaptive::Draw(Char_t *option)
{
  //Draw the histogram
  if(!fRootHisto)
    CreateRootHisto();
  
  Double_t kappa,psi;
  Int_t content,bin;
  for(Int_t i=fFirstXbin; i<=fLastXbin; i++)
    {
      kappa = GetBinCenterX(i);
      for(Int_t j=fFirstYbin; j<=fLastYbin; j++)
	{
	  psi = GetBinCenterY(j);
	  bin = GetBin(i,j);
	  content = GetBinContent(bin);
	  fRootHisto->Fill(kappa,psi,content);
	}
    }
  fRootHisto->Draw(option);
  return;
}

void AliHLTTPCHistogramAdaptive::Print() const
{
  //Print the contents of the histogram
  cout<<"Printing content of histogram "<<fName<<endl;
  for(Int_t i=0; i<fNcells; i++)
    {
      if(GetBinContent(i)==0) continue;
      cout<<"Bin "<<i<<": "<<GetBinContent(i)<<endl;
    }

}
