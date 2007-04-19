// @(#) $Id$
// origin: hough/AliL3Histogram1D.cxx,v 1.11 Tue Jun 14 10:55:20 2005 UTC by cvetan 

// Author: Anders Vestbo <mailto:vestbo@fi.uib.no>
//*-- Copyright &copy ALICE HLT Group

#include <strings.h>
#include "AliHLTStdIncludes.h"

#include "AliHLTTPCLogging.h"
#include "AliHLTTPCHistogram1D.h"

#ifdef use_root
#include <TH1.h>
#endif

#if __GNUC__ >= 3
using namespace std;
#endif

//_____________________________________________________________
// AliHLTTPCHistogram1D
//
// 1D histogram class.

ClassImp(AliHLTTPCHistogram1D)

AliHLTTPCHistogram1D::AliHLTTPCHistogram1D()
{
  //default ctor
  fNbins = 0;
  fNcells = 0;
  fEntries = 0;
  fXmin = 0;
  fXmax = 0;
#ifdef use_root
  fRootHisto = 0;
#endif
  fThreshold = 0;
  fContent = 0;
  
}
  
AliHLTTPCHistogram1D::AliHLTTPCHistogram1D(Char_t *name,Char_t */*id*/,Int_t nxbin,Double_t xmin,Double_t xmax)

{
  //normal ctor
  strcpy(fName,name);
  fNbins = nxbin;
  fNcells = fNbins + 2;
  fEntries = 0;
  fXmin = xmin;
  fXmax = xmax;
#ifdef use_root
  fRootHisto = 0;
#endif
  fThreshold = 0;
  
  fContent = new Double_t[fNcells];
  Reset();
}

AliHLTTPCHistogram1D::~AliHLTTPCHistogram1D()
{
  //Destructor
  if(fContent)
    delete [] fContent;
#ifdef use_root
  if(fRootHisto)
    delete fRootHisto;
#endif
}


void AliHLTTPCHistogram1D::Reset()
{
  //Reset histogram contents
#if defined(__DECCXX)
  bzero((char *)fContent,fNcells*sizeof(Double_t));
#else
  bzero(fContent,fNcells*sizeof(Double_t));
#endif
  fEntries=0;
}

void AliHLTTPCHistogram1D::Fill(Double_t x,Int_t weight)
{
  //Fill a given bin with weight
  Int_t bin = FindBin(x);
  AddBinContent(bin,weight);
}


Int_t AliHLTTPCHistogram1D::FindBin(Double_t x) const
{
  //Find a given bin
  if(x < fXmin || x > fXmax)
    return 0;
  
  return 1 + (Int_t)(fNbins*(x-fXmin)/(fXmax-fXmin));

}

Int_t AliHLTTPCHistogram1D::GetMaximumBin() const
{
  //Find the bin with the largest content
  Double_t maxvalue=0;
  Int_t maxbin=0;
  for(Int_t i=0; i<fNcells; i++)
    {
      if(fContent[i] > maxvalue)
	{
	  maxvalue=fContent[i];
	  maxbin = i;
	}
    }
  return maxbin;
}

Double_t AliHLTTPCHistogram1D::GetBinContent(Int_t bin) const
{
  //Get bin content
  if(bin >= fNcells)
    {
      LOG(AliHLTTPCLog::kError,"AliHLTTPCHistogram::GetBinContent","array")<<AliHLTTPCLog::kDec<<
	"bin out of range "<<bin<<ENDLOG;
      return 0;
    }
  
  if(fContent[bin] < fThreshold)
    return 0;
  return fContent[bin];
}


void AliHLTTPCHistogram1D::SetBinContent(Int_t bin,Int_t value)
{
  //Set bin content
  if(bin >= fNcells)
    {
      LOG(AliHLTTPCLog::kError,"AliHLTTPCHistogram::SetBinContent","array")<<AliHLTTPCLog::kDec<<
	"bin out of range "<<bin<<ENDLOG;
      return;
    }
  if(bin == 0)
    return;
  fContent[bin]=value;
  
}

void AliHLTTPCHistogram1D::AddBinContent(Int_t bin,Int_t weight)
{
  //Add weight to bin content
  if(bin < 0 || bin > fNcells)
    {
      LOG(AliHLTTPCLog::kError,"AliHLTTPCHistogram::AddBinContent","array")<<AliHLTTPCLog::kDec<<
	"bin-value out of range "<<bin<<ENDLOG;
      return;
    }
  if(bin == 0)
    return;
  fEntries++;
  fContent[bin] += weight;
}

Double_t AliHLTTPCHistogram1D::GetBinCenter(Int_t bin) const
{
  //Get bin center  
  Double_t binwidth = (fXmax - fXmin) / fNbins;
  return fXmin + (bin-1) * binwidth + 0.5*binwidth;
  
}

void AliHLTTPCHistogram1D::Draw(Char_t *option)
{
  //Draw the histogram
  fRootHisto = new TH1F(fName,"",fNbins,fXmin,fXmax);
  for(Int_t bin=0; bin<fNcells; bin++)
    fRootHisto->AddBinContent(bin,GetBinContent(bin));
  
  fRootHisto->Draw(option);
  
}
