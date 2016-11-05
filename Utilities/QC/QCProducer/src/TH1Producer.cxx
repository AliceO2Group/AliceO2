#include <TH1.h>

#include "QCProducer/TH1Producer.h"

TH1Producer::TH1Producer(const char * histogramName, const char * histogramTitle, const int numberOfbins) : mBeansNumber(numberOfbins)
{
	mHistogramName = histogramName;
	mHistogramTitle = histogramTitle;
}

TObject* TH1Producer::produceData() const
{
  TH1F* histogram = new TH1F(mHistogramName, mHistogramTitle, mBeansNumber, mXLow, mXUp);
  histogram->FillRandom("gaus", 1000);

  return histogram;
}
