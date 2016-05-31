#include <TH1.h>
#include <sstream>

#include "QCProducer/HistogramProducer.h"

using namespace std;

HistogramProducer::HistogramProducer(string histogramNamePrefix, string histogramTitle, float xLow, float xUp) : mProducedHistogramNumber(0)
{
	mHistogramNamePrefix = histogramNamePrefix;
	mHistogramTitle = histogramTitle;
	mBeansNumber = 100;
	mXLow = xLow;
	mXUp = xUp;
}

TObject* HistogramProducer::produceData()
{
	ostringstream histogramName;
	string histogramTitle = "Gauss_distribution";

	histogramName << mHistogramNamePrefix << mProducedHistogramNumber++;

  TH1F* histogram = new TH1F(histogramName.str().c_str(), mHistogramTitle.c_str(), mBeansNumber, mXLow, mXUp);
  histogram->FillRandom("gaus", 1000);
  return histogram;
}
