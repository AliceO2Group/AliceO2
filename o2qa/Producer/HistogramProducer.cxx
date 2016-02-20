#include <TH1.h>

#include "HistogramProducer.h"

using namespace std;

HistogramProducer::HistogramProducer(string histogramId, float xLow, float xUp)
{
	mHistogramId = histogramId;
	mBeansNumber = 100;
	mXLow = xLow;
	mXUp = xUp;
}

TObject* HistogramProducer::produceData() const
{
    TH1F* histogram = new TH1F(mHistogramId.c_str(), "Gauss distribution", mBeansNumber, mXLow, mXUp);
    histogram->FillRandom("gaus", 1000);
    return histogram;
}