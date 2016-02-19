#include <TH1.h>

#include "HistogramProducer.h"

using namespace std;

HistogramProducer::HistogramProducer(string histogramId, float xLow, float xUp)
{
	mHistogramProducerId = histogramId;
	mBeansNumber = 100;
	mXLow = xLow;
	mXUp = xUp;
}

TObject* HistogramProducer::produceData()
{
    auto histogram = new TH1F(mHistogramProducerId.c_str(), "Gauss distribution", mBeansNumber, mXLow, mXUp);
    histogram->FillRandom("gaus", 1000);
    return histogram;
}