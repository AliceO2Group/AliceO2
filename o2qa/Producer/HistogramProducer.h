#pragma once

#include "Producer.h"

#include <string>

class HistogramProducer : public Producer
{
public:
	HistogramProducer(std::string histogramNamePrefix, std::string histogramTitle, float xLow, float xUp);
	TObject* produceData() override;

private:
	std::string mHistogramNamePrefix;
	std::string mHistogramTitle;
	int mBeansNumber;
    double mXLow;
    double mXUp;
    int producedHistogramNumber;
};