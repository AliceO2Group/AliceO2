#pragma once

#include "Producer.h"

#include <string>

class HistogramProducer : public Producer
{
public:
	HistogramProducer(std::string histogramId, float xLow, float xUp);
	TObject* produceData() const override;

private:
	std::string mHistogramId;
	int mBeansNumber;
    double mXLow;
    double mXUp;
};