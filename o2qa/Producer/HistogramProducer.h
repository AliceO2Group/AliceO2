#pragma once

#include <string>

#include "Producer.h"

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
