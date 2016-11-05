#pragma once

#include "Producer.h"

class TH1Producer : public Producer
{
public:
	TH1Producer(const char * histogramName, const char * histogramTitle, const int numberOfBins);
	TObject* produceData() const override;

private:
	const char * mHistogramName;
	const char * mHistogramTitle;

  const int mBeansNumber;
  const double mXLow {-10};
  const double mXUp {10};
  const float mNumberOfEntries {1000};
};
