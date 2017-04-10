#pragma once

#include <Rtypes.h>

#include "Producer.h"

namespace o2
{
namespace qc
{
class THnProducer : public Producer
{
 public:
  THnProducer(const char* mHistogramName, const char* mHistogramTitle, const int bins);
  TObject* produceData() const override;

 private:
  const int mBins;
  const char* mHistogramName;
  const char* mHistogramTitle;
};
}
}