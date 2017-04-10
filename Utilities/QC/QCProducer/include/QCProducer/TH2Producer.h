#pragma once

#include <Rtypes.h>

#include "Producer.h"

namespace o2
{
namespace qc
{
class TH2Producer : public Producer
{
 public:
  TH2Producer(const char* histogramName, const char* histogramTitle, const int bins);
  TObject* produceData() const override;

 private:
  const char* mHistogramName;
  const char* mHistogramTitle;

  const Int_t mNbinsx;
  const Int_t mNbinsy;
  const Double_t mXlow{ -10 };
  const Double_t mXup{ 10 };
  const Double_t mYlow{ -20 };
  const Double_t mYup{ 20 };
};
}
}