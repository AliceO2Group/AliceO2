#pragma once

#include <Rtypes.h>

#include "Producer.h"

namespace o2
{
namespace qc
{
class TH3Producer : public Producer
{
 public:
  TH3Producer(const char* histogramName, const char* histogramTitle, const int bins);
  TObject* produceData() const override;

 private:
  const char* mHistogramName;
  const char* mHistogramTitle;

  const Int_t mNbinsx;
  const Int_t mNbinsy;
  const Int_t mNbinsz;

  const Double_t mXlow{ -10 };
  const Double_t mXup{ 10 };

  const Double_t mYlow{ -20 };
  const Double_t mYup{ 20 };

  const Double_t mZlow{ -30 };
  const Double_t mZup{ 30 };
};
}
}