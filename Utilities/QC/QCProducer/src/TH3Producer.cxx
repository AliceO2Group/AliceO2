#include "QCProducer/TH3Producer.h"

#include <TH3.h>
#include <TRandom.h>

using namespace std;

namespace o2
{
namespace qc
{
TH3Producer::TH3Producer(const char* histogramName, const char* histogramTitle, const int bins)
  : mHistogramName(histogramName), mHistogramTitle(histogramTitle), mNbinsx(bins), mNbinsy(bins), mNbinsz(bins)
{
}

TObject* TH3Producer::produceData() const
{
  auto* histogram =
    new TH3F(mHistogramName, mHistogramTitle, mNbinsx, mXlow, mXup, mNbinsy, mYlow, mYup, mNbinsz, mZlow, mZup);

  Double_t x, y, z;

  for (Int_t i = 0; i < 1000; ++i) {
    gRandom->Rannor(x, y);
    z = x * x + y * y;
    histogram->Fill(x, y, z);
  }

  return histogram;
}
}
}