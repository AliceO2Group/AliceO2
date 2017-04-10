#include "QCProducer/THnProducer.h"

#include <THn.h>
#include <TRandom.h>

namespace o2
{
namespace qc
{
THnProducer::THnProducer(const char* histogramName, const char* histogramTitle, const int bins)
  : mHistogramName(histogramName), mHistogramTitle(histogramTitle), mBins(bins)
{
}

TObject* THnProducer::produceData() const
{
  Int_t dim = 4;
  Int_t bins[] = { mBins, mBins, mBins, mBins };
  Double_t xmin[] = { -10, -10, -10, -10 };
  Double_t xmax[] = { 10, 10, 10, 10 };

  const Int_t valuesNumber = 1000;
  auto* values = new Double_t[valuesNumber];

  auto* histogram = new THnF(mHistogramName, mHistogramTitle, dim, bins, xmin, xmax);

  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < valuesNumber; ++j) {
      values[j] = gRandom->Gaus(0, 1);
    }

    histogram->Fill(values);
  }

  delete[] values;

  return histogram;
}
}
}