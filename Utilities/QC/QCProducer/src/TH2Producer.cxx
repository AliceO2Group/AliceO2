#include "QCProducer/TH2Producer.h"

#include <TF2.h>
#include <TH2F.h>

using namespace std;

namespace o2
{
namespace qc
{
TH2Producer::TH2Producer(const char* histogramName, const char* histogramTitle, const int bins)
  : mHistogramName(histogramName), mHistogramTitle(histogramTitle), mNbinsx(bins), mNbinsy(bins)
{
}

TObject* TH2Producer::produceData() const
{
  auto* histogram = new TH2F(mHistogramName, mHistogramTitle, mNbinsx, mXlow, mXup, mNbinsy, mYlow, mYup);

  for (int i = 0; i < mNbinsx; ++i) {
    for (int j = 0; j < mNbinsy; ++j) {
      histogram->Fill(i, j, 1.0);
    }
  }

  return histogram;
}
}
}