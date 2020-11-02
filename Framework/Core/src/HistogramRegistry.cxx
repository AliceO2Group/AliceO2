// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/HistogramRegistry.h"

namespace o2::framework
{

// define histogram callbacks for runtime histogram creation
#define CALLB(HType)                                            \
  {                                                             \
    k##HType,                                                   \
      [](HistogramSpec const& histSpec) {                       \
        return HistFactory::createHistVariant<HType>(histSpec); \
      }                                                         \
  }

const std::map<HistType, std::function<HistPtr(const HistogramSpec&)>> HistFactory::HistogramCreationCallbacks{
  CALLB(TH1D), CALLB(TH1F), CALLB(TH1I), CALLB(TH1C), CALLB(TH1S),
  CALLB(TH2D), CALLB(TH2F), CALLB(TH2I), CALLB(TH2C), CALLB(TH2S),
  CALLB(TH3D), CALLB(TH3F), CALLB(TH3I), CALLB(TH3C), CALLB(TH3S),
  CALLB(THnD), CALLB(THnF), CALLB(THnI), CALLB(THnC), CALLB(THnS), CALLB(THnL),
  CALLB(THnSparseD), CALLB(THnSparseF), CALLB(THnSparseI), CALLB(THnSparseC), CALLB(THnSparseS), CALLB(THnSparseL),
  CALLB(TProfile), CALLB(TProfile2D), CALLB(TProfile3D),
  //CALLB(StepTHnF), CALLB(StepTHnD)
};

#undef CALLB

void HistogramRegistry::print(bool showAxisDetails)
{
  std::vector<double> fillFractions{0.1, 0.25, 0.5};
  std::vector<double> totalSizes(fillFractions.size());

  uint32_t nHistos{};
  bool containsSparseHist{};
  auto printHistInfo = [&](auto&& hist) {
    if (hist) {
      using T = std::decay_t<decltype(*hist)>;
      bool isSparse{};
      if (hist->InheritsFrom(THnSparse::Class())) {
        isSparse = true;
        containsSparseHist = true;
      }
      ++nHistos;
      std::vector<double> sizes;
      std::string sizeInfo{};
      if (isSparse) {
        std::transform(std::begin(fillFractions), std::end(fillFractions), std::back_inserter(sizes), [&hist](auto& fraction) { return HistFiller::getSize(hist, fraction); });
        for (int i = 0; i < fillFractions.size(); ++i) {
          sizeInfo += fmt::format("{:.2f} kB ({:.0f} %)", sizes[i] * 1024, fillFractions[i] * 100);
          if (i != fillFractions.size() - 1) {
            sizeInfo += ", ";
          }
        }
      } else {
        double size = HistFiller::getSize(hist);
        sizes.resize(fillFractions.size(), size);
        sizeInfo = fmt::format("{:.2f} kB", sizes[0] * 1024);
      }
      std::transform(totalSizes.begin(), totalSizes.end(), sizes.begin(), totalSizes.begin(), std::plus<double>());
      LOGF(INFO, "Hist %03d: %-35s  %-19s [%s]", nHistos, hist->GetName(), hist->IsA()->GetName(), sizeInfo);

      if (showAxisDetails) {
        int nDim = 0;
        if constexpr (std::is_base_of_v<THnBase, T>) {
          nDim = hist->GetNdimensions();
        } else if constexpr (std::is_base_of_v<TH1, T>) {
          nDim = hist->GetDimension();
        }
        for (int d = 0; d < nDim; ++d) {
          TAxis* axis = HistFactory::getAxis(d, hist);
          LOGF(INFO, "- Axis %d: %-20s (%d bins)", d, axis->GetTitle(), axis->GetNbins());
        }
      }
    }
  };

  std::string titleString{"======================== HistogramRegistry ========================"};
  LOGF(INFO, "");
  LOGF(INFO, "%s", titleString);
  LOGF(INFO, "%s\"%s\"", std::string((int)(0.5 * titleString.size() - (1 + 0.5 * mName.size())), ' '), mName);
  std::sort(mRegisteredNames.begin(), mRegisteredNames.end());
  for (auto& curHistName : mRegisteredNames) {
    std::visit(printHistInfo, mRegistryValue[getHistIndex(curHistName.data())]);
  }
  std::string totalSizeInfo{};
  if (containsSparseHist) {
    for (int i = 0; i < totalSizes.size(); ++i) {
      totalSizeInfo += fmt::format("{:.2f} MB ({:.0f} %)", totalSizes[i], fillFractions[i] * 100);
      if (i != totalSizes.size() - 1) {
        totalSizeInfo += ", ";
      }
    }
  } else {
    totalSizeInfo = fmt::format("{:.2f} MB", totalSizes[0]);
  }
  LOGF(INFO, "%s", std::string(titleString.size(), '='), titleString);
  LOGF(INFO, "Total: %d histograms, ca. %s", nHistos, totalSizeInfo);
  LOGF(INFO, "%s", std::string(titleString.size(), '='), titleString);
  LOGF(INFO, "");
}

} // namespace o2::framework
