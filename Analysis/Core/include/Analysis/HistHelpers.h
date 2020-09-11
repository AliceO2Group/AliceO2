// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author Mario Krueger <mario.kruger@cern.ch>
//
// Some helper templates to simplify working with histograms
//

#ifndef HistHelpers_H
#define HistHelpers_H

#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <THn.h>
#include <THnSparse.h>
#include <TObjArray.h>

#include "Framework/Logger.h"

namespace o2
{
namespace experimental
{
namespace histhelpers
{

template <typename T>
struct is_shared_ptr : std::false_type {
};
template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {
};

//**************************************************************************************************
/**
 * Container class for storing and saving root histograms of any type.
 * TObjArray inheritance (and concomitant raw pointer gynmnastics) is used to interface with existing O2 file writing functionality.
 */
//**************************************************************************************************
class HistContainer : public TObjArray
{
 public:
  HistContainer(const std::string& name) : TObjArray()
  {
    SetBit(TObject::kSingleKey, false); // not working; seems like WriteObjectAny ignores this...
    SetOwner(false);                    // let container handle object deletion
    SetName(name.data());
  }

  using HistType = std::variant<std::shared_ptr<TH3>, std::shared_ptr<TH2>, std::shared_ptr<TH1>, std::shared_ptr<THn>, std::shared_ptr<THnSparse>>;

  template <typename T>
  void Add(uint8_t histID, T&& hist)
  {
    if (mHistos.find(histID) != mHistos.end()) {
      LOGF(WARNING, "HistContainer %s already holds a histogram at histID = %d. Overriding it now...", GetName(), histID);
      TObject* oldPtr = nullptr;
      std::visit([&](const auto& sharedPtr) { oldPtr = sharedPtr.get(); }, mHistos[histID]);
      TObjArray::Remove(oldPtr);
    }
    // if shared poiner is provided as argument, object itself is used, otherwise copied
    if constexpr (is_shared_ptr<T>::value)
      mHistos[histID] = DownCast(hist);
    else {
      mHistos[histID] = DownCast(std::make_shared<T>(hist));
    }
    TObject* rawPtr = nullptr;
    std::visit([&](const auto& sharedPtr) { rawPtr = sharedPtr.get(); }, mHistos[histID]);
    TObjArray::Add(rawPtr);
  }

  // gets the underlying histogram pointer
  // unfortunately we cannot automatically infer type here
  // so one has to use Get<TH1>(), Get<TH2>(), Get<TH3>(), Get<THn>(), Get<THnSparse>()
  template <typename T>
  std::shared_ptr<T> Get(uint8_t histID)
  {
    return *std::get_if<std::shared_ptr<T>>(&mHistos[histID]);
  }

  template <typename... Ts>
  void Fill(uint8_t histID, Ts&&... position)
  {
    std::visit([this, &position...](auto&& hist) { GenericFill(hist, std::forward<Ts>(position)...); }, mHistos[histID]);
  }

 private:
  std::map<uint8_t, HistType> mHistos;

  // disallow user to call TObjArray::Add()
  using TObjArray::Add;

  template <typename T, typename... Ts>
  void GenericFill(std::shared_ptr<T> hist, const Ts&... position)
  {
    constexpr bool isValidTH3 = (std::is_same_v<TH3, T> && sizeof...(Ts) == 3);
    constexpr bool isValidTH2 = (std::is_same_v<TH2, T> && sizeof...(Ts) == 2);
    constexpr bool isValidTH1 = (std::is_same_v<TH1, T> && sizeof...(Ts) == 1);
    constexpr bool isValidPrimitive = isValidTH1 || isValidTH2 || isValidTH3;
    // unfortunately we dont know at compile the dimension of THn(Sparse)
    constexpr bool isValidComplex = std::is_base_of<THnBase, T>::value;

    if constexpr (isValidPrimitive) {
      hist->Fill(static_cast<double>(position)...);
    } else if constexpr (isValidComplex) {
      // savety check for n dimensional histograms (runtime overhead)
      // if(hist->GetNdimensions() != sizeof...(position)) return;
      double tempArray[] = {static_cast<double>(position)...};
      hist->Fill(tempArray);
    }
  };

  template <typename T>
  HistType DownCast(std::shared_ptr<T> hist)
  {
    // since the variant HistType only knows the interface classes (TH1,TH2,TH3)
    // assigning an actual derived type of TH2 and TH3 (e.g. TH2F, TH3I)
    // will confuse the compiler since they are both TH2(3) and TH1 at the same time
    // it will not know which alternative of HistType to select
    // therefore, in these cases we have to explicitly cast to the correct interface type first
    if constexpr (std::is_base_of_v<TH3, T>) {
      return std::static_pointer_cast<TH3>(hist);
    } else if constexpr (std::is_base_of_v<TH2, T>) {
      return std::static_pointer_cast<TH2>(hist);
    } else {
      // all other cases can be left untouched
      return hist;
    }
  }
};

//**************************************************************************************************
/**
 * Helper class to build all kinds of root histograms with a streamlined user interface.
 */
//**************************************************************************************************

struct Axis {
  std::string name{};
  std::string title{};
  std::vector<double> binEdges{};
  int nBins{}; // 0 when bin edges are specified directly FIXME: make this std::optional
};

template <typename RootHist_t>
class HistBuilder
{
 public:
  HistBuilder() : mAxes{}, mHist{nullptr} {}
  HistBuilder(const HistBuilder&) = delete;            // non construction-copyable
  HistBuilder& operator=(const HistBuilder&) = delete; // non copyable

  void AddAxis(const Axis& axis) { mAxes.push_back(axis); }
  void AddAxes(const std::vector<Axis>& axes)
  {
    mAxes.insert(mAxes.end(), axes.begin(), axes.end());
  }
  void AddAxis(const std::string& name, const std::string& title, const int& nBins,
               const double& lowerEdge, const double& upperEdge)
  {
    mAxes.push_back({name, title, {lowerEdge, upperEdge}, nBins});
  }
  void AddAxis(const std::string& name, const std::string& title,
               const std::vector<double>& binEdges)
  {
    mAxes.push_back({name, title, binEdges, 0});
  }

  std::shared_ptr<RootHist_t> GenerateHist(const std::string& name, bool hasWeights = false)
  {
    const std::size_t MAX_DIM{10};
    const std::size_t nAxes{mAxes.size()};
    if (nAxes == 0 || nAxes > MAX_DIM)
      return nullptr;

    int nBins[MAX_DIM]{0};
    double lowerBounds[MAX_DIM]{0.0};
    double upperBounds[MAX_DIM]{0.0};

    // first figure out number of bins and dimensions
    std::string title = "[ ";
    for (std::size_t i = 0; i < nAxes; i++) {
      nBins[i] = (mAxes[i].nBins) ? mAxes[i].nBins : mAxes[i].binEdges.size() - 1;
      lowerBounds[i] = mAxes[i].binEdges.front();
      upperBounds[i] = mAxes[i].binEdges.back();
      title += mAxes[i].name;
      if (i < nAxes - 1)
        title += " : ";
      else
        title += " ]";
    }

    // create histogram
    mHist.reset(HistFactory(name, title, nAxes, nBins, lowerBounds, upperBounds));

    if (!mHist) {
      LOGF(WARNING, "ERROR: The number of specified dimensions does not match the type.");
      return nullptr;
    }

    // set axis properties
    for (std::size_t i = 0; i < nAxes; i++) {
      TAxis* axis{GetAxis(i)};
      if (axis) {
        axis->SetTitle(mAxes[i].title.data());
        if constexpr (std::is_base_of_v<THnBase, RootHist_t>)
          axis->SetName((std::to_string(i) + "-" + mAxes[i].name).data());

        // move the bin edges in case a variable binnining was requested
        if (!mAxes[i].nBins) {
          if (!std::is_sorted(std::begin(mAxes[i].binEdges), std::end(mAxes[i].binEdges))) {
            LOGF(WARNING, "ERROR: The bin edges specified for axis %s in histogram %s are not in increasing order!", mAxes[i].name, name);
            return nullptr;
          }
          axis->Set(nBins[i], mAxes[i].binEdges.data());
        }
      }
    }
    if (hasWeights)
      mHist->Sumw2();
    mAxes.clear(); // clean up after creating the root histogram
    return mHist;
  }

 private:
  std::vector<Axis> mAxes;
  std::shared_ptr<RootHist_t> mHist;

  RootHist_t* HistFactory(const std::string& name, const std::string& title, const std::size_t nDim,
                          const int nBins[], const double lowerBounds[], const double upperBounds[])
  {
    if constexpr (std::is_base_of_v<THnBase, RootHist_t>) {
      return new RootHist_t(name.data(), title.data(), nDim, nBins, lowerBounds, upperBounds);
    } else if constexpr (std::is_base_of_v<TH3, RootHist_t>) {
      return (nDim != 3) ? nullptr
                         : new RootHist_t(name.data(), title.data(), nBins[0], lowerBounds[0],
                                          upperBounds[0], nBins[1], lowerBounds[1], upperBounds[1],
                                          nBins[2], lowerBounds[2], upperBounds[2]);
    } else if constexpr (std::is_base_of_v<TH2, RootHist_t>) {
      return (nDim != 2) ? nullptr
                         : new RootHist_t(name.data(), title.data(), nBins[0], lowerBounds[0],
                                          upperBounds[0], nBins[1], lowerBounds[1], upperBounds[1]);
    } else if constexpr (std::is_base_of_v<TH1, RootHist_t>) {
      return (nDim != 1)
               ? nullptr
               : new RootHist_t(name.data(), title.data(), nBins[0], lowerBounds[0], upperBounds[0]);
    }
    return nullptr;
  }

  TAxis* GetAxis(const int i)
  {
    if constexpr (std::is_base_of_v<THnBase, RootHist_t>) {
      return mHist->GetAxis(i);
    } else {
      return (i == 0) ? mHist->GetXaxis()
                      : (i == 1) ? mHist->GetYaxis() : (i == 2) ? mHist->GetZaxis() : nullptr;
    }
  }
};

} // namespace histhelpers
} // namespace experimental
} // namespace o2

#endif
