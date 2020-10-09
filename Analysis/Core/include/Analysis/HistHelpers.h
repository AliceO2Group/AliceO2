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
#include <TProfile.h>
#include <TProfile2D.h>
#include <TProfile3D.h>

#include <TFolder.h>
#include <TObjArray.h>
#include <TList.h>

#include "Framework/Logger.h"

namespace o2::experimental::histhelpers
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
 * RootContainer (TObjArray, TList or TFolder) inheritance is used to interface with O2 file writing functionality.
 */
//**************************************************************************************************
template <class RootContainer>
class HistContainer : private RootContainer
{
 public:
  HistContainer(const std::string& name) : RootContainer()
  {
    RootContainer::SetOwner(false); // let container handle object deletion
    RootContainer::SetName(name.data());
  }
  HistContainer(const HistContainer& other)
  {
    // pseudo copy ctor to move around empty collection on construction (e.g. when put in OutputObj)
    // this is needed to make HistContainer also work with TLists since these dont have a copy constructor (as opposed to TObjArrays)
    RootContainer::SetOwner(false);
    RootContainer::SetName(other.GetName());
  }

  using HistType = std::variant<std::shared_ptr<THn>, std::shared_ptr<THnSparse>, std::shared_ptr<TH3>, std::shared_ptr<TH2>, std::shared_ptr<TH1>, std::shared_ptr<TProfile3D>, std::shared_ptr<TProfile2D>, std::shared_ptr<TProfile>>;

  template <uint8_t histID, typename T>
  void Add(T&& hist)
  {
    if (mHistos.find(histID) != mHistos.end()) {
      LOGF(WARNING, "HistContainer %s already holds a histogram at histID = %d. Overriding it now...", RootContainer::GetName(), histID);
      TObject* oldPtr = nullptr;
      std::visit([&](const auto& sharedPtr) { oldPtr = sharedPtr.get(); }, mHistos[histID]);
      RootContainer::Remove(oldPtr);
    }
    // if shared pointers or rvalue raw pointers are provided as argument, the existing object is used
    // otherwise the existing object is copied
    std::optional<HistType> histVariant{};
    if constexpr (is_shared_ptr<typename std::remove_reference<T>::type>::value)
      histVariant = GetHistVariant(hist);
    else if constexpr (std::is_pointer_v<T> && std::is_rvalue_reference_v<decltype(std::forward<T>(hist))>)
      histVariant = GetHistVariant(std::shared_ptr<std::remove_pointer_t<T>>(hist));
    else {
      histVariant = GetHistVariant(std::make_shared<T>(hist));
    }
    if (histVariant) {
      mHistos[histID] = *histVariant;
      TObject* rawPtr = nullptr;
      std::visit([&](const auto& sharedPtr) { rawPtr = sharedPtr.get(); }, mHistos[histID]);
      RootContainer::Add(rawPtr);
    } else {
      LOGF(FATAL, "Could not create histogram.");
    }
  }

  // gets the underlying histogram pointer
  // we cannot automatically infer type here so it has to be explicitly specified
  // -> Get<TH1>(), Get<TH2>(), Get<TH3>(), Get<THn>(), Get<THnSparse>(), Get<TProfile>(), Get<TProfile2D>(), Get<TProfile3D>()
  template <typename T>
  std::shared_ptr<T> Get(uint8_t histID)
  {
    return *std::get_if<std::shared_ptr<T>>(&mHistos[histID]);
  }

  // fill histogram or profile with arguments x,y,z,... and weight if requested
  template <uint8_t histID, bool fillWeight = false, typename... Ts>
  void Fill(Ts&&... position)
  {
    std::visit([this, &position...](auto&& hist) { GenericFill<fillWeight>(hist, std::forward<Ts>(position)...); }, mHistos[histID]);
  }
  template <uint8_t histID, typename... Ts>
  void FillWeight(Ts&&... positionAndWeight)
  {
    Fill<histID, true>(std::forward<Ts>(positionAndWeight)...);
  }

  // make accessible only the RootContainer functions needed for writing to file
  using RootContainer::Class;
  using RootContainer::GetName;

 private:
  // FIXME: map is most likely not the fastest -> array?
  std::map<uint8_t, HistType> mHistos;

  template <bool fillWeight = false, typename T, typename... Ts>
  void GenericFill(std::shared_ptr<T> hist, const Ts&... position)
  {
    // filling with weights requires one additional argument
    constexpr bool isTH3 = (std::is_same_v<TH3, T> && sizeof...(Ts) == 3 + fillWeight);
    constexpr bool isTH2 = (std::is_same_v<TH2, T> && sizeof...(Ts) == 2 + fillWeight);
    constexpr bool isTH1 = (std::is_same_v<TH1, T> && sizeof...(Ts) == 1 + fillWeight);
    constexpr bool isTProfile3D = (std::is_same_v<TProfile3D, T> && sizeof...(Ts) == 4 + fillWeight);
    constexpr bool isTProfile2D = (std::is_same_v<TProfile2D, T> && sizeof...(Ts) == 3 + fillWeight);
    constexpr bool isTProfile = (std::is_same_v<TProfile, T> && sizeof...(Ts) == 2 + fillWeight);

    constexpr bool isValidPrimitive = isTH1 || isTH2 || isTH3 || isTProfile || isTProfile2D || isTProfile3D;
    // unfortunately we dont know at compile the dimension of THn(Sparse)
    constexpr bool isValidComplex = std::is_base_of_v<THnBase, T>;

    if constexpr (isValidPrimitive) {
      hist->Fill(static_cast<double>(position)...);
    } else if constexpr (isValidComplex) {
      // savety check for n dimensional histograms (runtime overhead)
      // if (hist->GetNdimensions() != sizeof...(position) - fillWeight) return;
      double tempArray[] = {static_cast<double>(position)...};
      if constexpr (fillWeight)
        hist->Fill(tempArray, tempArray[sizeof...(Ts) - 1]);
      else
        hist->Fill(tempArray);
    }
  }

  template <typename T>
  std::optional<HistType> GetHistVariant(std::shared_ptr<TObject> obj)
  {
    if (obj->InheritsFrom(T::Class())) {
      return std::static_pointer_cast<T>(obj);
    }
    return std::nullopt;
  }
  template <typename T, typename Next, typename... Rest>
  std::optional<HistType> GetHistVariant(std::shared_ptr<TObject> obj)
  {
    if (auto hist = GetHistVariant<T>(obj)) {
      return hist;
    }
    return GetHistVariant<Next, Rest...>(obj);
  }
  std::optional<HistType> GetHistVariant(std::shared_ptr<TObject> obj)
  {
    if (obj) {
      // TProfile3D is TH3, TProfile2D is TH2, TH3 is TH1, TH2 is TH1, TProfile is TH1
      return GetHistVariant<THn, THnSparse, TProfile3D, TH3, TProfile2D, TH2, TProfile, TH1>(obj);
    }
    return std::nullopt;
  }
};

//**************************************************************************************************
using HistFolder = HistContainer<TFolder>;
using HistArray = HistContainer<TObjArray>;
using HistList = HistContainer<TList>;
//**************************************************************************************************

//**************************************************************************************************
/**
 * Helper class to build all kinds of root histograms with a streamlined user interface.
 */
//**************************************************************************************************

struct Axis {
  std::string name{};
  std::string title{};
  std::vector<double> binEdges{};
  std::optional<int> nBins{};
};

class Hist
{
 public:
  Hist() : mAxes{} {}
  Hist(const std::vector<Axis>& axes) : mAxes{axes} {}

  void AddAxis(const Axis& axis)
  {
    mAxes.push_back(axis);
  }

  void AddAxis(const std::string& name, const std::string& title, const int nBins,
               const double lowerEdge, const double upperEdge)
  {
    mAxes.push_back({name, title, {lowerEdge, upperEdge}, nBins});
  }

  void AddAxis(const std::string& name, const std::string& title,
               const std::vector<double>& binEdges)
  {
    mAxes.push_back({name, title, binEdges});
  }

  void AddAxes(const std::vector<Axis>& axes)
  {
    mAxes.insert(mAxes.end(), axes.begin(), axes.end());
  }

  // add axes defined in other Hist object
  void AddAxes(const Hist& other)
  {
    mAxes.insert(mAxes.end(), other.GetAxes().begin(), other.GetAxes().end());
  }

  void Reset()
  {
    mAxes.clear();
  }

  // create histogram with the defined axes
  template <typename RootHistType>
  std::shared_ptr<RootHistType> Create(const std::string& name, bool useWeights = false)
  {
    const std::size_t MAX_DIM{10};
    const std::size_t nAxes{mAxes.size()};
    if (nAxes == 0 || nAxes > MAX_DIM)
      return nullptr;

    int nBins[MAX_DIM]{0};
    double lowerBounds[MAX_DIM]{0.};
    double upperBounds[MAX_DIM]{0.};

    // first figure out number of bins and dimensions
    std::string title = "[ ";
    for (std::size_t i = 0; i < nAxes; i++) {
      nBins[i] = (mAxes[i].nBins) ? *mAxes[i].nBins : mAxes[i].binEdges.size() - 1;
      lowerBounds[i] = mAxes[i].binEdges.front();
      upperBounds[i] = mAxes[i].binEdges.back();
      title += mAxes[i].name;
      if (i < nAxes - 1)
        title += " : ";
      else
        title += " ]";
    }

    // create histogram
    std::shared_ptr<RootHistType> hist(HistFactory<RootHistType>(name, title, nAxes, nBins, lowerBounds, upperBounds));

    if (!hist) {
      LOGF(FATAL, "The number of specified dimensions does not match the type.");
      return nullptr;
    }

    // set axis properties
    for (std::size_t i = 0; i < nAxes; i++) {
      TAxis* axis{GetAxis(i, hist)};
      if (axis) {
        axis->SetTitle(mAxes[i].title.data());
        if constexpr (std::is_base_of_v<THnBase, RootHistType>)
          axis->SetName((std::to_string(i) + "-" + mAxes[i].name).data());

        // move the bin edges in case a variable binning was requested
        if (!mAxes[i].nBins) {
          if (!std::is_sorted(std::begin(mAxes[i].binEdges), std::end(mAxes[i].binEdges))) {
            LOGF(FATAL, "The bin edges specified for axis %s in histogram %s are not in increasing order!", mAxes[i].name, name);
            return nullptr;
          }
          axis->Set(nBins[i], mAxes[i].binEdges.data());
        }
      }
    }
    if (useWeights)
      hist->Sumw2();
    return hist;
  }

  const std::vector<Axis>& GetAxes() const { return mAxes; };

 private:
  std::vector<Axis> mAxes;

  template <typename RootHistType>
  RootHistType* HistFactory(const std::string& name, const std::string& title, const std::size_t nDim,
                            const int nBins[], const double lowerBounds[], const double upperBounds[])
  {
    if constexpr (std::is_base_of_v<THnBase, RootHistType>) {
      return new RootHistType(name.data(), title.data(), nDim, nBins, lowerBounds, upperBounds);
    } else if constexpr (std::is_base_of_v<TH3, RootHistType>) {
      return (nDim != 3) ? nullptr
                         : new RootHistType(name.data(), title.data(), nBins[0], lowerBounds[0],
                                            upperBounds[0], nBins[1], lowerBounds[1], upperBounds[1],
                                            nBins[2], lowerBounds[2], upperBounds[2]);
    } else if constexpr (std::is_base_of_v<TH2, RootHistType>) {
      return (nDim != 2) ? nullptr
                         : new RootHistType(name.data(), title.data(), nBins[0], lowerBounds[0],
                                            upperBounds[0], nBins[1], lowerBounds[1], upperBounds[1]);
    } else if constexpr (std::is_base_of_v<TH1, RootHistType>) {
      return (nDim != 1)
               ? nullptr
               : new RootHistType(name.data(), title.data(), nBins[0], lowerBounds[0], upperBounds[0]);
    }
    return nullptr;
  }

  template <typename RootHistType>
  TAxis* GetAxis(const int i, std::shared_ptr<RootHistType> hist)
  {
    if constexpr (std::is_base_of_v<THnBase, RootHistType>) {
      return hist->GetAxis(i);
    } else {
      return (i == 0) ? hist->GetXaxis()
                      : (i == 1) ? hist->GetYaxis() : (i == 2) ? hist->GetZaxis() : nullptr;
    }
  }
};

} // namespace o2::experimental::histhelpers

#endif
