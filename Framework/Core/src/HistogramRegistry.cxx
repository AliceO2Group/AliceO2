// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/HistogramRegistry.h"
#include <regex>
#include <TList.h>
#include <TClass.h>

namespace o2::framework
{

template void HistogramRegistry::fill(const HistName& histName, double);
template void HistogramRegistry::fill(const HistName& histName, float);
template void HistogramRegistry::fill(const HistName& histName, int);

constexpr HistogramRegistry::HistName::HistName(char const* const name)
  : str(name),
    hash(runtime_hash(name)),
    idx(hash & REGISTRY_BITMASK)
{
}

HistogramRegistry::HistogramRegistry(char const* const name, std::vector<HistogramSpec> histSpecs, OutputObjHandlingPolicy policy, bool sortHistos, bool createRegistryDir)
  : mName(name), mPolicy(policy), mCreateRegistryDir(createRegistryDir), mSortHistos(sortHistos), mRegistryKey(), mRegistryValue()
{
  mRegistryKey.fill(0u);
  for (auto& histSpec : histSpecs) {
    insert(histSpec);
  }
}

// return the OutputSpec associated to the HistogramRegistry
OutputSpec const HistogramRegistry::spec()
{
  header::DataDescription desc{};
  auto lhash = runtime_hash(mName.data());
  std::memset(desc.str, '_', 16);
  std::stringstream s;
  s << std::hex << lhash;
  s << std::hex << mTaskHash;
  s << std::hex << reinterpret_cast<uint64_t>(this);
  std::memcpy(desc.str, s.str().data(), 12);
  return OutputSpec{OutputLabel{mName}, "ATSK", desc, 0, Lifetime::QA};
}

OutputRef HistogramRegistry::ref(uint16_t pipelineIndex, uint16_t pipelineSize) const
{
  OutputObjHeader header{mPolicy, OutputObjSourceType::HistogramRegistrySource, mTaskHash, pipelineIndex, pipelineSize};
  // Copy the name of the registry to the haeder.
  strncpy(header.containerName, mName.data(), 64);
  return OutputRef{std::string{mName}, 0, o2::header::Stack{header}};
}

void HistogramRegistry::setHash(uint32_t hash)
{
  mTaskHash = hash;
}

// create histogram from specification and insert it into the registry
HistPtr HistogramRegistry::insert(const HistogramSpec& histSpec)
{
  validateHistName(histSpec.name, histSpec.hash);
  const uint32_t idx = imask(histSpec.hash);
  for (auto i = 0u; i < MAX_REGISTRY_SIZE; ++i) {
    TObject* rawPtr = nullptr;
    std::visit([&](const auto& sharedPtr) { rawPtr = sharedPtr.get(); }, mRegistryValue[imask(idx + i)]);
    if (!rawPtr) {
      registerName(histSpec.name);
      mRegistryKey[imask(idx + i)] = histSpec.hash;
      mRegistryValue[imask(idx + i)] = HistFactory::createHistVariant(histSpec);
      lookup += i;
      return mRegistryValue[imask(idx + i)];
    }
  }
  LOGF(fatal, R"(Internal array of HistogramRegistry "%s" is full.)", mName);
  return HistPtr();
}

// helper function that checks if histogram name can be used in registry
void HistogramRegistry::validateHistName(const std::string& name, const uint32_t hash)
{
  // check that there are still slots left in the registry
  if (mRegisteredNames.size() == MAX_REGISTRY_SIZE) {
    LOGF(fatal, R"(HistogramRegistry "%s" is full! It can hold only %d histograms.)", mName, MAX_REGISTRY_SIZE);
  }

  // validate that hash is unique
  auto it = std::find(mRegistryKey.begin(), mRegistryKey.end(), hash);
  if (it != mRegistryKey.end()) {
    auto idx = it - mRegistryKey.begin();
    std::string collidingName{};
    std::visit([&](const auto& hist) { collidingName = hist->GetName(); }, mRegistryValue[idx]);
    LOGF(fatal, R"(Hash collision in HistogramRegistry "%s"! Please rename histogram "%s" or "%s".)", mName, name, collidingName);
  }

  // validate that name contains only allowed characters
  if (!std::regex_match(name, std::regex("([a-zA-Z0-9])(([\\/_-])?[a-zA-Z0-9])*"))) {
    LOGF(fatal, R"(Histogram name "%s" contains invalid characters. Only letters, numbers, and (except for the beginning or end of the word) the special characters '/', '_', '-' are allowed.)", name);
  }
}

HistPtr HistogramRegistry::add(const HistogramSpec& histSpec)
{
  return insert(histSpec);
}

HistPtr HistogramRegistry::add(char const* const name, char const* const title, const HistogramConfigSpec& histConfigSpec, bool callSumw2)
{
  return insert({name, title, histConfigSpec, callSumw2});
}

HistPtr HistogramRegistry::add(char const* const name, char const* const title, HistType histType, const std::vector<AxisSpec>& axes, bool callSumw2)
{
  return insert({name, title, {histType, axes}, callSumw2});
}

HistPtr HistogramRegistry::add(const std::string& name, char const* const title, HistType histType, const std::vector<AxisSpec>& axes, bool callSumw2)
{
  return add(name.c_str(), title, histType, axes, callSumw2);
}

// store a copy of an existing histogram (or group of histograms) under a different name
void HistogramRegistry::addClone(const std::string& source, const std::string& target)
{
  auto doInsertClone = [&](const auto& sharedPtr) {
    if (!sharedPtr.get()) {
      return;
    }
    std::string sourceName{((TNamed*)sharedPtr.get())->GetName()};
    // search for histograms starting with source_ substring
    if (sourceName.rfind(source, 0) == 0) {
      // when cloning groups of histograms source_ and target_ must end with "/"
      if (sourceName.size() != source.size() && (source.back() != '/' || target.back() != '/')) {
        return;
      }
      // when cloning a single histogram the specified target_ must not be a group name
      if (sourceName.size() == source.size() && target.back() == '/') {
        LOGF(fatal, "Cannot turn histogram into folder!");
      }
      std::string targetName{target};
      targetName += sourceName.substr(sourceName.find(source) + source.size());
      insertClone(targetName.data(), sharedPtr);
    }
  };

  for (auto& histVariant : mRegistryValue) {
    std::visit(doInsertClone, histVariant);
  }
}

// function to query if name is already in use
bool HistogramRegistry::contains(const HistName& histName)
{
  // check for all occurances of the hash
  auto iter = mRegistryKey.begin();
  while ((iter = std::find(iter, mRegistryKey.end(), histName.hash)) != mRegistryKey.end()) {
    const char* curName = nullptr;
    std::visit([&](auto&& hist) { if(hist) { curName = hist->GetName(); } }, mRegistryValue[iter - mRegistryKey.begin()]);
    // if hash is the same, make sure that name is indeed the same
    if (strcmp(curName, histName.str) == 0) {
      return true;
    }
  }
  return false;
}

// get rough estimate for size of histogram stored in registry
double HistogramRegistry::getSize(const HistName& histName, double fillFraction)
{
  double size{};
  std::visit([&fillFraction, &size](auto&& hist) { size = HistFiller::getSize(hist, fillFraction); }, mRegistryValue[getHistIndex(histName)]);
  return size;
}

// get rough estimate for size of all histograms stored in registry
double HistogramRegistry::getSize(double fillFraction)
{
  double size{};
  for (auto j = 0u; j < MAX_REGISTRY_SIZE; ++j) {
    std::visit([&fillFraction, &size](auto&& hist) { if(hist) { size += HistFiller::getSize(hist, fillFraction);} }, mRegistryValue[j]);
  }
  return size;
}

void HistogramRegistry::clean()
{
  for (auto& value : mRegistryValue) {
    std::visit([](auto&& hist) { hist.reset(); }, value);
  }
}

// print some useful meta-info about the stored histograms
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
      LOGF(info, "Hist %03d: %-35s  %-19s [%s]", nHistos, hist->GetName(), hist->IsA()->GetName(), sizeInfo);

      if (showAxisDetails) {
        int nDim = 0;
        if constexpr (std::is_base_of_v<THnBase, T>) {
          nDim = hist->GetNdimensions();
        } else if constexpr (std::is_base_of_v<TH1, T>) {
          nDim = hist->GetDimension();
        }
        TAxis* axis{nullptr};
        for (int d = 0; d < nDim; ++d) {
          if constexpr (std::is_base_of_v<THnBase, T> || std::is_base_of_v<StepTHn, T>) {
            axis = hist->GetAxis(d);
          } else {
            if (d == 0) {
              axis = hist->GetXaxis();
            } else if (d == 1) {
              axis = hist->GetYaxis();
            } else if (d == 2) {
              axis = hist->GetZaxis();
            }
          }
          LOGF(info, "- Axis %d: %-20s (%d bins)", d, axis->GetTitle(), axis->GetNbins());
        }
      }
    }
  };

  std::string titleString{"======================== HistogramRegistry ========================"};
  LOGF(info, "");
  LOGF(info, "%s", titleString);
  LOGF(info, "%s\"%s\"", std::string((int)(0.5 * titleString.size() - (1 + 0.5 * mName.size())), ' '), mName);
  for (auto& curHistName : mRegisteredNames) {
    std::visit(printHistInfo, mRegistryValue[getHistIndex(HistName{curHistName.data()})]);
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
  LOGF(info, "%s", std::string(titleString.size(), '='), titleString);
  LOGF(info, "Total: %d histograms, ca. %s", nHistos, totalSizeInfo);
  if (lookup) {
    LOGF(info, "Due to index collisions, histograms were shifted by %d registry slots in total.", lookup);
  }
  LOGF(info, "%s", std::string(titleString.size(), '='), titleString);
  LOGF(info, "");
}

void HistogramRegistry::apply(std::function<void(HistogramRegistry const&, TNamed* named)> callback) const
{
  // Keep the list sorted as originally done to avoid hidden dependency on the order, for now , for now.
  auto finalList = mRegisteredNames;
  auto caseInsensitiveCompare = [](const std::string& s1, const std::string& s2) {
    return std::lexicographical_compare(s1.begin(), s1.end(), s2.begin(), s2.end(),
                                        [](char c1, char c2) { return std::tolower(static_cast<unsigned char>(c1)) < std::tolower(static_cast<unsigned char>(c2)); });
  };
  if (mSortHistos) {
    std::sort(finalList.begin(), finalList.end(), caseInsensitiveCompare);
  }
  for (auto& curHistName : finalList) {
    TNamed* rawPtr = nullptr;
    std::visit([&](const auto& sharedPtr) { rawPtr = (TNamed*)sharedPtr.get(); }, mRegistryValue[getHistIndex(HistName{curHistName.data()})]);
    if (!rawPtr) {
      // Skipping empty histograms
      continue;
    }
    callback(*this, rawPtr);
  }
}

// helper function to split user defined path/to/hist/name string
std::deque<std::string> HistogramRegistry::splitPath(const std::string& pathAndNameUser)
{
  std::istringstream pathAndNameStream(pathAndNameUser);
  std::deque<std::string> pathAndName;
  std::string curDir;
  while (std::getline(pathAndNameStream, curDir, '/')) {
    pathAndName.push_back(curDir);
  }
  return pathAndName;
}

// helper function that checks if name of histogram is reasonable and keeps track of names already in use
void HistogramRegistry::registerName(const std::string& name)
{
  if (name.empty() || name.back() == '/') {
    LOGF(fatal, "Invalid name for a histogram.");
  }
  std::deque<std::string> path = splitPath(name);
  std::string cumulativeName{};
  int depth = path.size();
  for (auto& step : path) {
    if (step.empty()) {
      LOGF(fatal, R"(Found empty group name in path for histogram "%s".)", name);
    }
    cumulativeName += step;
    for (auto& curName : mRegisteredNames) {
      // there is already a histogram where we want to put a folder or histogram
      if (cumulativeName == curName) {
        LOGF(fatal, R"(Histogram name "%s" is not compatible with existing names.)", name);
      }
      // for the full new histogram name we need to check that none of the existing histograms already uses this as a group name
      if (depth == 1) {
        if (curName.rfind(cumulativeName, 0) == 0 && curName.size() > cumulativeName.size() && curName.at(cumulativeName.size()) == '/') {
          LOGF(fatal, R"(Histogram name "%s" is not compatible with existing names.)", name);
        }
      }
    }
    --depth;
    cumulativeName += "/";
  }
  mRegisteredNames.push_back(name);
}

void HistFiller::badHistogramFill(char const* name)
{
  LOGF(fatal, "The number of arguments in fill function called for histogram %s is incompatible with histogram dimensions.", name);
}

template <typename T>
HistPtr HistogramRegistry::insertClone(const HistName& histName, const std::shared_ptr<T> originalHist)
{
  validateHistName(histName.str, histName.hash);
  for (auto i = 0u; i < MAX_REGISTRY_SIZE; ++i) {
    TObject* rawPtr = nullptr;
    std::visit([&](const auto& sharedPtr) { rawPtr = sharedPtr.get(); }, mRegistryValue[imask(histName.idx + i)]);
    if (!rawPtr) {
      registerName(histName.str);
      mRegistryKey[imask(histName.idx + i)] = histName.hash;
      mRegistryValue[imask(histName.idx + i)] = std::shared_ptr<T>(static_cast<T*>(originalHist->Clone(histName.str)));
      lookup += i;
      return mRegistryValue[imask(histName.idx + i)];
    }
  }
  LOGF(fatal, R"(Internal array of HistogramRegistry "%s" is full.)", mName);
  return HistPtr();
}

template HistPtr HistogramRegistry::insertClone(const HistName&, const std::shared_ptr<TH1>);
template HistPtr HistogramRegistry::insertClone(const HistName&, const std::shared_ptr<TH2>);
template HistPtr HistogramRegistry::insertClone(const HistName&, const std::shared_ptr<TH3>);
template HistPtr HistogramRegistry::insertClone(const HistName&, const std::shared_ptr<TProfile>);
template HistPtr HistogramRegistry::insertClone(const HistName&, const std::shared_ptr<TProfile2D>);
template HistPtr HistogramRegistry::insertClone(const HistName&, const std::shared_ptr<TProfile3D>);
template HistPtr HistogramRegistry::insertClone(const HistName&, const std::shared_ptr<THnSparse>);
template HistPtr HistogramRegistry::insertClone(const HistName&, const std::shared_ptr<THn>);
template HistPtr HistogramRegistry::insertClone(const HistName&, const std::shared_ptr<StepTHn>);

template <typename T>
std::shared_ptr<T> HistogramRegistry::add(char const* const name, char const* const title, const HistogramConfigSpec& histConfigSpec, bool callSumw2)
{
  auto histVariant = add(name, title, histConfigSpec, callSumw2);
  if (auto histPtr = std::get_if<std::shared_ptr<T>>(&histVariant)) {
    return *histPtr;
  } else {
    throw runtime_error_f(R"(Histogram type specified in add<>("%s") does not match the actual type of the histogram!)", name);
  }
}

template <typename T>
std::shared_ptr<T> HistogramRegistry::add(char const* const name, char const* const title, HistType histType, const std::vector<AxisSpec>& axes, bool callSumw2)
{
  auto histVariant = add(name, title, histType, axes, callSumw2);
  if (auto histPtr = std::get_if<std::shared_ptr<T>>(&histVariant)) {
    return *histPtr;
  } else {
    throw runtime_error_f(R"(Histogram type specified in add<>("%s") does not match the actual type of the histogram!)", name);
  }
}

template std::shared_ptr<TH1> HistogramRegistry::add<TH1>(char const* const name, char const* const title, const HistogramConfigSpec& histConfigSpec, bool callSumw2);
template std::shared_ptr<TH1> HistogramRegistry::add<TH1>(char const* const name, char const* const title, HistType histType, const std::vector<AxisSpec>& axes, bool callSumw2);
template std::shared_ptr<TH2> HistogramRegistry::add<TH2>(char const* const name, char const* const title, const HistogramConfigSpec& histConfigSpec, bool callSumw2);
template std::shared_ptr<TH2> HistogramRegistry::add<TH2>(char const* const name, char const* const title, HistType histType, const std::vector<AxisSpec>& axes, bool callSumw2);
template std::shared_ptr<TH3> HistogramRegistry::add<TH3>(char const* const name, char const* const title, const HistogramConfigSpec& histConfigSpec, bool callSumw2);
template std::shared_ptr<TH3> HistogramRegistry::add<TH3>(char const* const name, char const* const title, HistType histType, const std::vector<AxisSpec>& axes, bool callSumw2);
template std::shared_ptr<TProfile> HistogramRegistry::add<TProfile>(char const* const name, char const* const title, const HistogramConfigSpec& histConfigSpec, bool callSumw2);
template std::shared_ptr<TProfile> HistogramRegistry::add<TProfile>(char const* const name, char const* const title, HistType histType, const std::vector<AxisSpec>& axes, bool callSumw2);
template std::shared_ptr<TProfile2D> HistogramRegistry::add<TProfile2D>(char const* const name, char const* const title, const HistogramConfigSpec& histConfigSpec, bool callSumw2);
template std::shared_ptr<TProfile2D> HistogramRegistry::add<TProfile2D>(char const* const name, char const* const title, HistType histType, const std::vector<AxisSpec>& axes, bool callSumw2);
template std::shared_ptr<TProfile3D> HistogramRegistry::add<TProfile3D>(char const* const name, char const* const title, const HistogramConfigSpec& histConfigSpec, bool callSumw2);
template std::shared_ptr<TProfile3D> HistogramRegistry::add<TProfile3D>(char const* const name, char const* const title, HistType histType, const std::vector<AxisSpec>& axes, bool callSumw2);
template std::shared_ptr<THn> HistogramRegistry::add<THn>(char const* const name, char const* const title, const HistogramConfigSpec& histConfigSpec, bool callSumw2);
template std::shared_ptr<THn> HistogramRegistry::add<THn>(char const* const name, char const* const title, HistType histType, const std::vector<AxisSpec>& axes, bool callSumw2);
template std::shared_ptr<THnSparse> HistogramRegistry::add<THnSparse>(char const* const name, char const* const title, const HistogramConfigSpec& histConfigSpec, bool callSumw2);
template std::shared_ptr<THnSparse> HistogramRegistry::add<THnSparse>(char const* const name, char const* const title, HistType histType, const std::vector<AxisSpec>& axes, bool callSumw2);
template std::shared_ptr<StepTHn> HistogramRegistry::add<StepTHn>(char const* const name, char const* const title, const HistogramConfigSpec& histConfigSpec, bool callSumw2);
template std::shared_ptr<StepTHn> HistogramRegistry::add<StepTHn>(char const* const name, char const* const title, HistType histType, const std::vector<AxisSpec>& axes, bool callSumw2);

} // namespace o2::framework
