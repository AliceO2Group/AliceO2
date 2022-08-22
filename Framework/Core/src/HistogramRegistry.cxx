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

namespace o2::framework
{

constexpr HistogramRegistry::HistName::HistName(char const* const name)
  : str(name),
    hash(compile_time_hash(name)),
    idx(hash & REGISTRY_BITMASK)
{
}

HistogramRegistry::HistogramRegistry(char const* const name, std::vector<HistogramSpec> histSpecs, OutputObjHandlingPolicy policy, bool sortHistos, bool createRegistryDir)
  : mName(name), mPolicy(policy), mRegistryKey(), mRegistryValue(), mSortHistos(sortHistos), mCreateRegistryDir(createRegistryDir)
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
  auto lhash = compile_time_hash(mName.data());
  std::memset(desc.str, '_', 16);
  std::stringstream s;
  s << std::hex << lhash;
  s << std::hex << mTaskHash;
  s << std::hex << reinterpret_cast<uint64_t>(this);
  std::memcpy(desc.str, s.str().data(), 12);
  return OutputSpec{OutputLabel{mName}, "ATSK", desc, 0, Lifetime::QA};
}

OutputRef HistogramRegistry::ref()
{
  return OutputRef{std::string{mName}, 0, o2::header::Stack{OutputObjHeader{mPolicy, OutputObjSourceType::HistogramRegistrySource, mTaskHash}}};
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

// create output structure will be propagated to file-sink
TList* HistogramRegistry::operator*()
{
  TList* list = new TList();
  list->SetName(mName.data());

  if (mSortHistos) {
    auto caseInsensitiveCompare = [](const std::string& s1, const std::string& s2) {
      return std::lexicographical_compare(s1.begin(), s1.end(), s2.begin(), s2.end(),
                                          [](char c1, char c2) { return std::tolower(static_cast<unsigned char>(c1)) < std::tolower(static_cast<unsigned char>(c2)); });
    };
    std::sort(mRegisteredNames.begin(), mRegisteredNames.end(), caseInsensitiveCompare);
  }

  for (auto& curHistName : mRegisteredNames) {
    TNamed* rawPtr = nullptr;
    std::visit([&](const auto& sharedPtr) { rawPtr = (TNamed*)sharedPtr.get(); }, mRegistryValue[getHistIndex(HistName{curHistName.data()})]);
    if (rawPtr) {
      std::deque<std::string> path = splitPath(rawPtr->GetName());
      std::string name = path.back();
      path.pop_back();
      TList* targetList{getSubList(list, path)};
      if (targetList) {
        rawPtr->SetName(name.data());
        targetList->Add(rawPtr);
      } else {
        LOGF(fatal, "Specified subfolder could not be created.");
      }
    }
  }

  // place lists always at the top
  std::function<void(TList*)> moveListsToTop;
  moveListsToTop = [&](TList* list) {
    TIter next(list);
    TNamed* subList = nullptr;
    std::vector<TObject*> subLists;
    while ((subList = (TNamed*)next())) {
      if (subList->InheritsFrom(TList::Class())) {
        subLists.push_back(subList);
        moveListsToTop((TList*)subList);
      }
    }
    std::reverse(subLists.begin(), subLists.end());
    for (auto curList : subLists) {
      list->Remove(curList);
      list->AddFirst(curList);
    }
  };
  moveListsToTop(list);

  // create dedicated directory containing all of the registrys histograms
  if (mCreateRegistryDir) {
    // propagate this to the writer by adding a 'flag' to the output list
    list->AddLast(new TNamed("createFolder", ""));
  }
  return list;
}

// helper function to create resp. find the subList defined by path
TList* HistogramRegistry::getSubList(TList* list, std::deque<std::string>& path)
{
  if (path.empty()) {
    return list;
  }
  TList* targetList{nullptr};
  std::string nextList = path[0];
  path.pop_front();
  if (auto subList = (TList*)list->FindObject(nextList.data())) {
    if (subList->InheritsFrom(TList::Class())) {
      targetList = getSubList((TList*)subList, path);
    } else {
      return nullptr;
    }
  } else {
    subList = new TList();
    subList->SetName(nextList.data());
    list->Add(subList);
    targetList = getSubList(subList, path);
  }
  return targetList;
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

} // namespace o2::framework
