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

/**
 * @file Error.cxx
 * @brief implementation of the MCH processing errors
 * @author Philippe Pillot, Subatech
 */

#include "MCHBase/Error.h"

#include <fmt/format.h>

namespace o2::mch
{

const std::map<ErrorGroup, std::string> Error::groupNames = {
  {ErrorGroup::Unassigned, "Unassigned"},
  {ErrorGroup::Decoding, "Decoding"},
  {ErrorGroup::Filtering, "Filtering"},
  {ErrorGroup::TimeClustering, "TimeClustering"},
  {ErrorGroup::PreClustering, "PreClustering"},
  {ErrorGroup::Clustering, "Clustering"},
  {ErrorGroup::Tracking, "Tracking"}};

const std::map<ErrorType, std::string> Error::typeNames = {
  {ErrorType::PreClustering_MultipleDigitsInSamePad, "MultipleDigitsInSamePad"},
  {ErrorType::PreClustering_LostDigit, "LostDigit"},
  {ErrorType::Clustering_TooManyLocalMaxima, "TooManyLocalMaxima"},
  {ErrorType::Tracking_TooManyCandidates, "TooManyCandidates"},
  {ErrorType::Tracking_TooLong, "TooLong"}};

const std::map<ErrorType, std::string> Error::typeDescriptions = {
  {ErrorType::PreClustering_MultipleDigitsInSamePad, "multiple digits on the same pad"},
  {ErrorType::PreClustering_LostDigit, "lost digit"},
  {ErrorType::Clustering_TooManyLocalMaxima, "too many local maxima"},
  {ErrorType::Tracking_TooManyCandidates, "too many track candidates"},
  {ErrorType::Tracking_TooLong, "too long"}};

const std::map<ErrorType, std::string> Error::getTypeNames(ErrorGroup group)
{
  std::map<ErrorType, std::string> groupTypeNames{};
  for (const auto& typeName : typeNames) {
    if (errorGroup(typeName.first) == group) {
      groupTypeNames.emplace(typeName);
    }
  }
  return groupTypeNames;
}

std::string Error::getGroupName() const
{
  const auto itName = groupNames.find(getGroup());
  if (itName != groupNames.end()) {
    return itName->second;
  }
  return "Unknown";
}

std::string Error::getTypeName() const
{
  const auto itName = typeNames.find(type);
  if (itName != typeNames.end()) {
    return itName->second;
  }
  return "Unknown";
}

std::string Error::getTypeDescription() const
{
  const auto itDescription = typeDescriptions.find(type);
  if (itDescription != typeDescriptions.end()) {
    return itDescription->second;
  }
  return "";
}

std::string Error::asString() const
{
  auto description = fmt::format("{} error: {}", getGroupName(), getTypeDescription());

  // add extra description when relevant
  switch (type) {
    case ErrorType::PreClustering_MultipleDigitsInSamePad:
      description += fmt::format(" (DE {} pad {})", id0, id1);
      break;
    case ErrorType::Clustering_TooManyLocalMaxima:
      description += fmt::format(" (DE {})", id0);
      break;
    default:
      break;
  }

  return description + fmt::format(": seen {} time{}", count, count > 1 ? "s" : "");
}

} // namespace o2::mch
