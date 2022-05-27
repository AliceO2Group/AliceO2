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

#ifndef GRP_LHCIF_FILE_READER_H_
#define GRP_LHCIF_FILE_READER_H_

#include "Rtypes.h"
#include <gsl/span>
#include "Framework/Logger.h"
#include "CommonUtils/StringUtils.h"
#include "CommonConstants/LHCConstants.h"

/// @brief Class to read the LHC InterFace file coming from the DCS filepush service

namespace o2
{
namespace grp
{
class LHCIFfileReader
{
 public:
  LHCIFfileReader() = default;  // default constructor
  ~LHCIFfileReader() = default; // default destructor

  void loadLHCIFfile(const std::string& fileName);     // load LHCIF file
  void loadLHCIFfile(gsl::span<const char> configBuf); // load LHCIF file from buffer
  template <typename T>
  void readValue(const std::string& alias, std::string& type, int& nel, int& nmeas, std::vector<std::pair<long, std::vector<T>>>& meas);

 private:
  std::string mFileBuffStr; // buffer containing content of LHC IF file

  ClassDefNV(LHCIFfileReader, 1);
};

template <typename T>
void LHCIFfileReader::readValue(const std::string& alias, std::string& type, int& nele, int& nmeas, std::vector<std::pair<long, std::vector<T>>>& meas)
{
  // look for value 'value' in the string from the LHC

  auto posStart = mFileBuffStr.find(alias);
  if (posStart == std::string::npos) {
    LOG(info) << alias << " not found in LHC IF file";
    return;
  }
  auto posEnd = mFileBuffStr.find("\n", posStart);
  LOG(debug) << "posStart = " << posStart << ", posEnd = " << posEnd;
  if (posEnd == std::string::npos) {
    posEnd = mFileBuffStr.size();
  }
  std::string subStr = mFileBuffStr.substr(posStart, posEnd - posStart);
  LOG(debug) << "subStr = " << subStr;
  auto tokensStr = o2::utils::Str::tokenize(subStr, '\t', true, false);
  LOG(debug) << "size of tokensStr = " << tokensStr.size();
  if (tokensStr.size() < 5) {
    LOG(fatal) << "Number of tokens too small: " << tokensStr.size() << ", should be at 5 (alias, type, nelements, value(s), timestamp(s)";
  }
  auto tokensStr_type = o2::utils::Str::tokenize(tokensStr[1], ':', true, false);
  LOG(debug) << "size of tokensStr_type = " << tokensStr_type.size();

  type = tokensStr_type[0];
  LOG(debug) << "type = " << type;

  nele = std::stoi(tokensStr_type[1]); // number of elements per measurement
  nmeas = std::stoi(tokensStr[2]);     // number of measurements
  LOG(debug) << "nele = " << nele << ", nmeas = " << nmeas;
  int shift = 3;                                          // number of tokens that are not measurments (alias, type, number of measurements)

  // RS: this check is wrong: the provided pair might be simply empty, but they are requested by the check above (Number of tokens too small...)
  //  if ((tokensStr.size() - shift) != (nele + 1) * nmeas) { // +1 to account for the timestamp
  //    LOG(fatal) << "Wrong number of pairs (value(s), timestamp): " << tokensStr.size() - 3 << ", should be " << (nele + 1) * nmeas;
  //  }
  meas.reserve(nmeas);

  for (int idx = 0; idx < nmeas; ++idx) {
    std::vector<T> vect;
    vect.reserve(nele);
    if constexpr (std::is_same<T, int32_t>::value) {
      if (type == "i" || type == "b") {
        for (int iele = 0; iele < nele; ++iele) {
          LOG(debug) << alias << ": value int/bool = " << tokensStr[shift + iele];
          vect.emplace_back(std::stoi(tokensStr[shift + iele]));
        }
      } else {
        LOG(fatal) << "templated function called with wrong type, should be int32_t or bool, but it is " << type;
      }
    } else if constexpr (std::is_same<T, float>::value) {
      if (type == "f") {
        for (int iele = 0; iele < nele; ++iele) {
          LOG(debug) << alias << ": value float = " << tokensStr[shift + iele];
          vect.emplace_back(std::stof(tokensStr[shift + iele]));
        }
      } else {
        LOG(fatal) << "templated function called with wrong type, should be float";
      }
    }

    else if constexpr (std::is_same<T, std::string>::value) {
      if (type == "s") {
        for (int iele = 0; iele < nele; ++iele) {
          LOG(debug) << alias << ": value string = " << tokensStr[shift + iele];
          vect.emplace_back(tokensStr[shift + iele]);
        }
      } else {
        LOG(fatal) << "templated function called with wrong type, should be string";
      }
    }

    LOG(debug) << "timestamp = " << std::stof(tokensStr[shift + nele]);
    meas.emplace_back(std::stol(tokensStr[shift + nele]) * 1e6, vect); // measurement comes in seconds, we want it in microseconds
  }
}

} // namespace grp
} // namespace o2
#endif
