// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TPC_UTILS_H_
#define ALICEO2_TPC_UTILS_H_

///
/// \file   Utils.h
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
///

#include <vector>
#include <string_view>

class TObjArray;
class TCanvas;
class TH1;

namespace o2
{
namespace tpc
{

template <class T>
class CalDet;

template <class T>
class CalArray;

/// \namespace utils
/// \brief Common utility functions
///
/// Common utility functions for drawing, saving, ...
///
/// origin: TPC
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

namespace utils
{

const std::vector<std::string> tokenize(const std::string_view input, const std::string_view pattern);
TH1* getBinInfoXY(int& binx, int& biny, float& bincx, float& bincy);
void addFECInfo();
void saveCanvases(TObjArray& arr, std::string_view outDir, std::string_view types = "png,pdf", std::string_view rootFileName = "");
void saveCanvas(TCanvas& c, std::string_view outDir, std::string_view types);

} // namespace utils
} // namespace tpc
} // namespace o2

#endif
