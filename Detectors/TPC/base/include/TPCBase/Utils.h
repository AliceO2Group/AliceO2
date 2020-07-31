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

#include "TPCBase/CalDet.h"

class TObjArray;
class TCanvas;
class TH1;

namespace o2
{
namespace tpc
{

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
void saveCanvases(std::vector<TCanvas*> canvases, std::string_view outDir, std::string_view types = "png,pdf", std::string_view rootFileName = "");
void saveCanvas(TCanvas& c, std::string_view outDir, std::string_view types);
std::vector<CalPad*> readCalPads(const std::string_view fileName, const std::vector<std::string>& calPadNames);
std::vector<CalPad*> readCalPads(const std::string_view fileName, const std::string_view calPadNames);

/// Merge cal pad objects from different files
///
/// Requires that all objects have the same name in the differnet files.
/// Objects are simply added.
/// \param outputFileName name of the output file
/// \param inputFileNames input file names. Perforams file system 'ls' in case the string includes '.root'. Otherwise it assumes a text input file with line by line file names.
/// \param calPadNames comma separated list of names of the CalPad objects as stored in the file.
void mergeCalPads(std::string_view outputFileName, std::string_view inputFileNames, std::string_view calPadNames);

} // namespace utils
} // namespace tpc
} // namespace o2

#endif
