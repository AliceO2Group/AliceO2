// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TPC_PAINTER_H_
#define ALICEO2_TPC_PAINTER_H_

///
/// \file   Painter.h
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
///

#include <vector>
#include "DataFormatsTPC/Defs.h"

class TH1;
class TH2;
class TCanvas;

namespace o2
{
namespace tpc
{

template <class T>
class CalDet;

template <class T>
class CalArray;

/// \namespace painter
/// \brief Drawing helper functions
///
/// In this namespace drawing function for calibration objects are implemented
///
/// origin: TPC
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

namespace painter
{
//using T=float;
/// Drawing of a CalDet object
/// \param CalDet object to draw
/// \return TCanvas containing CalDet content
template <class T>
TCanvas* draw(const CalDet<T>& calDet, int nbins1D = 300, float xMin1D = 0, float xMax1D = 0);

/// Drawing of a CalDet object
/// \param CalArray object to draw
/// \return TCanvas containing CalArray content
template <class T>
TCanvas* draw(const CalArray<T>& calArray);

/// fill existing 2D histogram for CalDet object
/// \param h2D histogram to fill
/// \param CalDet object with data
/// \param side side which to get the histogram for
template <class T>
void fillHistogram2D(TH2& h2D, const CalDet<T>& calDet, Side side);

/// fill existing 2D histogram for CalArray object
/// \param h2D histogram to fill
/// \param CalArray object with data
template <class T>
void fillHistogram2D(TH2& h2D, const CalArray<T>& calArray);

/// get 2D histogram for CalDet object
/// \param CalDet object with data
/// \param side side which to get the histogram for
/// \return 2D histogram with data
template <class T>
TH2* getHistogram2D(const CalDet<T>& calDet, Side side);

/// get 2D histogram for CalArray object
/// \param CalDet object with data
/// \param side side which to get the histogram for
/// \return 2D histogram with data
template <class T>
TH2* getHistogram2D(const CalArray<T>& calArray);

/// Create summary canvases for a CalDet object
///
/// 1 Canvas with 2D and 1D distributions for each side
/// 1 Canvas with 2D distributions for all ROCs
/// 1 Canvas with 1D distributions for all ROCs
/// \param CalDet object to draw
/// \param nbins1D number of bins used for the 1D projections
/// \param xMin1D minimum value for 1D distribution (xMin = 0 and xMax = 0 for auto scaling)
/// \param xMax1D maximum value for 1D distribution (xMin = 0 and xMax = 0 for auto scaling)
/// \return TCanvas containing CalDet content
template <class T>
std::vector<TCanvas*> makeSummaryCanvases(const CalDet<T>& calDet, int nbins1D = 300, float xMin1D = 0, float xMax1D = 0, bool onlyFilled = true);

/// Create summary canvases for a CalDet object
///
/// 1 Canvas with 2D and 1D distributions for each side
/// 1 Canvas with 2D distributions for all ROCs
/// 1 Canvas with 1D distributions for all ROCs
/// \param CalDet object to draw
/// \param nbins1D number of bins used for the 1D projections
/// \param xMin1D minimum value for 1D distribution (xMin = 0 and xMax = 0 for auto scaling)
/// \param xMax1D maximum value for 1D distribution (xMin = 0 and xMax = 0 for auto scaling)
/// \param fileName input file name
/// \param calPadNames comma separated list of names of the CalPad objects as stored in the file.
/// \return TCanvas containing CalDet content
std::vector<TCanvas*> makeSummaryCanvases(const std::string_view fileName, const std::string_view calPadNames, int nbins1D = 300, float xMin1D = 0, float xMax1D = 0, bool onlyFilled = true);

} // namespace painter

} // namespace tpc

} // namespace o2

#endif // ALICEO2_TPC_PAINTER_H_
