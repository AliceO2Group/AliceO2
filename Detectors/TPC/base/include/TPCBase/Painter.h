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
class TH2Poly;
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

/// pad corner coordinates
struct PadCoordinates {
  std::array<double, 4> xVals;
  std::array<double, 4> yVals;

  void rotate(float angDeg)
  {
    const auto ang = 0.017453292519943295 * angDeg;
    const auto cs = std::cos(ang);
    const auto sn = std::sin(ang);
    for (int i = 0; i < xVals.size(); ++i) {
      const auto x = xVals[i] * cs - yVals[i] * sn;
      const auto y = xVals[i] * sn + yVals[i] * cs;
      xVals[i] = x;
      yVals[i] = y;
    }
  }
};

/// create a vector of pad corner coordinate for one full sector
std::vector<PadCoordinates> getPadCoordinatesSector();

//using T=float;
/// Drawing of a CalDet object
/// \param CalDet object to draw
/// \return TCanvas containing CalDet content
template <class T>
TCanvas* draw(const CalDet<T>& calDet, int nbins1D = 300, float xMin1D = 0, float xMax1D = 0, TCanvas* outputCanvas = nullptr);

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

/// make a sector-wise histogram with correct pad corners
TH2Poly* makeSectorHist(const std::string_view name = "hSector", const std::string_view title = "Sector;local #it{x} (cm);local #it{y} (cm)");

/// make a side-wise histogram with correct pad corners
TH2Poly* makeSideHist(Side side);

/// fill existing TH2Poly histogram for CalDet object
/// \param h2D histogram to fill
/// \param CalDet object with data
/// \param side side which to get the histogram for
template <class T>
void fillPoly2D(TH2Poly& h2D, const CalDet<T>& calDet, Side side);

/// Create summary canvases for a CalDet object
///
/// 1 Canvas with 2D and 1D distributions for each side
/// 1 Canvas with 2D distributions for all ROCs
/// 1 Canvas with 1D distributions for all ROCs
/// \param CalDet object to draw
/// \param nbins1D number of bins used for the 1D projections
/// \param xMin1D minimum value for 1D distribution (xMin = 0 and xMax = 0 for auto scaling)
/// \param xMax1D maximum value for 1D distribution (xMin = 0 and xMax = 0 for auto scaling)
/// \param outputCanvases if outputCanvases are given, use them instead of creating new ones, 3 are required
/// \return TCanvas containing CalDet content
template <class T>
std::vector<TCanvas*> makeSummaryCanvases(const CalDet<T>& calDet, int nbins1D = 300, float xMin1D = 0, float xMax1D = 0, bool onlyFilled = true, std::vector<TCanvas*>* outputCanvases = nullptr);

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

/// draw sector boundaris, side name and sector numbers
void drawSectorsXY(Side side, int sectorLineColor = 920, int sectorTextColor = 1);

} // namespace painter

} // namespace tpc

} // namespace o2

#endif // ALICEO2_TPC_PAINTER_H_
