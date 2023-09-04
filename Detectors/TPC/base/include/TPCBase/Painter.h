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

#ifndef ALICEO2_TPC_PAINTER_H_
#define ALICEO2_TPC_PAINTER_H_

///
/// \file   Painter.h
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
///

#include <vector>
#include <string>
#include <string_view>
#include "DataFormatsTPC/Defs.h"
#include "DataFormatsTPC/LtrCalibData.h"

class TH1;
class TH2;
class TH3F;
class TH2Poly;
class TCanvas;
class TMultiGraph;
class TTree;

namespace o2::tpc
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

struct painter {

  enum class Type : int {
    Pad,   ///< drawing pads
    Stack, ///< drawing stacks
    FEC    ///< drawing of FECs
  };

  static std::array<int, 6> colors;
  static std::array<int, 10> markers;

  /// pad corner coordinates
  struct PadCoordinates {
    std::vector<double> xVals = std::vector<double>(4);
    std::vector<double> yVals = std::vector<double>(4);

    void rotate(float angDeg)
    {
      const auto ang = 0.017453292519943295 * angDeg;
      const auto cs = std::cos(ang);
      const auto sn = std::sin(ang);
      for (size_t i = 0; i < xVals.size(); ++i) {
        const auto x = xVals[i] * cs - yVals[i] * sn;
        const auto y = xVals[i] * sn + yVals[i] * cs;
        xVals[i] = x;
        yVals[i] = y;
      }
    }
  };

  /// create a vector of pad corner coordinate for one full sector
  static std::vector<PadCoordinates> getPadCoordinatesSector();

  /// create a vector of stack corner coordinate for one full sector
  static std::vector<PadCoordinates> getStackCoordinatesSector();

  /// create a vector of FEC corner coordinates for one full sector
  static std::vector<PadCoordinates> getFECCoordinatesSector();

  /// \return returns coordinates for given type
  static std::vector<o2::tpc::painter::PadCoordinates> getCoordinates(const Type type);

  /// binning vector with radial pad-row positions (in cm)
  /// \param roc roc number (0-35 IROC, 36-71 OROC, >=72 full sector)
  static std::vector<double> getRowBinningCM(uint32_t roc = 72);

  /// ROC title from ROC number
  static std::string getROCTitle(const int rocNumber);

  // using T=float;
  /// Drawing of a CalDet object
  /// \param CalDet object to draw
  /// \return TCanvas containing CalDet content
  template <class T>
  static TCanvas* draw(const CalDet<T>& calDet, int nbins1D = 300, float xMin1D = 0, float xMax1D = 0, TCanvas* outputCanvas = nullptr);

  /// Drawing of a CalDet object
  /// \param CalArray object to draw
  /// \return TCanvas containing CalArray content
  template <class T>
  static TCanvas* draw(const CalArray<T>& calArray);

  /// fill existing 2D histogram for CalDet object
  /// \param h2D histogram to fill
  /// \param CalDet object with data
  /// \param side side which to get the histogram for
  template <class T>
  static void fillHistogram2D(TH2& h2D, const CalDet<T>& calDet, Side side);

  /// fill existing 2D histogram for CalArray object
  /// \param h2D histogram to fill
  /// \param CalArray object with data
  template <class T>
  static void fillHistogram2D(TH2& h2D, const CalArray<T>& calArray);

  /// get 2D histogram for CalDet object
  /// \param CalDet object with data
  /// \param side side which to get the histogram for
  /// \return 2D histogram with data
  template <class T>
  static TH2* getHistogram2D(const CalDet<T>& calDet, Side side);

  /// get 2D histogram for CalArray object
  /// \param CalDet object with data
  /// \param side side which to get the histogram for
  /// \return 2D histogram with data
  template <class T>
  static TH2* getHistogram2D(const CalArray<T>& calArray);

  /// make a sector-wise histogram with correct pad corners
  /// \param xMin minimum x coordinate of the histogram
  /// \param xMax maximum x coordinate of the histogram
  /// \param yMin minimum y coordinate of the histogram
  /// \param yMax maximum y coordinate of the histogram
  /// \param type granularity of the histogram (per pad or per stack)
  static TH2Poly* makeSectorHist(const std::string_view name = "hSector", const std::string_view title = "Sector;local #it{x} (cm);local #it{y} (cm)", const float xMin = 83.65f, const float xMax = 247.7f, const float yMin = -43.7f, const float yMax = 43.7f, const Type type = Type::Pad);

  /// make a side-wise histogram with correct pad corners
  /// \param type granularity of the histogram (per pad or per stack)
  static TH2Poly* makeSideHist(Side side, const Type type = Type::Pad);

  /// fill existing TH2Poly histogram for CalDet object
  /// \param h2D histogram to fill
  /// \param CalDet object with data
  /// \param side side which to get the histogram for
  template <class T>
  static void fillPoly2D(TH2Poly& h2D, const CalDet<T>& calDet, Side side);

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
  static std::vector<TCanvas*> makeSummaryCanvases(const CalDet<T>& calDet, int nbins1D = 300, float xMin1D = 0, float xMax1D = 0, bool onlyFilled = true, std::vector<TCanvas*>* outputCanvases = nullptr);

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
  static std::vector<TCanvas*> makeSummaryCanvases(const std::string_view fileName, const std::string_view calPadNames, int nbins1D = 300, float xMin1D = 0, float xMax1D = 0, bool onlyFilled = true);

  /// draw sector boundaris, side name and sector numbers
  static void drawSectorsXY(Side side, int sectorLineColor = 920, int sectorTextColor = 1);

  /// draw information of the sector: pad number in row
  /// \param padTextColor text color of pad number
  /// \param lineScalePS setting the width of the lines of the pads which are drawn
  static void drawSectorLocalPadNumberPoly(short padTextColor = kBlack, float lineScalePS = 1);

  /// draw information of the sector: pad row in region, global pad row, lines for separating the regions
  /// \param regionLineColor color of the line which is drawn at the start of a sector
  /// \param rowTextColor color of the text which is drawn
  static void drawSectorInformationPoly(short regionLineColor = kRed, short rowTextColor = kRed);

  /// convert std::vector<CalDet> objects (pad granularity) to a 3D-histogram with rxphixz binning. Each CalDet will be filled in a unique slice in the histogram
  /// \param calDet input objects which will be converted to a 3D histogram
  /// \param norm whether to normalize the histogram (weighted mean) or to just integrate the values of the CalDet
  /// \param nRBins number of bins in r direction of the output histogram
  /// \param rMin min r value of the output histogram
  /// \param rMax max r value of the output histogram
  /// \param nPhiBins number of bins in phi direction of the output histogram
  /// \param zMax z range of the output histogram (-zMax to zMax)
  template <typename DataT>
  static TH3F convertCalDetToTH3(const std::vector<CalDet<DataT>>& calDet, const bool norm = true, const int nRBins = 150, const float rMin = 83.5, const float rMax = 254.5, const int nPhiBins = 720, const float zMax = 1);

  /// make summary canvases for laser calibration data
  static std::vector<TCanvas*> makeSummaryCanvases(const LtrCalibData& ltr, std::vector<TCanvas*>* outputCanvases = nullptr);

  /// make a canvas for junk detection data
  static TCanvas* makeJunkDetectionCanvas(const TObjArray* data, TCanvas* outputCanvas = nullptr);

  /// Adjust the X2 position and the tick length of the color axis
  /// \param h histogram to get z axis
  /// \param x2ndc new x2 value of color axis in NDC
  /// \param tickLength tick length of the color axis
  static void adjustPalette(TH1* h, float x2ndc, float tickLength = 0.015);

  /// Make a multi graph with sparse option
  ///
  /// Produce a multi graph from a tree, where `varX` is a common x variable, e.g. a run number.
  /// TTree::Draw is used to fill the variables, so all expressions supported can be used.
  /// As in TTree::Draw, varsY have to be seperated by a ':'. For each variable a Graph will be created.
  /// Optionally one can give 'errVarsY', which must have the same number of entries as 'varsY'.
  /// In case the 'sparse' option is set (default), the x-axis will have as many bins as `varX` has entries. The bins will be labelled accordingly.
  /// Colors and marker are assigned automatically using the ones defined in 'colors' and 'markers'
  ///
  /// \param tree the input tree to use
  /// \param varX the x variable to create the sparse binning from
  /// \param varsY the y variable for ech of which to create a graph, separated by ':'
  /// \param errVarsY the y variable errors, separated by ':', same number of variables as in varsY required
  /// \param cuts cut string to be used in the TTree::Draw
  /// \param makeSparse make a sparse graph (default)
  static TMultiGraph* makeMultiGraph(TTree& tree, std::string_view varX, std::string_view varsY, std::string_view errVarsY = "", std::string_view cut = "", bool makeSparse = true);

}; // struct painter

} // namespace o2::tpc

#endif // ALICEO2_TPC_PAINTER_H_
