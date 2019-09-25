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

#include "DataFormatsTPC/Defs.h"

class TH1;
class TH2;

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
template <class T>
void draw(const CalDet<T>& calDet);

/// Drawing of a CalDet object
/// \param CalArray object to draw
template <class T>
void draw(const CalArray<T>& calArray);

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

} // namespace painter

} // namespace tpc

} // namespace o2

#endif // ALICEO2_TPC_PAINTER_H_
