#ifndef ALICEO2_TPC_PAINTER_H_
#define ALICEO2_TPC_PAINTER_H_

///
/// \file   Painter.h
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
///

namespace o2
{
namespace TPC 
{

template <class T>
class CalDet;

template <class T>
class CalArray;

/// \namespace Painter
/// \brief Drawing helper functions
///
/// In this namespace drawing function for calibration objects are implemented
///
/// origin: TPC
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

namespace Painter
{
  /// Drawing of a CalDet object
  /// \param CalDet object to draw
  template <class T>
  void Draw(CalDet<T> calDet);

  /// Drawing of a CalDet object
  /// \param CalArray object to draw
  template <class T>
  void Draw(CalArray<T> calArray);
} // namespace Painter

} // namespace TPC

} // namespace AliceO2

#endif // ALICEO2_TPC_PAINTER_H_
