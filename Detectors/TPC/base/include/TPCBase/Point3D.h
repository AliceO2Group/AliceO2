///
/// @file   Point3D.h
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
///

/// @brief  Simple templated 3D point
///
/// origin: TPC
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef AliceO2_TPC_Point3D_H
#define AliceO2_TPC_Point3D_H

namespace o2 {
namespace TPC {

template < class T >
class Point3D {
  public:
    Point3D() = default;
    Point3D(const T &x, const T &y, const T&z) : mX(x), mY(y), mZ(z) {}

    const T& getX() const { return mX; }
    const T& getY() const { return mY; }
    const T& getZ() const { return mZ; }

  private:
    T mX{};  /// x-position
    T mY{};  /// y-position
    T mZ{};  /// y-position
};

}
}

#endif
