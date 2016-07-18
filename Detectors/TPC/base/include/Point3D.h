#ifndef AliceO2_TPC_Point3D_H
#define AliceO2_TPC_Point3D_H

// TODO: For some reason the code does not compile if I inlude Point2D here
// #inclule "Point2D.h"

namespace AliceO2 {
namespace TPC {

template < class T >
class Point3D {
  public:
    Point3D() {}
    Point3D(const T &x, const T &y, const T&z) : mX(x), mY(y), mZ(z) {}
//     Point3D(const Point2D<T> &p, const T&z) : mX(p.getX()), mY(p.GetY()), mZ(z)) {}

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
