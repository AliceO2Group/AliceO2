#ifndef AliceO2_TPC_Point2D_H
#define AliceO2_TPC_Point2D_H

namespace AliceO2 {
namespace TPC {

template < class T >
class Point2D {
  public:
    //Point2D(unsigned char index, unsigned char connector, unsigned

  private:
    T mX{};  /// x-position
    T mY{};  /// y-position
};

}
}

#endif
