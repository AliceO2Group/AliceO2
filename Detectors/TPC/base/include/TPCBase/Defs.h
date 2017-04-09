///
/// @file   Defs.h
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
///

/// @brief  Global TPC definitions and constants

#ifndef AliceO2_TPC_Defs_H
#define AliceO2_TPC_Defs_H

#include <cmath>

#include "Point2D.h"
#include "Point3D.h"

namespace o2 {
namespace TPC {

/// TPC readout sidE
enum Side {A=0, C=1};
//   enum class Side {A=0, C=1};
//  Problem with root cint. does not seem to support enum class ...
constexpr unsigned char SECTORSPERSIDE=18;
constexpr unsigned char SIDES=2;

constexpr double PI          = 3.14159265358979323846;
constexpr double TWOPI       = 2.*PI;
constexpr double SECPHIWIDTH = TWOPI/18.;


/// TPC ROC types
enum RocType {IROC=0, OROC=1};
// enum class RocType {IROC=0, OROC=1};

/// TPC GEM stack types
enum GEMstack {IROCgem=0, OROC1gem=1, OROC2gem=2, OROC3gem=3};

/// Definition of the different pad subsets
enum class PadSubset : char {
  ROC,        ///< ROCs (up to 72)
  Partition,  ///< Partitions (up to 36*5)
  Region      ///< Regions (up to 36*10)
};
/// Pad centres as 2D float
typedef Point2D<float> PadCentre;
typedef Point2D<float> GlobalPosition2D;
typedef Point2D<float> LocalPosition2D;
typedef Point3D<float> GlobalPosition3D;
typedef Point3D<float> LocalPosition3D;

/// global pad number
typedef unsigned short GlobalPadNumber;

// GlobalPosition3D LocalToGlobal(const LocalPosition3D pos, const float alpha)
// {
//   const double cs=cos(alpha), sn=sin(alpha);
//   return GlobalPosition3D(pos.getX()*cs-pos.getY()*sn,pos.getX()*sn+pos.getY()*cs,pos.getZ());
// }

// LocalPosition3D GlobalToLocal(const GlobalPosition3D& pos, const float alpha)
// {
//   const double cs=cos(-alpha), sn=sin(-alpha);
//   return LocalPosition3D(pos.getX()*cs-pos.getY()*sn,pos.getX()*sn+pos.getY()*cs,pos.getZ());
// }


/**
 * simple class to allow for range for loops over enums
 * e.g. for (auto &side : Enum<Sides>() ) { cout << side << endl; }
 * taken from http://stackoverflow.com/questions/8498300/allow-for-range-based-for-with-enum-classes
 */

template< typename T >
class Enum
{
  public:
    class Iterator
    {
      public:
        Iterator( int value ) :
          m_value( value )
      { }

        T operator*( ) const
        {
          return (T)m_value;
        }

        void operator++( )
        {
          ++m_value;
        }

        bool operator!=( Iterator rhs )
        {
          return m_value != rhs.m_value;
        }

      private:
        int m_value;
    };

};

  template< typename T >
typename Enum<T>::Iterator begin( Enum<T> )
{
  return typename Enum<T>::Iterator( (int)T::First );
}

  template< typename T >
typename Enum<T>::Iterator end( Enum<T> )
{
  return typename Enum<T>::Iterator( ((int)T::Last) + 1 );
}

}
}


#endif
