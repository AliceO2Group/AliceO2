#ifndef AliceO2_TPC_Defs_H
#define AliceO2_TPC_Defs_H

#include "Point2D.h"

namespace AliceO2 {
namespace TPC {

/// TPC readout sidE
enum Side {A=0, C=1};
//   enum class Side {A=0, C=1};
//  Problem with root cint. does not seem to support enum class ...

/// TPC ROC types
enum RocType {IROC=0, OROC=1};
// enum class RocType {IROC=0, OROC=1};


/// Pad centres as 2D float
typedef Point2D<float> PadCentre;

/// global pad number
typedef unsigned short GlobalPadNumber;

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

        T operator*( void ) const
        {
          return (T)m_value;
        }

        void operator++( void )
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
