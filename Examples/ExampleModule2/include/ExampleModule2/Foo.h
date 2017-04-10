///
/// @file    Foo.h
/// @author  Barthelemy von Haller
///

#ifndef ALICE_O2_EXAMPLEMODULE2_FOO_H
#define ALICE_O2_EXAMPLEMODULE2_FOO_H

#include "Rtypes.h"   // for ClassDef

/// @brief    Here you put a short description of the namespace
/// Extended documentation for this namespace
/// @author  	Barthelemy von Haller
namespace o2 {
namespace Examples {
namespace ExampleModule2 {

/// @brief   Here you put a short description of the class
/// Extended documentation for this class.
/// @author 	Barthelemy von Haller
class Foo
{
  public:

    /// @brief   Greets the caller
    /// @author 	Barthelemy von Haller
    /// @brief	Simple hello world
    void greet();

    /// @brief   Returns the value passed to it
    /// Longer description that is useless here.
    /// @author 	Barthelemy von Haller
    /// @param n (In) input number.
    /// @return Returns the input number given.
    int returnsN(int n);

  ClassDefNV(Foo, 1)
};

} // namespace ExampleModule2
} // namespace Examples
} // namespace AliceO2

#endif // ALICE_O2_EXAMPLEMODULE2_FOO_H
