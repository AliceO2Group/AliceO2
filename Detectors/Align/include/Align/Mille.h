#ifndef MILLE_H
#define MILLE_H

#include <fstream>
#include <TArrayI.h>
#include <TArrayF.h>

/**
 * \class Mille
 *
 *  Class to write a C binary (cf. below) file of a given name and to fill it
 *  with information used as input to **pede**.
 *  Use its member functions \c mille(), \c special(), \c kill() and \c end()
 *  as you would use the fortran \ref mille.f90 "MILLE"
 *  and its entry points \c MILLSP, \c KILLE and \c ENDLE.
 *
 *  For debugging purposes constructor flags enable switching to text output and/or
 *  to write also derivatives and labels which are ==0.
 *  But note that **pede** will not be able to read text output and has not been tested with
 *  derivatives/labels ==0.
 *
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.3 $
 *  $Date: 2007/04/16 17:47:38 $
 *  (last update by $Author: flucke $)
 */

/// Class to write C binary file.
class Mille 
{
 public:
  Mille(const char *outFileName, bool asBinary = true, bool writeZero = false);
  ~Mille();

  void mille(int NLC, const float *derLc, int NGL, const float *derGl,
	     const int *label, float rMeas, float sigma);
  void special(int nSpecial, const float *floatings, const int *integers);
  void kill();
  int  end();

 private:
  void newSet();
  bool checkBufferSize(int nLocal, int nGlobal);

  std::ofstream myOutFile; ///< C-binary for output
  bool myAsBinary;         ///< if false output as text
  bool myWriteZero;        ///< if true also write out derivatives/labels ==0
  /// buffer size for ints and floats
  int   myBufferSize;      ///< buffer size for ints and floats
  TArrayI myBufferInt;    ///< to collect labels etc.
  TArrayF myBufferFloat;  ///< to collect derivatives etc.
  int   myBufferPos;       ///< position in buffer
  bool  myHasSpecial;      ///< if true, special(..) already called for this record
  /// largest label allowed: 2^31 - 1
  enum {myMaxLabel = (0xFFFFFFFF - (1 << 31))};
};
#endif
