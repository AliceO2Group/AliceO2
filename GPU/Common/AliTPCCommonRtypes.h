#ifndef ALITPCCOMMONRTYPES_H
#define ALITPCCOMMONRTYPES_H

#if defined(HLTCA_STANDALONE) || defined(__OPENCL__) || defined(HLTCA_GPULIBRARY)
  #if !defined(ROOT_Rtypes) && !defined(__CLING__)
    #define ClassDef(name,id)
    #define ClassDefNV(name, id)
    #define ClassDefOverride(name, id)
    #define ClassImp(name)
  #endif
#else
  #include "Rtypes.h"
#endif

#endif
