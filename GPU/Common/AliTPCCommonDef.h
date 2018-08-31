#ifndef ALITPCCOMMONDEF_H
#define ALITPCCOMMONDEF_H

#if defined(__CINT__) || defined(__ROOTCINT__)
#define CON_DELETE
#define CON_DEFAULT
#else
#define CON_DELETE = delete
#define CON_DEFAULT = default
#endif

#endif
