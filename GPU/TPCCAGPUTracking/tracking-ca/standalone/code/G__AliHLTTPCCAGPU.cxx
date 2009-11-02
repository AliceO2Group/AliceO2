//
// File generated by rootcint at Mon Nov  2 04:04:41 2009

// Do NOT change. Changes will be lost next time file is generated
//

#include "RConfig.h" //rootcint 4834
#if !defined(R__ACCESS_IN_SYMBOL)
//Break the privacy of classes -- Disabled for the moment
#define private public
#define protected public
#endif

// Since CINT ignores the std namespace, we need to do so in this file.
namespace std {} using namespace std;
#include "G__AliHLTTPCCAGPU.h"

#include "TClass.h"
#include "TBuffer.h"
#include "TMemberInspector.h"
#include "TError.h"

#ifndef G__ROOT
#define G__ROOT
#endif

#include "RtypesImp.h"
#include "TIsAProxy.h"

// START OF SHADOWS

namespace ROOT {
   namespace Shadow {
   } // of namespace Shadow
} // of namespace ROOT
// END OF SHADOWS

namespace ROOT {
   void AliHLTTPCCAGPUTrackerNVCC_ShowMembers(void *obj, TMemberInspector &R__insp, char *R__parent);
   static void *new_AliHLTTPCCAGPUTrackerNVCC(void *p = 0);
   static void *newArray_AliHLTTPCCAGPUTrackerNVCC(Long_t size, void *p);
   static void delete_AliHLTTPCCAGPUTrackerNVCC(void *p);
   static void deleteArray_AliHLTTPCCAGPUTrackerNVCC(void *p);
   static void destruct_AliHLTTPCCAGPUTrackerNVCC(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::AliHLTTPCCAGPUTrackerNVCC*)
   {
      ::AliHLTTPCCAGPUTrackerNVCC *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::AliHLTTPCCAGPUTrackerNVCC >(0);
      static ::ROOT::TGenericClassInfo 
         instance("AliHLTTPCCAGPUTrackerNVCC", ::AliHLTTPCCAGPUTrackerNVCC::Class_Version(), "HLT/TPCLib/tracking-ca/AliHLTTPCCAGPUTrackerNVCC.h", 20,
                  typeid(::AliHLTTPCCAGPUTrackerNVCC), DefineBehavior(ptr, ptr),
                  &::AliHLTTPCCAGPUTrackerNVCC::Dictionary, isa_proxy, 4,
                  sizeof(::AliHLTTPCCAGPUTrackerNVCC) );
      instance.SetNew(&new_AliHLTTPCCAGPUTrackerNVCC);
      instance.SetNewArray(&newArray_AliHLTTPCCAGPUTrackerNVCC);
      instance.SetDelete(&delete_AliHLTTPCCAGPUTrackerNVCC);
      instance.SetDeleteArray(&deleteArray_AliHLTTPCCAGPUTrackerNVCC);
      instance.SetDestructor(&destruct_AliHLTTPCCAGPUTrackerNVCC);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::AliHLTTPCCAGPUTrackerNVCC*)
   {
      return GenerateInitInstanceLocal((::AliHLTTPCCAGPUTrackerNVCC*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstanceLocal((const ::AliHLTTPCCAGPUTrackerNVCC*)0x0); R__UseDummy(_R__UNIQUE_(Init));
} // end of namespace ROOT

//______________________________________________________________________________
TClass *AliHLTTPCCAGPUTrackerNVCC::fgIsA = 0;  // static to hold class pointer

//______________________________________________________________________________
const char *AliHLTTPCCAGPUTrackerNVCC::Class_Name()
{
   return "AliHLTTPCCAGPUTrackerNVCC";
}

//______________________________________________________________________________
const char *AliHLTTPCCAGPUTrackerNVCC::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::AliHLTTPCCAGPUTrackerNVCC*)0x0)->GetImplFileName();
}

//______________________________________________________________________________
int AliHLTTPCCAGPUTrackerNVCC::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::AliHLTTPCCAGPUTrackerNVCC*)0x0)->GetImplFileLine();
}

//______________________________________________________________________________
void AliHLTTPCCAGPUTrackerNVCC::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::AliHLTTPCCAGPUTrackerNVCC*)0x0)->GetClass();
}

//______________________________________________________________________________
TClass *AliHLTTPCCAGPUTrackerNVCC::Class()
{
   if (!fgIsA) fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::AliHLTTPCCAGPUTrackerNVCC*)0x0)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
void AliHLTTPCCAGPUTrackerNVCC::Streamer(TBuffer &R__b)
{
   // Stream an object of class AliHLTTPCCAGPUTrackerNVCC.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(AliHLTTPCCAGPUTrackerNVCC::Class(),this);
   } else {
      R__b.WriteClassBuffer(AliHLTTPCCAGPUTrackerNVCC::Class(),this);
   }
}

//______________________________________________________________________________
void AliHLTTPCCAGPUTrackerNVCC::ShowMembers(TMemberInspector &R__insp, char *R__parent)
{
      // Inspect the data members of an object of class AliHLTTPCCAGPUTrackerNVCC.
      TClass *R__cl = ::AliHLTTPCCAGPUTrackerNVCC::IsA();
      Int_t R__ncp = strlen(R__parent);
      if (R__ncp || R__cl || R__insp.IsA()) { }
      R__insp.Inspect(R__cl, R__parent, "*fGpuTracker", &fGpuTracker);
      R__insp.Inspect(R__cl, R__parent, "*fGPUMemory", &fGPUMemory);
      R__insp.Inspect(R__cl, R__parent, "*fHostLockedMemory", &fHostLockedMemory);
      R__insp.Inspect(R__cl, R__parent, "fDebugLevel", &fDebugLevel);
      R__insp.Inspect(R__cl, R__parent, "*fOutFile", &fOutFile);
      R__insp.Inspect(R__cl, R__parent, "fGPUMemSize", &fGPUMemSize);
      R__insp.Inspect(R__cl, R__parent, "*fpCudaStreams", &fpCudaStreams);
      R__insp.Inspect(R__cl, R__parent, "fSliceCount", &fSliceCount);
      R__insp.Inspect(R__cl, R__parent, "fSlaveTrackers[36]", fSlaveTrackers);
      R__insp.Inspect(R__cl, R__parent, "*fOutputControl", &fOutputControl);
      R__insp.Inspect(R__cl, R__parent, "fThreadId", &fThreadId);
      R__insp.Inspect(R__cl, R__parent, "fCudaInitialized", &fCudaInitialized);
      ::ROOT::GenericShowMembers("AliHLTTPCCAGPUTracker", ( ::AliHLTTPCCAGPUTracker *) (this ), R__insp, R__parent, false);
      AliHLTLogging::ShowMembers(R__insp, R__parent);
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_AliHLTTPCCAGPUTrackerNVCC(void *p) {
      return  p ? ::new((::ROOT::TOperatorNewHelper*)p) ::AliHLTTPCCAGPUTrackerNVCC : new ::AliHLTTPCCAGPUTrackerNVCC;
   }
   static void *newArray_AliHLTTPCCAGPUTrackerNVCC(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::TOperatorNewHelper*)p) ::AliHLTTPCCAGPUTrackerNVCC[nElements] : new ::AliHLTTPCCAGPUTrackerNVCC[nElements];
   }
   // Wrapper around operator delete
   static void delete_AliHLTTPCCAGPUTrackerNVCC(void *p) {
      delete ((::AliHLTTPCCAGPUTrackerNVCC*)p);
   }
   static void deleteArray_AliHLTTPCCAGPUTrackerNVCC(void *p) {
      delete [] ((::AliHLTTPCCAGPUTrackerNVCC*)p);
   }
   static void destruct_AliHLTTPCCAGPUTrackerNVCC(void *p) {
      typedef ::AliHLTTPCCAGPUTrackerNVCC current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::AliHLTTPCCAGPUTrackerNVCC

/********************************************************
* HLT/tgt_linuxx8664gcc/G__AliHLTTPCCAGPU.cxx
* CAUTION: DON'T CHANGE THIS FILE. THIS FILE IS AUTOMATICALLY GENERATED
*          FROM HEADER FILES LISTED IN G__setup_cpp_environmentXXX().
*          CHANGE THOSE HEADER FILES AND REGENERATE THIS FILE.
********************************************************/

#ifdef G__MEMTEST
#undef malloc
#undef free
#endif

#if defined(__GNUC__) && (__GNUC__ > 3) && (__GNUC_MINOR__ > 1)
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

extern "C" void G__cpp_reset_tagtableG__AliHLTTPCCAGPU();

extern "C" void G__set_cpp_environmentG__AliHLTTPCCAGPU() {
  G__add_compiledheader("TObject.h");
  G__add_compiledheader("TMemberInspector.h");
  G__add_compiledheader("HLT/TPCLib/tracking-ca/AliHLTTPCCAGPUTrackerNVCC.h");
  G__cpp_reset_tagtableG__AliHLTTPCCAGPU();
}
#include <new>
extern "C" int G__cpp_dllrevG__AliHLTTPCCAGPU() { return(30051515); }

/*********************************************************
* Member function Interface Method
*********************************************************/

/* AliHLTTPCCAGPUTrackerNVCC */
static int G__G__AliHLTTPCCAGPU_229_0_1(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)
{
   AliHLTTPCCAGPUTrackerNVCC* p = NULL;
   char* gvp = (char*) G__getgvp();
   int n = G__getaryconstruct();
   if (n) {
     if ((gvp == (char*)G__PVOID) || (gvp == 0)) {
       p = new AliHLTTPCCAGPUTrackerNVCC[n];
     } else {
       p = new((void*) gvp) AliHLTTPCCAGPUTrackerNVCC[n];
     }
   } else {
     if ((gvp == (char*)G__PVOID) || (gvp == 0)) {
       p = new AliHLTTPCCAGPUTrackerNVCC;
     } else {
       p = new((void*) gvp) AliHLTTPCCAGPUTrackerNVCC;
     }
   }
   result7->obj.i = (long) p;
   result7->ref = (long) p;
   G__set_tagnum(result7,G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCAGPUTrackerNVCC));
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__AliHLTTPCCAGPU_229_0_26(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)
{
      G__letint(result7, 85, (long) AliHLTTPCCAGPUTrackerNVCC::Class());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__AliHLTTPCCAGPU_229_0_27(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)
{
      G__letint(result7, 67, (long) AliHLTTPCCAGPUTrackerNVCC::Class_Name());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__AliHLTTPCCAGPU_229_0_28(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)
{
      G__letint(result7, 115, (long) AliHLTTPCCAGPUTrackerNVCC::Class_Version());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__AliHLTTPCCAGPU_229_0_29(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)
{
      AliHLTTPCCAGPUTrackerNVCC::Dictionary();
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__AliHLTTPCCAGPU_229_0_33(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)
{
      ((AliHLTTPCCAGPUTrackerNVCC*) G__getstructoffset())->StreamerNVirtual(*(TBuffer*) libp->para[0].ref);
      G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__AliHLTTPCCAGPU_229_0_34(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)
{
      G__letint(result7, 67, (long) AliHLTTPCCAGPUTrackerNVCC::DeclFileName());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__AliHLTTPCCAGPU_229_0_35(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)
{
      G__letint(result7, 105, (long) AliHLTTPCCAGPUTrackerNVCC::ImplFileLine());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__AliHLTTPCCAGPU_229_0_36(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)
{
      G__letint(result7, 67, (long) AliHLTTPCCAGPUTrackerNVCC::ImplFileName());
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__G__AliHLTTPCCAGPU_229_0_37(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)
{
      G__letint(result7, 105, (long) AliHLTTPCCAGPUTrackerNVCC::DeclFileLine());
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
typedef AliHLTTPCCAGPUTrackerNVCC G__TAliHLTTPCCAGPUTrackerNVCC;
static int G__G__AliHLTTPCCAGPU_229_0_38(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)
{
   char* gvp = (char*) G__getgvp();
   long soff = G__getstructoffset();
   int n = G__getaryconstruct();
   //
   //has_a_delete: 0
   //has_own_delete1arg: 0
   //has_own_delete2arg: 0
   //
   if (!soff) {
     return(1);
   }
   if (n) {
     if (gvp == (char*)G__PVOID) {
       delete[] (AliHLTTPCCAGPUTrackerNVCC*) soff;
     } else {
       G__setgvp((long) G__PVOID);
       for (int i = n - 1; i >= 0; --i) {
         ((AliHLTTPCCAGPUTrackerNVCC*) (soff+(sizeof(AliHLTTPCCAGPUTrackerNVCC)*i)))->~G__TAliHLTTPCCAGPUTrackerNVCC();
       }
       G__setgvp((long)gvp);
     }
   } else {
     if (gvp == (char*)G__PVOID) {
       delete (AliHLTTPCCAGPUTrackerNVCC*) soff;
     } else {
       G__setgvp((long) G__PVOID);
       ((AliHLTTPCCAGPUTrackerNVCC*) (soff))->~G__TAliHLTTPCCAGPUTrackerNVCC();
       G__setgvp((long)gvp);
     }
   }
   G__setnull(result7);
   return(1 || funcname || hash || result7 || libp) ;
}


/* Setting up global function */

/*********************************************************
* Member function Stub
*********************************************************/

/* AliHLTTPCCAGPUTrackerNVCC */

/*********************************************************
* Global function Stub
*********************************************************/

/*********************************************************
* Get size of pointer to member function
*********************************************************/
class G__Sizep2memfuncG__AliHLTTPCCAGPU {
 public:
  G__Sizep2memfuncG__AliHLTTPCCAGPU(): p(&G__Sizep2memfuncG__AliHLTTPCCAGPU::sizep2memfunc) {}
    size_t sizep2memfunc() { return(sizeof(p)); }
  private:
    size_t (G__Sizep2memfuncG__AliHLTTPCCAGPU::*p)();
};

size_t G__get_sizep2memfuncG__AliHLTTPCCAGPU()
{
  G__Sizep2memfuncG__AliHLTTPCCAGPU a;
  G__setsizep2memfunc((int)a.sizep2memfunc());
  return((size_t)a.sizep2memfunc());
}


/*********************************************************
* virtual base class offset calculation interface
*********************************************************/

   /* Setting up class inheritance */

/*********************************************************
* Inheritance information setup/
*********************************************************/
extern "C" void G__cpp_setup_inheritanceG__AliHLTTPCCAGPU() {

   /* Setting up class inheritance */
   if(0==G__getnumbaseclass(G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCAGPUTrackerNVCC))) {
     AliHLTTPCCAGPUTrackerNVCC *G__Lderived;
     G__Lderived=(AliHLTTPCCAGPUTrackerNVCC*)0x1000;
     {
       AliHLTTPCCAGPUTracker *G__Lpbase=(AliHLTTPCCAGPUTracker*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCAGPUTrackerNVCC),G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCAGPUTracker),(long)G__Lpbase-(long)G__Lderived,1,1);
     }
     {
       AliHLTLogging *G__Lpbase=(AliHLTLogging*)G__Lderived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCAGPUTrackerNVCC),G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_AliHLTLogging),(long)G__Lpbase-(long)G__Lderived,1,1);
     }
   }
}

/*********************************************************
* typedef information setup/
*********************************************************/
extern "C" void G__cpp_setup_typetableG__AliHLTTPCCAGPU() {

   /* Setting up typedef entry */
   G__search_typename2("Version_t",115,-1,0,-1);
   G__setnewtype(-1,"Class version identifier (short)",0);
   G__search_typename2("vector<TSchemaHelper>",117,G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_vectorlEROOTcLcLTSchemaHelpercOallocatorlEROOTcLcLTSchemaHelpergRsPgR),0,-1);
   G__setnewtype(-1,NULL,0);
   G__search_typename2("reverse_iterator<const_iterator>",117,G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_reverse_iteratorlEvectorlEROOTcLcLTSchemaHelpercOallocatorlEROOTcLcLTSchemaHelpergRsPgRcLcLiteratorgR),0,G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_vectorlEROOTcLcLTSchemaHelpercOallocatorlEROOTcLcLTSchemaHelpergRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("reverse_iterator<iterator>",117,G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_reverse_iteratorlEvectorlEROOTcLcLTSchemaHelpercOallocatorlEROOTcLcLTSchemaHelpergRsPgRcLcLiteratorgR),0,G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_vectorlEROOTcLcLTSchemaHelpercOallocatorlEROOTcLcLTSchemaHelpergRsPgR));
   G__setnewtype(-1,NULL,0);
   G__search_typename2("vector<ROOT::TSchemaHelper>",117,G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_vectorlEROOTcLcLTSchemaHelpercOallocatorlEROOTcLcLTSchemaHelpergRsPgR),0,-1);
   G__setnewtype(-1,NULL,0);
}

/*********************************************************
* Data Member information setup/
*********************************************************/

   /* Setting up class,struct,union tag member variable */

   /* AliHLTTPCCAGPUTrackerNVCC */
static void G__setup_memvarAliHLTTPCCAGPUTrackerNVCC(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCAGPUTrackerNVCC));
   { AliHLTTPCCAGPUTrackerNVCC *p; p=(AliHLTTPCCAGPUTrackerNVCC*)0x1000; if (p) { }
   G__memvar_setup((void*)0,85,0,0,G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCATracker),-1,-1,4,"fGpuTracker=",0,(char*)NULL);
   G__memvar_setup((void*)0,89,0,0,-1,-1,-1,4,"fGPUMemory=",0,(char*)NULL);
   G__memvar_setup((void*)0,89,0,0,-1,-1,-1,4,"fHostLockedMemory=",0,(char*)NULL);
   G__memvar_setup((void*)0,105,0,0,-1,-1,-1,4,"fDebugLevel=",0,"Debug Level for GPU Tracker");
   G__memvar_setup((void*)0,85,0,0,G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_basic_ostreamlEcharcOchar_traitslEchargRsPgR),G__defined_typename("ostream"),-1,4,"fOutFile=",0,"Debug Output Stream Pointer");
   G__memvar_setup((void*)0,109,0,0,-1,-1,-1,4,"fGPUMemSize=",0,"Memory Size to allocate on GPU");
   G__memvar_setup((void*)0,89,0,0,-1,-1,-1,4,"fpCudaStreams=",0,(char*)NULL);
   G__memvar_setup((void*)0,105,0,0,-1,-1,-1,4,"fSliceCount=",0,(char*)NULL);
   G__memvar_setup((void*)0,105,0,1,-1,-1,-2,4,"fgkNSlices=",0,(char*)NULL);
   G__memvar_setup((void*)0,117,0,0,G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCATracker),-1,-1,4,"fSlaveTrackers[36]=",0,(char*)NULL);
   G__memvar_setup((void*)0,85,0,0,G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCASliceOutputcLcLoutputControlStruct),-1,-1,4,"fOutputControl=",0,(char*)NULL);
   G__memvar_setup((void*)0,103,0,0,-1,-1,-2,4,"fgGPUUsed=",0,(char*)NULL);
   G__memvar_setup((void*)0,105,0,0,-1,-1,-1,4,"fThreadId=",0,(char*)NULL);
   G__memvar_setup((void*)0,105,0,0,-1,-1,-1,4,"fCudaInitialized=",0,(char*)NULL);
   G__memvar_setup((void*)0,85,0,0,G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_TClass),-1,-2,4,"fgIsA=",0,(char*)NULL);
   }
   G__tag_memvar_reset();
}

extern "C" void G__cpp_setup_memvarG__AliHLTTPCCAGPU() {
}
/***********************************************************
************************************************************
************************************************************
************************************************************
************************************************************
************************************************************
************************************************************
***********************************************************/

/*********************************************************
* Member function information setup for each class
*********************************************************/
static void G__setup_memfuncAliHLTTPCCAGPUTrackerNVCC(void) {
   /* AliHLTTPCCAGPUTrackerNVCC */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCAGPUTrackerNVCC));
   G__memfunc_setup("AliHLTTPCCAGPUTrackerNVCC",2123,G__G__AliHLTTPCCAGPU_229_0_1, 105, G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCAGPUTrackerNVCC), -1, 0, 0, 1, 1, 0, "", (char*)NULL, (void*) NULL, 0);
   G__memfunc_setup("InitGPU",640,(G__InterfaceMethod) NULL,105, -1, -1, 0, 2, 1, 1, 0, 
"i - - 0 '12' sliceCount i - - 0 '-1' forceDeviceID", (char*)NULL, (void*) NULL, 1);
   G__memfunc_setup("Reconstruct",1180,(G__InterfaceMethod) NULL,105, -1, -1, 0, 4, 1, 1, 0, 
"U 'AliHLTTPCCASliceOutput' - 2 - pOutput U 'AliHLTTPCCAClusterData' - 0 - pClusterData "
"i - - 0 - fFirstSlice i - - 0 '-1' fSliceCount", (char*)NULL, (void*) NULL, 1);
   G__memfunc_setup("ExitGPU",646,(G__InterfaceMethod) NULL,105, -1, -1, 0, 0, 1, 1, 0, "", (char*)NULL, (void*) NULL, 1);
   G__memfunc_setup("SetDebugLevel",1291,(G__InterfaceMethod) NULL,121, -1, -1, 0, 2, 1, 1, 0, 
"i - - 10 - dwLevel U 'basic_ostream<char,char_traits<char> >' 'ostream' 40 'NULL' NewOutFile", (char*)NULL, (void*) NULL, 1);
   G__memfunc_setup("SetGPUTrackerOption",1885,(G__InterfaceMethod) NULL,105, -1, -1, 0, 2, 1, 1, 0, 
"C - - 0 - OptionName i - - 0 - OptionValue", (char*)NULL, (void*) NULL, 1);
   G__memfunc_setup("PerfTimer",910,(G__InterfaceMethod) NULL,77, -1, -1, 0, 2, 1, 1, 0, 
"i - - 0 - iSlice h - - 0 - i", (char*)NULL, (void*) NULL, 1);
   G__memfunc_setup("InitializeSliceParam",2035,(G__InterfaceMethod) NULL,105, -1, -1, 0, 2, 1, 1, 0, 
"i - - 0 - iSlice u 'AliHLTTPCCAParam' - 1 - param", (char*)NULL, (void*) NULL, 1);
   G__memfunc_setup("SetOutputControl",1694,(G__InterfaceMethod) NULL,121, -1, -1, 0, 1, 1, 1, 0, "U 'AliHLTTPCCASliceOutput::outputControlStruct' - 0 - val", (char*)NULL, (void*) NULL, 1);
   G__memfunc_setup("OutputControl",1394,(G__InterfaceMethod) NULL,85, G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCASliceOutputcLcLoutputControlStruct), -1, 0, 0, 1, 1, 9, "", (char*)NULL, (void*) NULL, 1);
   G__memfunc_setup("GetSliceCount",1305,(G__InterfaceMethod) NULL,105, -1, -1, 0, 0, 1, 1, 8, "", (char*)NULL, (void*) NULL, 1);
   G__memfunc_setup("RowMemory",945,(G__InterfaceMethod) NULL, 89, -1, -1, 0, 2, 3, 4, 0, 
"Y - - 40 - BaseMemory i - - 0 - iSlice", (char*)NULL, (void*) NULL, 0);
   G__memfunc_setup("CommonMemory",1250,(G__InterfaceMethod) NULL, 89, -1, -1, 0, 2, 3, 4, 0, 
"Y - - 40 - BaseMemory i - - 0 - iSlice", (char*)NULL, (void*) NULL, 0);
   G__memfunc_setup("SliceDataMemory",1507,(G__InterfaceMethod) NULL, 89, -1, -1, 0, 2, 3, 4, 0, 
"Y - - 40 - BaseMemory i - - 0 - iSlice", (char*)NULL, (void*) NULL, 0);
   G__memfunc_setup("GlobalMemory",1226,(G__InterfaceMethod) NULL, 89, -1, -1, 0, 2, 1, 4, 8, 
"Y - - 40 - BaseMemory i - - 0 - iSlice", (char*)NULL, (void*) NULL, 0);
   G__memfunc_setup("TracksMemory",1249,(G__InterfaceMethod) NULL, 89, -1, -1, 0, 2, 1, 4, 8, 
"Y - - 40 - BaseMemory i - - 0 - iSlice", (char*)NULL, (void*) NULL, 0);
   G__memfunc_setup("TrackerMemory",1349,(G__InterfaceMethod) NULL, 89, -1, -1, 0, 2, 1, 4, 8, 
"Y - - 40 - BaseMemory i - - 0 - iSlice", (char*)NULL, (void*) NULL, 0);
   G__memfunc_setup("DumpRowBlocks",1324,(G__InterfaceMethod) NULL, 121, -1, -1, 0, 3, 1, 4, 0, 
"U 'AliHLTTPCCATracker' - 0 - tracker i - - 0 - iSlice "
"g - - 0 'true' check", (char*)NULL, (void*) NULL, 0);
   G__memfunc_setup("GetThread",888,(G__InterfaceMethod) NULL, 105, -1, -1, 0, 0, 1, 4, 0, "", (char*)NULL, (void*) NULL, 0);
   G__memfunc_setup("ReleaseGlobalLock",1691,(G__InterfaceMethod) NULL, 121, -1, -1, 0, 1, 1, 4, 0, "Y - - 0 - sem", (char*)NULL, (void*) NULL, 0);
   G__memfunc_setup("CheckMemorySizes",1637,(G__InterfaceMethod) NULL, 105, -1, -1, 0, 1, 1, 4, 0, "i - - 0 - sliceCount", (char*)NULL, (void*) NULL, 0);
   G__memfunc_setup("CUDASync",698,(G__InterfaceMethod) NULL, 105, -1, -1, 0, 1, 1, 4, 0, "C - - 0 '\"UNKNOWN\"' state", (char*)NULL, (void*) NULL, 0);
   G__memfunc_setup("StandalonePerfTime",1829,(G__InterfaceMethod) NULL, 121, -1, -1, 0, 2, 1, 4, 0, 
"i - - 0 - iSlice i - - 0 - i", (char*)NULL, (void*) NULL, 0);
   G__memfunc_setup("AliHLTTPCCAGPUTrackerNVCC",2123,(G__InterfaceMethod) NULL, 105, G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCAGPUTrackerNVCC), -1, 0, 1, 1, 4, 0, "u 'AliHLTTPCCAGPUTrackerNVCC' - 11 - -", (char*)NULL, (void*) NULL, 0);
   G__memfunc_setup("operator=",937,(G__InterfaceMethod) NULL, 117, G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCAGPUTrackerNVCC), -1, 1, 1, 1, 4, 0, "u 'AliHLTTPCCAGPUTrackerNVCC' - 11 - -", (char*)NULL, (void*) NULL, 0);
   G__memfunc_setup("Class",502,G__G__AliHLTTPCCAGPU_229_0_26, 85, G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_TClass), -1, 0, 0, 3, 1, 0, "", (char*)NULL, (void*) G__func2void( (TClass* (*)())(&AliHLTTPCCAGPUTrackerNVCC::Class) ), 0);
   G__memfunc_setup("Class_Name",982,G__G__AliHLTTPCCAGPU_229_0_27, 67, -1, -1, 0, 0, 3, 1, 1, "", (char*)NULL, (void*) G__func2void( (const char* (*)())(&AliHLTTPCCAGPUTrackerNVCC::Class_Name) ), 0);
   G__memfunc_setup("Class_Version",1339,G__G__AliHLTTPCCAGPU_229_0_28, 115, -1, G__defined_typename("Version_t"), 0, 0, 3, 1, 0, "", (char*)NULL, (void*) G__func2void( (Version_t (*)())(&AliHLTTPCCAGPUTrackerNVCC::Class_Version) ), 0);
   G__memfunc_setup("Dictionary",1046,G__G__AliHLTTPCCAGPU_229_0_29, 121, -1, -1, 0, 0, 3, 1, 0, "", (char*)NULL, (void*) G__func2void( (void (*)())(&AliHLTTPCCAGPUTrackerNVCC::Dictionary) ), 0);
   G__memfunc_setup("IsA",253,(G__InterfaceMethod) NULL,85, G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_TClass), -1, 0, 0, 1, 1, 8, "", (char*)NULL, (void*) NULL, 1);
   G__memfunc_setup("ShowMembers",1132,(G__InterfaceMethod) NULL,121, -1, -1, 0, 2, 1, 1, 0, 
"u 'TMemberInspector' - 1 - insp C - - 0 - parent", (char*)NULL, (void*) NULL, 1);
   G__memfunc_setup("Streamer",835,(G__InterfaceMethod) NULL,121, -1, -1, 0, 1, 1, 1, 0, "u 'TBuffer' - 1 - b", (char*)NULL, (void*) NULL, 1);
   G__memfunc_setup("StreamerNVirtual",1656,G__G__AliHLTTPCCAGPU_229_0_33, 121, -1, -1, 0, 1, 1, 1, 0, "u 'TBuffer' - 1 - b", (char*)NULL, (void*) NULL, 0);
   G__memfunc_setup("DeclFileName",1145,G__G__AliHLTTPCCAGPU_229_0_34, 67, -1, -1, 0, 0, 3, 1, 1, "", (char*)NULL, (void*) G__func2void( (const char* (*)())(&AliHLTTPCCAGPUTrackerNVCC::DeclFileName) ), 0);
   G__memfunc_setup("ImplFileLine",1178,G__G__AliHLTTPCCAGPU_229_0_35, 105, -1, -1, 0, 0, 3, 1, 0, "", (char*)NULL, (void*) G__func2void( (int (*)())(&AliHLTTPCCAGPUTrackerNVCC::ImplFileLine) ), 0);
   G__memfunc_setup("ImplFileName",1171,G__G__AliHLTTPCCAGPU_229_0_36, 67, -1, -1, 0, 0, 3, 1, 1, "", (char*)NULL, (void*) G__func2void( (const char* (*)())(&AliHLTTPCCAGPUTrackerNVCC::ImplFileName) ), 0);
   G__memfunc_setup("DeclFileLine",1152,G__G__AliHLTTPCCAGPU_229_0_37, 105, -1, -1, 0, 0, 3, 1, 0, "", (char*)NULL, (void*) G__func2void( (int (*)())(&AliHLTTPCCAGPUTrackerNVCC::DeclFileLine) ), 0);
   // automatic destructor
   G__memfunc_setup("~AliHLTTPCCAGPUTrackerNVCC", 2249, G__G__AliHLTTPCCAGPU_229_0_38, (int) ('y'), -1, -1, 0, 0, 1, 1, 0, "", (char*) NULL, (void*) NULL, 1);
   G__tag_memfunc_reset();
}


/*********************************************************
* Member function information setup
*********************************************************/
extern "C" void G__cpp_setup_memfuncG__AliHLTTPCCAGPU() {
}

/*********************************************************
* Global variable information setup for each class
*********************************************************/
static void G__cpp_setup_global0() {

   /* Setting up global variables */
   G__resetplocal();

}

static void G__cpp_setup_global1() {
}

static void G__cpp_setup_global2() {
}

static void G__cpp_setup_global3() {
}

static void G__cpp_setup_global4() {

   G__resetglobalenv();
}
extern "C" void G__cpp_setup_globalG__AliHLTTPCCAGPU() {
  G__cpp_setup_global0();
  G__cpp_setup_global1();
  G__cpp_setup_global2();
  G__cpp_setup_global3();
  G__cpp_setup_global4();
}

/*********************************************************
* Global function information setup for each class
*********************************************************/
static void G__cpp_setup_func0() {
   G__lastifuncposition();

}

static void G__cpp_setup_func1() {
}

static void G__cpp_setup_func2() {
}

static void G__cpp_setup_func3() {
}

static void G__cpp_setup_func4() {
}

static void G__cpp_setup_func5() {
}

static void G__cpp_setup_func6() {
}

static void G__cpp_setup_func7() {
}

static void G__cpp_setup_func8() {
}

static void G__cpp_setup_func9() {
}

static void G__cpp_setup_func10() {
}

static void G__cpp_setup_func11() {

   G__resetifuncposition();
}

extern "C" void G__cpp_setup_funcG__AliHLTTPCCAGPU() {
  G__cpp_setup_func0();
  G__cpp_setup_func1();
  G__cpp_setup_func2();
  G__cpp_setup_func3();
  G__cpp_setup_func4();
  G__cpp_setup_func5();
  G__cpp_setup_func6();
  G__cpp_setup_func7();
  G__cpp_setup_func8();
  G__cpp_setup_func9();
  G__cpp_setup_func10();
  G__cpp_setup_func11();
}

/*********************************************************
* Class,struct,union,enum tag information setup
*********************************************************/
/* Setup class/struct taginfo */
G__linked_taginfo G__G__AliHLTTPCCAGPULN_TClass = { "TClass" , 99 , -1 };
G__linked_taginfo G__G__AliHLTTPCCAGPULN_TBuffer = { "TBuffer" , 99 , -1 };
G__linked_taginfo G__G__AliHLTTPCCAGPULN_TMemberInspector = { "TMemberInspector" , 99 , -1 };
G__linked_taginfo G__G__AliHLTTPCCAGPULN_basic_ostreamlEcharcOchar_traitslEchargRsPgR = { "basic_ostream<char,char_traits<char> >" , 99 , -1 };
G__linked_taginfo G__G__AliHLTTPCCAGPULN_vectorlEROOTcLcLTSchemaHelpercOallocatorlEROOTcLcLTSchemaHelpergRsPgR = { "vector<ROOT::TSchemaHelper,allocator<ROOT::TSchemaHelper> >" , 99 , -1 };
G__linked_taginfo G__G__AliHLTTPCCAGPULN_reverse_iteratorlEvectorlEROOTcLcLTSchemaHelpercOallocatorlEROOTcLcLTSchemaHelpergRsPgRcLcLiteratorgR = { "reverse_iterator<vector<ROOT::TSchemaHelper,allocator<ROOT::TSchemaHelper> >::iterator>" , 99 , -1 };
G__linked_taginfo G__G__AliHLTTPCCAGPULN_AliHLTTPCCASliceOutput = { "AliHLTTPCCASliceOutput" , 99 , -1 };
G__linked_taginfo G__G__AliHLTTPCCAGPULN_AliHLTTPCCASliceOutputcLcLoutputControlStruct = { "AliHLTTPCCASliceOutput::outputControlStruct" , 115 , -1 };
G__linked_taginfo G__G__AliHLTTPCCAGPULN_AliHLTTPCCAClusterData = { "AliHLTTPCCAClusterData" , 99 , -1 };
G__linked_taginfo G__G__AliHLTTPCCAGPULN_AliHLTTPCCAParam = { "AliHLTTPCCAParam" , 99 , -1 };
G__linked_taginfo G__G__AliHLTTPCCAGPULN_AliHLTTPCCAGPUTracker = { "AliHLTTPCCAGPUTracker" , 99 , -1 };
G__linked_taginfo G__G__AliHLTTPCCAGPULN_AliHLTTPCCATracker = { "AliHLTTPCCATracker" , 99 , -1 };
G__linked_taginfo G__G__AliHLTTPCCAGPULN_AliHLTLogging = { "AliHLTLogging" , 99 , -1 };
G__linked_taginfo G__G__AliHLTTPCCAGPULN_AliHLTTPCCAGPUTrackerNVCC = { "AliHLTTPCCAGPUTrackerNVCC" , 99 , -1 };

/* Reset class/struct taginfo */
extern "C" void G__cpp_reset_tagtableG__AliHLTTPCCAGPU() {
  G__G__AliHLTTPCCAGPULN_TClass.tagnum = -1 ;
  G__G__AliHLTTPCCAGPULN_TBuffer.tagnum = -1 ;
  G__G__AliHLTTPCCAGPULN_TMemberInspector.tagnum = -1 ;
  G__G__AliHLTTPCCAGPULN_basic_ostreamlEcharcOchar_traitslEchargRsPgR.tagnum = -1 ;
  G__G__AliHLTTPCCAGPULN_vectorlEROOTcLcLTSchemaHelpercOallocatorlEROOTcLcLTSchemaHelpergRsPgR.tagnum = -1 ;
  G__G__AliHLTTPCCAGPULN_reverse_iteratorlEvectorlEROOTcLcLTSchemaHelpercOallocatorlEROOTcLcLTSchemaHelpergRsPgRcLcLiteratorgR.tagnum = -1 ;
  G__G__AliHLTTPCCAGPULN_AliHLTTPCCASliceOutput.tagnum = -1 ;
  G__G__AliHLTTPCCAGPULN_AliHLTTPCCASliceOutputcLcLoutputControlStruct.tagnum = -1 ;
  G__G__AliHLTTPCCAGPULN_AliHLTTPCCAClusterData.tagnum = -1 ;
  G__G__AliHLTTPCCAGPULN_AliHLTTPCCAParam.tagnum = -1 ;
  G__G__AliHLTTPCCAGPULN_AliHLTTPCCAGPUTracker.tagnum = -1 ;
  G__G__AliHLTTPCCAGPULN_AliHLTTPCCATracker.tagnum = -1 ;
  G__G__AliHLTTPCCAGPULN_AliHLTLogging.tagnum = -1 ;
  G__G__AliHLTTPCCAGPULN_AliHLTTPCCAGPUTrackerNVCC.tagnum = -1 ;
}


extern "C" void G__cpp_setup_tagtableG__AliHLTTPCCAGPU() {

   /* Setting up class,struct,union tag entry */
   G__get_linked_tagnum_fwd(&G__G__AliHLTTPCCAGPULN_TClass);
   G__get_linked_tagnum_fwd(&G__G__AliHLTTPCCAGPULN_TBuffer);
   G__get_linked_tagnum_fwd(&G__G__AliHLTTPCCAGPULN_TMemberInspector);
   G__get_linked_tagnum_fwd(&G__G__AliHLTTPCCAGPULN_basic_ostreamlEcharcOchar_traitslEchargRsPgR);
   G__get_linked_tagnum_fwd(&G__G__AliHLTTPCCAGPULN_vectorlEROOTcLcLTSchemaHelpercOallocatorlEROOTcLcLTSchemaHelpergRsPgR);
   G__get_linked_tagnum_fwd(&G__G__AliHLTTPCCAGPULN_reverse_iteratorlEvectorlEROOTcLcLTSchemaHelpercOallocatorlEROOTcLcLTSchemaHelpergRsPgRcLcLiteratorgR);
   G__get_linked_tagnum_fwd(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCASliceOutput);
   G__get_linked_tagnum_fwd(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCASliceOutputcLcLoutputControlStruct);
   G__get_linked_tagnum_fwd(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCAClusterData);
   G__get_linked_tagnum_fwd(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCAParam);
   G__get_linked_tagnum_fwd(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCAGPUTracker);
   G__get_linked_tagnum_fwd(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCATracker);
   G__get_linked_tagnum_fwd(&G__G__AliHLTTPCCAGPULN_AliHLTLogging);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__AliHLTTPCCAGPULN_AliHLTTPCCAGPUTrackerNVCC),sizeof(AliHLTTPCCAGPUTrackerNVCC),-1,265984,(char*)NULL,G__setup_memvarAliHLTTPCCAGPUTrackerNVCC,G__setup_memfuncAliHLTTPCCAGPUTrackerNVCC);
}
extern "C" void G__cpp_setupG__AliHLTTPCCAGPU(void) {
  G__check_setup_version(30051515,"G__cpp_setupG__AliHLTTPCCAGPU()");
  G__set_cpp_environmentG__AliHLTTPCCAGPU();
  G__cpp_setup_tagtableG__AliHLTTPCCAGPU();

  G__cpp_setup_inheritanceG__AliHLTTPCCAGPU();

  G__cpp_setup_typetableG__AliHLTTPCCAGPU();

  G__cpp_setup_memvarG__AliHLTTPCCAGPU();

  G__cpp_setup_memfuncG__AliHLTTPCCAGPU();
  G__cpp_setup_globalG__AliHLTTPCCAGPU();
  G__cpp_setup_funcG__AliHLTTPCCAGPU();

   if(0==G__getsizep2memfunc()) G__get_sizep2memfuncG__AliHLTTPCCAGPU();
  return;
}
class G__cpp_setup_initG__AliHLTTPCCAGPU {
  public:
    G__cpp_setup_initG__AliHLTTPCCAGPU() { G__add_setup_func("G__AliHLTTPCCAGPU",(G__incsetup)(&G__cpp_setupG__AliHLTTPCCAGPU)); G__call_setup_funcs(); }
   ~G__cpp_setup_initG__AliHLTTPCCAGPU() { G__remove_setup_func("G__AliHLTTPCCAGPU"); }
};
G__cpp_setup_initG__AliHLTTPCCAGPU G__cpp_setup_initializerG__AliHLTTPCCAGPU;

