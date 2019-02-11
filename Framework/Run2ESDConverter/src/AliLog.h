#ifndef ALILOG_H
#define ALILOG_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

#include <TClass.h>
#include <TObjArray.h>
#include <TObject.h>
#include <TString.h>

using std::ostream;

// deprecation macro
#if defined(__GNUC__) && (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
# define ALIROOT_DEPRECATED(func) func  __attribute__ ((deprecated))
#elif defined(_MSC_VER) && _MSC_VER >= 1300
# define ALIROOT_DEPRECATED(func) __declspec(deprecated) func
#else
# define ALIROOT_DEPRECATED(func) func
#endif

/**
 * class for logging debug, info and error messages
 */
class AliLog: public TObject
{
 public:

		// Log4j log levels: TRACE, DEBUG, INFO, WARN, ERROR, FATAL
  enum EType_t {kFatal = 0, kError, kWarning, kInfo, kDebug, kMaxType};
  typedef void (*AliLogNotification)(EType_t type, const char* message );

		// NB: singleton constructor & destructor should not be public!
		// ALIROOT_DEPRECATED(AliLog());
		// ALIROOT_DEPRECATED(virtual ~AliLog());

		// NB: singleton deprecated static instance method
		// ALIROOT_DEPRECATED(static AliLog* Instance() {return fgInstance;};)

		// get root logger singleton instance
		static AliLog *GetRootLogger();

		// delete root logger singleton instance
		static void DeleteRootLogger(); 

		// NB: the following functions should not be static
		// NB: deprecated: logging configuration should be made through to a configuration file
  static void  EnableCoreDump(Bool_t enabled);
  static void MakeCoreDump(const char *fout);	
  static void  EnableDebug(Bool_t enabled);
  static void  SetGlobalLogLevel(EType_t type);
  static Int_t GetGlobalLogLevel();
  static void  SetGlobalDebugLevel(Int_t level);
  static Int_t GetGlobalDebugLevel();
  static void  SetModuleDebugLevel(const char* module, Int_t level);
  static void  ClearModuleDebugLevel(const char* module);
  static void  SetClassDebugLevel(const char* className, Int_t level);
  static void  ClearClassDebugLevel(const char* className);

  static void  SetStandardOutput();
  static void  SetStandardOutput(EType_t type);
  static void  SetErrorOutput();
  static void  SetErrorOutput(EType_t type);
  static void  SetFileOutput(const char* fileName);
  static void  SetFileOutput(EType_t type, const char* fileName);
  static void  SetStreamOutput(ostream* stream);
  static void  SetStreamOutput(EType_t type, ostream* stream);
  static void  SetLogNotification(AliLogNotification pCallBack);
  static void  SetLogNotification(EType_t type, AliLogNotification pCallBack);
  static void  Flush();

  static void  SetHandleRootMessages(Bool_t on);

  static void  SetPrintType(Bool_t on);
  static void  SetPrintType(EType_t type, Bool_t on);
  static void  SetPrintModule(Bool_t on);
  static void  SetPrintModule(EType_t type, Bool_t on);
  static void  SetPrintScope(Bool_t on);
  static void  SetPrintScope(EType_t type, Bool_t on);
  static void  SetPrintLocation(Bool_t on);
  static void  SetPrintLocation(EType_t type, Bool_t on);

  static void  SetPrintRepetitions(Bool_t on);

  static void  WriteToFile(const char* name, Int_t option = 0);

  // the following public methods are used by the preprocessor macros 
  // and should not be called directly
  static Bool_t IsDebugEnabled() {return fgDebugEnabled;}
  static Int_t GetDebugLevel(const char* module, const char* className);
  static void  Message(UInt_t level, const char* message, 
                       const char* module, const char* className,
                       const char* function, const char* file, Int_t line);
  static void  Debug(UInt_t level, const char* message, 
                     const char* module, const char* className,
                     const char* function, const char* file, Int_t line);

  static Int_t RedirectStdoutTo(EType_t type, UInt_t level, const char* module, 
                                const char* className, const char* function,
                                const char* file, Int_t line, Bool_t print);
  static Int_t RedirectStderrTo(EType_t type, UInt_t level, const char* module, 
                                const char* className, const char* function,
                                const char* file, Int_t line, Bool_t print);
  static void  RestoreStdout(Int_t original);
  static void  RestoreStderr(Int_t original);

  static ostream& Stream(EType_t type, UInt_t level,
                         const char* module, const char* className,
                         const char* function, const char* file, Int_t line);
  static void TestException(Int_t level=10); 
 private:

		// constructor is made private for implementing a singleton
		AliLog();
		virtual ~AliLog();

		// NOT IMPLEMENTED?
  AliLog(const AliLog& log);
  AliLog& operator = (const AliLog& log);

  void           ReadEnvSettings();

  static void    RootErrorHandler(Int_t level, Bool_t abort, 
				  const char* location, const char* message);

  void           CloseFile(Int_t type);
  FILE*          GetOutputStream(Int_t type);

  UInt_t         GetLogLevel(const char* module, const char* className) const;
  void           PrintMessage(UInt_t type, const char* message, 
                              const char* module, const char* className,
                              const char* function, 
                              const char* file, Int_t line);

  void           PrintString(Int_t type, FILE* stream, const char* format, ...);
  void           PrintRepetitions();

  Int_t          RedirectTo(FILE* stream, EType_t type, UInt_t level,
                            const char* module, const char* className,
                            const char* function,
                            const char* file, Int_t line, Bool_t print);

  ostream&       GetStream(EType_t type, UInt_t level,
                           const char* module, const char* className,
                           const char* function, const char* file, Int_t line);

  enum {kDebugOffset = kDebug-1};

  static AliLog* fgInstance;                 //! pointer to current instance
  static Bool_t  fgDebugEnabled;             // flag for debug en-/disabling
  static Bool_t  fgCoreEnabled;             // flag for core dump en-/disabling

  UInt_t         fGlobalLogLevel;            // global logging level
  TObjArray      fModuleDebugLevels;         // debug levels for modules
  TObjArray      fClassDebugLevels;          // debug levels for classes

  Int_t          fOutputTypes[kMaxType];     // types of output streams
  TString        fFileNames[kMaxType];       // file names
  FILE*          fOutputFiles[kMaxType];     //! log output files
  ostream*       fOutputStreams[kMaxType];   //! log output streams

  Bool_t         fPrintType[kMaxType];       // print type on/off
  Bool_t         fPrintModule[kMaxType];     // print module on/off
  Bool_t         fPrintScope[kMaxType];      // print scope/class name on/off
  Bool_t         fPrintLocation[kMaxType];   // print file and line on/off

  Bool_t         fPrintRepetitions;          // print number of repetitions instead of repeated message on/off

  Int_t          fRepetitions;               //! counter of repetitions
  UInt_t         fLastType;                  //! type of last message
  TString        fLastMessage;               //! last message
  TString        fLastModule;                //! module name of last message
  TString        fLastClassName;             //! class name of last message
  TString        fLastFunction;              //! function name of last message
  TString        fLastFile;                  //! file name of last message
  Int_t          fLastLine;                  //! line number of last message
  AliLogNotification fCallBacks[kMaxType];   //! external notification callback

  ClassDef(AliLog, 1)   // class for logging debug, info and error messages
};


// module name macro
#ifdef _MODULE_
# define MODULENAME() _MODULE_
#else
# define MODULENAME() "NoModule"
#endif

// function name macro
#if defined(__GNUC__) || defined(__ICC) || defined(__ECC) || defined(__APPLE__)
# define FUNCTIONNAME() __FUNCTION__
// #elif defined(__HP_aCC) || defined(__alpha) || defined(__DECCXX)
// #define FUNCTIONNAME() __FUNC__
#else
# define FUNCTIONNAME() "???"
#endif

// redirection
/** 
 * Redirect output to std::cout to specified log stream 
 * 
 * @param type      Type of stream to re-direct to
 * @param level     Level of output
 * @param scope     Scope
 * @param whatever  Any code that will output to std::cout 
 */
#define REDIRECTSTDOUT(type, level, scope, whatever) \
  do {Int_t originalStdout = AliLog::RedirectStdoutTo(type, level, MODULENAME(), scope, FUNCTIONNAME(), __FILE__, __LINE__, kFALSE); \
    whatever; AliLog::RestoreStdout(originalStdout);} while(false)
/** 
 * Redirect output to std::cerr to specified log stream 
 * 
 * @param type      Type of stream to re-direct to
 * @param level     Level of output
 * @param scope     Scope
 * @param whatever  Any code that will output to std::cout 
 */
#define REDIRECTSTDERR(type, level, scope, whatever) \
  do {Int_t originalStderr = AliLog::RedirectStderrTo(type, level, MODULENAME(), scope, FUNCTIONNAME(), __FILE__, __LINE__, kFALSE); \
    whatever; AliLog::RestoreStderr(originalStderr);} while(false)
/** 
 * Redirect output to std::cout and std::cerr to specified log stream 
 * 
 * @param type      Type of stream to re-direct to
 * @param level     Level of output
 * @param scope     Scope
 * @param whatever  Any code that will output to std::cout or std::cerr
 */
#define REDIRECTSTDOUTANDSTDERR(type, level, scope, whatever) \
  do {Int_t originalStdout = AliLog::RedirectStdoutTo(type, level, MODULENAME(), scope, FUNCTIONNAME(), __FILE__, __LINE__, kFALSE); \
    Int_t originalStderr = AliLog::RedirectStderrTo(type, level, MODULENAME(), scope, FUNCTIONNAME(), __FILE__, __LINE__, kFALSE); \
    whatever; AliLog::RestoreStderr(originalStderr); AliLog::RestoreStdout(originalStdout);} while(false)


// debug level
#ifdef LOG_NO_DEBUG
# define AliDebugLevel() -1
# define AliDebugLevelClass() -1
# define AliDebugLevelGeneral(scope) -1
#else
/** 
 * Get the object scope debug level
 */
# define AliDebugLevel() ((AliLog::IsDebugEnabled()) ? AliLog::GetDebugLevel(MODULENAME(), ClassName()) : -1)
/** 
 * Get the class (ROOT-enabled) debug level
 */
# define AliDebugLevelClass() ((AliLog::IsDebugEnabled()) ? AliLog::GetDebugLevel(MODULENAME(), Class()->GetName()) : -1)
/**
 * Get the debug level associated with scope 
 * @param scope Scope 
 */
# define AliDebugLevelGeneral(scope) ((AliLog::IsDebugEnabled()) ? AliLog::GetDebugLevel(MODULENAME(), scope) : -1)
#endif

// debug messages
#ifdef LOG_NO_DEBUG
# define AliDebug(level, message) do { } while (false)
# define AliDebugClass(level, message) do { } while (false)
# define AliDebugGeneral(scope, level, message) do { } while (false)
# define AliDebugF(level, message,...) do { } while (false)
# define AliDebugClassF(level, message,...) do { } while (false)
# define AliDebugGeneralF(scope, level, message,...) do { } while (false)
#else

// inspired by log4cxx code (see log4cxx/Logger.h)
// implements GCC branch prediction for increasing logging performance
# if !defined(ALIROOT_UNLIKELY)
#  if defined(__GNUC__) && (__GNUC__ >= 3)
/**
 * Provides optimization hint to the compiler
 * to optimize for the expression being false.
 * @param expr boolean expression.
 * @returns value of expression.
 */
#   define ALIROOT_UNLIKELY(expr) __builtin_expect(expr, 0)
#  else
/**
 * Provides optimization hint to the compiler
 * to optimize for the expression being false.
 * @param expr boolean expression.
 * @returns value of expression.
 */
#   define ALIROOT_UNLIKELY(expr) expr
#  endif
# endif 

/**
 * 
 * Logs a message to a specified logger with the DEBUG level.
 * 
 * @param logLevel the debug level.
 * @param message message to print in the following format: Form(message).
 * Note, that message should contain balanced parenthesis, like 
 * <code>AliDebug(1, Form("Failed to decode line %d of %s", line, filename));</code> 
 */
# define AliDebug(logLevel, message) \
        do { if (ALIROOT_UNLIKELY(AliLog::IsDebugEnabled() && AliLog::GetDebugLevel(MODULENAME(), ClassName()) >= logLevel)) {\
	  AliLog::Debug(logLevel, message, MODULENAME(), ClassName(), FUNCTIONNAME(), __FILE__, __LINE__); }} while (0)
/**
 * 
 * Logs a message to a specified logger with the DEBUG level.  For use
 * in static member functions of a class 
 * 
 * @param logLevel the debug level.
 * @param message message to print in the following format: Form(message).
 * Note, that message should contain balanced parenthesis, like 
 * <code>AliDebug(1, Form("Failed to decode line %d of %s", line, filename));</code> 
 */
# define AliDebugClass(logLevel, message) \
	do { if (ALIROOT_UNLIKELY(AliLog::IsDebugEnabled() && AliLog::GetDebugLevel(MODULENAME(), Class()->GetName()) >= logLevel)) {\
	  AliLog::Debug(logLevel, message, MODULENAME(), Class()->GetName(), FUNCTIONNAME(), __FILE__, __LINE__); }} while (0)

/**
 * Logs a message to a specified logger with the DEBUG level.  For use
 * in non-ROOT-enabled-class scope.
 * 
 * @param scope the logging scope.
 * @param logLevel the debug level.
 * @param message message to print in the following format: Form(message).
 * Note, that message should contain balanced parenthesis, like 
 * <code>AliDebug(1, Form("Failed to decode line %d of %s", line, filename));</code> 
*/
# define AliDebugGeneral(scope, logLevel, message) \
	do { if (ALIROOT_UNLIKELY(AliLog::IsDebugEnabled() && AliLog::GetDebugLevel(MODULENAME(), scope) >= logLevel)) {\
	  AliLog::Debug(logLevel, message, MODULENAME(), scope, FUNCTIONNAME(), __FILE__, __LINE__); }} while (0)
/** 
 * Macro to output debugging information.  This excepts a printf-like
 * format statement.   Note, at least 3 arguments (in total) must be
 * passed. 
 * 
 * @param logLevel Debug level
 * @param format   Printf-like format. 
 */
# define AliDebugF(logLevel,format,...) \
do { if (ALIROOT_UNLIKELY(AliLog::IsDebugEnabled() && AliLog::GetDebugLevel(MODULENAME(), ClassName()) >= logLevel)) { \
    TString m;m.Form(format,__VA_ARGS__);					\
    AliLog::Debug(logLevel, m, MODULENAME(), ClassName(), FUNCTIONNAME(), __FILE__, __LINE__); }} while (0)
/** 
 * Outut debug information, filtered on debug level.  For use in
 * static member function of a ROOT-enabled class. This excepts a
 * printf-like format statement.  Note, at least 3 arguments (in
 * total) must be passed.
 * 
 * @param logLevel Debug level
 * @param format   Printf-like format 
 * 
 * @return 
 */
# define AliDebugClassF(logLevel,format,...) \
  do { if (ALIROOT_UNLIKELY(AliLog::IsDebugEnabled() && AliLog::GetDebugLevel(MODULENAME(), Class()->GetName()) >= logLevel)) { \
      TString m;m.Form(format,__VA_ARGS__);					\
      AliLog::Debug(logLevel, m, MODULENAME(), Class()->GetName(), FUNCTIONNAME(), __FILE__, __LINE__); }} while (0)
/** 
 * Outut debug information, filtered on debug level.  For use in
 * static member function of a non-ROOT-enabled class-scope. This
 * excepts a printf-like format statement.  Note, at least 3 arguments
 * (in total) must be passed.
 * 
 * @param scope    Scope 
 * @param logLevel Debug level
 * @param format   Printf-like format 
 * 
 * @return 
 */
# define AliDebugGeneralF(scope,logLevel,format,...) \
  do { if (ALIROOT_UNLIKELY(AliLog::IsDebugEnabled() && AliLog::GetDebugLevel(MODULENAME(), scope) >= logLevel)) { \
      TString m;m.Form(format,__VA_ARGS__);					\
      AliLog::Debug(logLevel, m, MODULENAME(), scope, FUNCTIONNAME(), __FILE__, __LINE__); }} while (0)
    
#endif

// redirection to debug
#define StdoutToAliDebug(level, whatever) REDIRECTSTDOUT(AliLog::kDebug, level, ClassName(), whatever)
#define StderrToAliDebug(level, whatever) REDIRECTSTDERR(AliLog::kDebug, level, ClassName(), whatever)
#define ToAliDebug(level, whatever) REDIRECTSTDOUTANDSTDERR(AliLog::kDebug, level, ClassName(), whatever)
#define StdoutToAliDebugClass(level, whatever) REDIRECTSTDOUT(AliLog::kDebug, level, Class()->GetName(), whatever)
#define StderrToAliDebugClass(level, whatever) REDIRECTSTDERR(AliLog::kDebug, level, Class()->GetName(), whatever)
#define ToAliDebugClass(level, whatever) REDIRECTSTDOUTANDSTDERR(AliLog::kDebug, level, Class()->GetName(), whatever)
#define StdoutToAliDebugGeneral(scope, level, whatever) REDIRECTSTDOUT(AliLog::kDebug, level, scope, whatever)
#define StderrToAliDebugGeneral(scope, level, whatever) REDIRECTSTDERR(AliLog::kDebug, level, scope, whatever)
#define ToAliDebugGeneral(scope, level, whatever) REDIRECTSTDOUTANDSTDERR(AliLog::kDebug, level, scope, whatever)

// debug stream objects
#define AliDebugStream(level) AliLog::Stream(AliLog::kDebug, level, MODULENAME(), ClassName(), FUNCTIONNAME(), __FILE__, __LINE__)
#define AliDebugClassStream(level) AliLog::Stream(AliLog::kDebug, level, MODULENAME(), Class()->GetName(), FUNCTIONNAME(), __FILE__, __LINE__)
#define AliDebugGeneralStream(scope, level) AliLog::Stream(AliLog::kDebug, level, MODULENAME(), scope, FUNCTIONNAME(), __FILE__, __LINE__)


/** 
 * Macro that will output stuff using the logging facilities. 
 * 
 * @param lvl     Message level 
 * @param message Message to show 
 */
#define AliMessage(lvl,message) do { \
      AliLog::Message(lvl, message, MODULENAME(), ClassName(), FUNCTIONNAME(), __FILE__, __LINE__);} while(false) 
/** 
 * Macro that will output stuff using the logging facilities. 
 * For use in static member function of ROOT-enabled class-scope.
 *
 * @param lvl     Message level 
 * @param message Message to show 
 */
#define AliMessageClass(lvl,message) do { \
    AliLog::Message(lvl, message, MODULENAME(), Class()->GetName(), FUNCTIONNAME(), __FILE__, __LINE__);} while(false) 
/** 
 * Macro that will output stuff using the logging facilities. 
 * For use in non-ROOT-enabled class-scope.
 *
 * @param scope   Scope 
 * @param lvl     Message level 
 * @param message Message to show 
 */
#define AliMessageGeneral(scope,lvl,message) do {			\
    AliLog::Message(lvl, message, MODULENAME(), scope, FUNCTIONNAME(), __FILE__, __LINE__);} while(false) 
/** 
 * Print a message using the AliLog logging facility. This macro
 * accepts printf-like format arguments.  Note, at least 3 arguments
 * must be passed.  
 * @code
 *   AliMessageF(1, "foo");        // <-- Failes
 *   AliMessageF(1, "foo %d", 42); // <-- OK
 * @endcode
 *
 * @param lvl     Message level
 * @param format  printf-like format
 */
#define AliMessageF(lvl,format,...) do { \
  TString m; m.Form(format,__VA_ARGS__); \
  AliLog::Message(lvl, m, MODULENAME(), ClassName(), FUNCTIONNAME(), __FILE__, __LINE__);} while(false) 
/** 
 * Print a message using the AliLog logging facility. This macro
 * accepts printf-like format arguments.  Note, at least 3 arguments
 * must be passed.  
 * @code
 *   AliMessageF(1, "foo");        // <-- Failes
 *   AliMessageF(1, "foo %d", 42); // <-- OK
 * @endcode
 *
 * This is for static member function in for ROOT-enabled class-scope
 *
 * @param lvl     Message level
 * @param format  printf-like format
 */
#define AliMessageClassF(lvl,format,...) do { \
  TString m; m.Form(format,__VA_ARGS__); \
  AliLog::Message(lvl, m, MODULENAME(), Class()->GetName(), FUNCTIONNAME(), __FILE__, __LINE__);} while(false) 
/** 
 * Print a message using the AliLog logging facility. This macro
 * accepts printf-like format arguments.  Note, at least 3 arguments
 * must be passed.  
 * @code
 *   AliMessageF(1, "foo");        // <-- Failes
 *   AliMessageF(1, "foo %d", 42); // <-- OK
 * @endcode
 *
 * This is for non-ROOT-enabled class-scope
 *
 * @param scope   Scope 
 * @param lvl     Message level
 * @param format  printf-like format
 */
#define AliMessageGeneralF(scope,lvl,format,...) do {	\
  TString m; m.Form(format,__VA_ARGS__); \
  AliLog::Message(lvl, m, MODULENAME(), scope, FUNCTIONNAME(), __FILE__, __LINE__);} while(false) 

// info messages
#ifdef LOG_NO_INFO
# define AliInfo(message) do { } while (false)
# define AliInfoClass(message) do { } while (false)
# define AliInfoGeneral(scope, message) do { } while (false)
# define AliInfoF(message,...) do { } while (false)
# define AliInfoClassF(message,...) do { } while (false)
# define AliInfoGeneralF(scope, message,...) do { } while (false)
#else
/**
 * Forwards to AliMessage with log level of AliLog::kInfo
 * @see AliMessage 
 */
# define AliInfo(message)               AliMessage(AliLog::kInfo, message)
/**
 * Forwards to AliMessageClass with log level of AliLog::kInfo
 * @see AliMessageClass 
 */
# define AliInfoClass(message)          AliMessageClass(AliLog::kInfo, message)
/**
 * Forwards to AliMessageGeneral with log level of AliLog::kInfo
 * @see AliMessageGeneral
 */
# define AliInfoGeneral(scope, message) AliMessageGeneral(scope, AliLog::kInfo, message)
/**
 * Forwards to AliMessageF with log level of AliLog::kInfo
 * @see AliMessageF 
 */
# define AliInfoF(message,...)               AliMessageF(AliLog::kInfo, message, __VA_ARGS__)
/**
 * Forwards to AliMessageClassF with log level of AliLog::kInfo
 * @see AliMessageClassF 
 */
# define AliInfoClassF(message,...)          AliMessageClassF(AliLog::kInfo, message, __VA_ARGS__)
/**
 * Forwards to AliMessageGeneralF with log level of AliLog::kInfo
 * @see AliMessageGeneralF
 */
# define AliInfoGeneralF(scope,message,...)  AliMessageGeneralF(scope, AliLog::kInfo, message, __VA_ARGS__)
#endif

// redirection to info
#define StdoutToAliInfo(whatever) REDIRECTSTDOUT(AliLog::kInfo, 0, ClassName(), whatever)
#define StderrToAliInfo(whatever) REDIRECTSTDERR(AliLog::kInfo, 0, ClassName(), whatever)
#define ToAliInfo(whatever) REDIRECTSTDOUTANDSTDERR(AliLog::kInfo, 0, ClassName(), whatever)
#define StdoutToAliInfoClass(whatever) REDIRECTSTDOUT(AliLog::kInfo, 0, Class()->GetName(), whatever)
#define StderrToAliInfoClass(whatever) REDIRECTSTDERR(AliLog::kInfo, 0, Class()->GetName(), whatever)
#define ToAliInfoClass(whatever) REDIRECTSTDOUTANDSTDERR(AliLog::kInfo, 0, Class()->GetName(), whatever)
#define StdoutToAliInfoGeneral(scope, whatever) REDIRECTSTDOUT(AliLog::kInfo, 0, scope, whatever)
#define StderrToAliInfoGeneral(scope, whatever) REDIRECTSTDERR(AliLog::kInfo, 0, scope, whatever)
#define ToAliInfoGeneral(scope, whatever) REDIRECTSTDOUTANDSTDERR(AliLog::kInfo, 0, scope, whatever)

// info stream objects
#define AliInfoStream() AliLog::Stream(AliLog::kInfo, 0, MODULENAME(), ClassName(), FUNCTIONNAME(), __FILE__, __LINE__)
#define AliInfoClassStream() AliLog::Stream(AliLog::kInfo, 0, MODULENAME(), Class()->GetName(), FUNCTIONNAME(), __FILE__, __LINE__)
#define AliInfoGeneralStream(scope) AliLog::Stream(AliLog::kInfo, 0, MODULENAME(), scope, FUNCTIONNAME(), __FILE__, __LINE__)

// warning messages
#ifdef LOG_NO_WARNING
# define AliWarning(message) do { } while (false)
# define AliWarningClass(message) do { } while (false)
# define AliWarningGeneral(scope, message) do { } while (false)
# define AliWarningF(message,...) do { } while (false)
# define AliWarningClassF(message,...) do { } while (false)
# define AliWarningGeneralF(scope, message,...) do { } while (false)
#else
/**
 * Forwards to AliMessage with log level of AliLog::kWarning
 * @see AliMessage 
 */
# define AliWarning(message)               AliMessage(AliLog::kWarning, message)
/**
 * Forwards to AliMessageClass with log level of AliLog::kWarning
 * @see AliMessageClass 
 */
# define AliWarningClass(message)          AliMessageClass(AliLog::kWarning, message)
/**
 * Forwards to AliMessageGeneral with log level of AliLog::kWarning
 * @see AliMessageGeneral
 */
# define AliWarningGeneral(scope, message) AliMessageGeneral(scope, AliLog::kWarning, message)
/**
 * Forwards to AliMessageF with log level of AliLog::kWarning
 * @see AliMessageF 
 */
# define AliWarningF(message,...)               AliMessageF(AliLog::kWarning, message, __VA_ARGS__)
/**
 * Forwards to AliMessageClassF with log level of AliLog::kWarning
 * @see AliMessageClassF 
 */
# define AliWarningClassF(message,...)          AliMessageClassF(AliLog::kWarning, message, __VA_ARGS__)
/**
 * Forwards to AliMessageGeneralF with log level of AliLog::kWarning
 * @see AliMessageGeneralF
 */
# define AliWarningGeneralF(scope,message,...)  AliMessageGeneralF(scope, AliLog::kWarning, message, __VA_ARGS__)
#endif

// redirection to warning
#define StdoutToAliWarning(whatever) REDIRECTSTDOUT(AliLog::kWarning, 0, ClassName(), whatever)
#define StderrToAliWarning(whatever) REDIRECTSTDERR(AliLog::kWarning, 0, ClassName(), whatever)
#define ToAliWarning(whatever) REDIRECTSTDOUTANDSTDERR(AliLog::kWarning, 0, ClassName(), whatever)
#define StdoutToAliWarningClass(whatever) REDIRECTSTDOUT(AliLog::kWarning, 0, Class()->GetName(), whatever)
#define StderrToAliWarningClass(whatever) REDIRECTSTDERR(AliLog::kWarning, 0, Class()->GetName(), whatever)
#define ToAliWarningClass(whatever) REDIRECTSTDOUTANDSTDERR(AliLog::kWarning, 0, Class()->GetName(), whatever)
#define StdoutToAliWarningGeneral(scope, whatever) REDIRECTSTDOUT(AliLog::kWarning, 0, scope, whatever)
#define StderrToAliWarningGeneral(scope, whatever) REDIRECTSTDERR(AliLog::kWarning, 0, scope, whatever)
#define ToAliWarningGeneral(scope, whatever) REDIRECTSTDOUTANDSTDERR(AliLog::kWarning, 0, scope, whatever)

// warning stream objects
#define AliWarningStream() AliLog::Stream(AliLog::kWarning, 0, MODULENAME(), ClassName(), FUNCTIONNAME(), __FILE__, __LINE__)
#define AliWarningClassStream() AliLog::Stream(AliLog::kWarning, 0, MODULENAME(), Class()->GetName(), FUNCTIONNAME(), __FILE__, __LINE__)
#define AliWarningGeneralStream(scope) AliLog::Stream(AliLog::kWarning, 0, MODULENAME(), scope, FUNCTIONNAME(), __FILE__, __LINE__)


// error messages
/**
 * Forwards to AliMessage with log level of AliLog::kError
 * @see AliMessage 
 */
#define AliError(message)               AliMessage(AliLog::kError, message)
/**
 * Forwards to AliMessageClass with log level of AliLog::kError
 * @see AliMessageClass 
 */
#define AliErrorClass(message)          AliMessageClass(AliLog::kError, message)
/**
 * Forwards to AliMessageGeneral with log level of AliLog::kError
 * @see AliMessageGeneral
 */
#define AliErrorGeneral(scope, message) AliMessageGeneral(scope, AliLog::kError, message)
/**
 * Forwards to AliMessageF with log level of AliLog::kError
 * @see AliMessageF 
 */
#define AliErrorF(message,...)               AliMessageF(AliLog::kError, message, __VA_ARGS__)
/**
 * Forwards to AliMessageClassF with log level of AliLog::kError
 * @see AliMessageClassF 
 */
#define AliErrorClassF(message,...)          AliMessageClassF(AliLog::kError, message, __VA_ARGS__)
/**
 * Forwards to AliMessageGeneralF with log level of AliLog::kError
 * @see AliMessageGeneralF
 */
#define AliErrorGeneralF(scope,message,...)  AliMessageGeneralF(scope, AliLog::kError, message, __VA_ARGS__)

// redirection to error
#define StdoutToAliError(whatever) REDIRECTSTDOUT(AliLog::kError, 0, ClassName(), whatever)
#define StderrToAliError(whatever) REDIRECTSTDERR(AliLog::kError, 0, ClassName(), whatever)
#define ToAliError(whatever) REDIRECTSTDOUTANDSTDERR(AliLog::kError, 0, ClassName(), whatever)
#define StdoutToAliErrorClass(whatever) REDIRECTSTDOUT(AliLog::kError, 0, Class()->GetName(), whatever)
#define StderrToAliErrorClass(whatever) REDIRECTSTDERR(AliLog::kError, 0, Class()->GetName(), whatever)
#define ToAliErrorClass(whatever) REDIRECTSTDOUTANDSTDERR(AliLog::kError, 0, Class()->GetName(), whatever)
#define StdoutToAliErrorGeneral(scope, whatever) REDIRECTSTDOUT(AliLog::kError, 0, scope, whatever)
#define StderrToAliErrorGeneral(scope, whatever) REDIRECTSTDERR(AliLog::kError, 0, scope, whatever)
#define ToAliErrorGeneral(scope, whatever) REDIRECTSTDOUTANDSTDERR(AliLog::kError, 0, scope, whatever)

// error stream objects
#define AliErrorStream() AliLog::Stream(AliLog::kError, 0, MODULENAME(), ClassName(), FUNCTIONNAME(), __FILE__, __LINE__)
#define AliErrorClassStream() AliLog::Stream(AliLog::kError, 0, MODULENAME(), Class()->GetName(), FUNCTIONNAME(), __FILE__, __LINE__)
#define AliErrorGeneralStream(scope) AliLog::Stream(AliLog::kError, 0, MODULENAME(), scope, FUNCTIONNAME(), __FILE__, __LINE__)


// fatal messages
/**
 * Forwards to AliMessage with log level of AliLog::kFatal
 * @see AliMessage 
 */
#define AliFatal(message)               AliMessage(AliLog::kFatal, message)
/**
 * Forwards to AliMessageClass with log level of AliLog::kFatal
 * @see AliMessageClass 
 */
#define AliFatalClass(message)          AliMessageClass(AliLog::kFatal, message)
/**
 * Forwards to AliMessageGeneral with log level of AliLog::kFatal
 * @see AliMessageGeneral
 */
#define AliFatalGeneral(scope, message) AliMessageGeneral(scope, AliLog::kFatal, message)
/**
 * Forwards to AliMessageF with log level of AliLog::kFatal
 * @see AliMessageF 
 */
#define AliFatalF(message,...)               AliMessageF(AliLog::kFatal, message, __VA_ARGS__)
/**
 * Forwards to AliMessageClassF with log level of AliLog::kFatal
 * @see AliMessageClassF 
 */
#define AliFatalClassF(message,...)          AliMessageClassF(AliLog::kFatal, message, __VA_ARGS__)
/**
 * Forwards to AliMessageGeneralF with log level of AliLog::kFatal
 * @see AliMessageGeneralF
 */
#define AliFatalGeneralF(scope,message,...)  AliMessageGeneralF(scope, AliLog::kFatal, message, __VA_ARGS__)

#endif
