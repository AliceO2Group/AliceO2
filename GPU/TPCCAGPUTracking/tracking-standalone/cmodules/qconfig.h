#ifdef QCONFIG_PARSE

	#define QCONFIG_COMPARE(optname, optnameshort) (thisoption[0] == '-' && ( \
		(preoptshort == 0 && thisoption[1] == optnameshort && thisoption[2] == 0) || \
		(thisoption[1] == '-' && strlen(preopt) == 0 && strcmp(&thisoption[2], optname) == 0) || \
		(preoptshort != 0 && thisoption[1] == preoptshort && thisoption[2] == optnameshort && thisoption[3] == 0) || \
		(thisoption[1] == '-' && strlen(preopt) && strncmp(&thisoption[2], preopt, strlen(preopt)) == 0 && strcmp(&thisoption[2 + strlen(preopt)], optname) == 0) \
		))

	#define AddOption(name, type, default, optname, optnameshort, ...) \
		else if (QCONFIG_COMPARE(optname, optnameshort)) \
		{ \
			int retVal = qConfigType<type>::qAddOption(tmp.name, i, argv, argc, default, __VA_ARGS__); \
			if (retVal) return(retVal); \
		}
		
	#define AddOptionSet(name, type, value, optname, optnameshort, ...) \
		else if (QCONFIG_COMPARE(optname, optnameshort)) \
		{ \
			int retVal = qConfigType<type>::qAddOption(tmp.name, i, nullptr, 0, value, __VA_ARGS__, set(value)); \
			if (retVal) return(retVal); \
		}

	#define AddOptionVec(name, type, optname, optnameshort, ...) \
		else if (QCONFIG_COMPARE(optname, optnameshort)) \
		{ \
			int retVal = qConfigType<type>::qAddOptionVec(tmp.name, i, argv, argc, __VA_ARGS__); \
			if (retVal) return(retVal); \
		}

	#define AddSubConfig(name, instance)

	#define BeginConfig(name, instance) \
		{ \
			constexpr const char* preopt = ""; \
			constexpr const char preoptshort = 0; \
			name& tmp = instance; \
			bool tmpfound = true; \
			if (found);

	#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr) \
		{ \
			constexpr const char* preopt = preoptname; \
			constexpr const char preoptshort = preoptnameshort; \
			name& tmp = parent.instance; \
			bool tmpfound = true; \
			if (found);

	#define EndConfig() \
			else tmpfound = false; \
			if (tmpfound) found = true; \
		}

	#define AddHelp(cmd, cmdshort) \
		else if (QCONFIG_COMPARE(cmd, cmdshort)) \
		{ \
			const char* arg = getArg(i, argv, argc); \
			qConfigHelp(arg ? arg : preopt); \
			return(qcrHelp); \
		}

	#define AddHelpAll(cmd, cmdshort) \
		else if (QCONFIG_COMPARE(cmd, cmdshort)) \
		{ \
			const char* arg = getArg(i, argv, argc); \
			qConfigHelp(arg ? arg : "", true); \
			return(qcrHelp); \
		}

	#define AddCommand(cmd, cmdshort, command, help) \
		else if (QCONFIG_COMPARE(cmd, cmdshort)) \
		{ \
			if (command) return(qcrCmd); \
		}
		
	#define AddShortcut(cmd, cmdshort, forward, help, ...) \
		else if (QCONFIG_COMPARE(cmd, cmdshort)) \
		{ \
			const char* options[] = {"", __VA_ARGS__, NULL}; \
			const int nOptions = sizeof(options) / sizeof(options[0]) - 1; \
			qConfigParse(nOptions, options, NULL); \
			thisoption = forward; \
			goto repeat; \
		}
	
#elif defined(QCONFIG_HELP)
	#define AddOption(name, type, default, optname, optnameshort, ...) qConfigType<type>::qConfigHelpOption(qon_mxstr(name), qon_mxstr(type), qon_mxstr(default), optname, optnameshort, preopt, preoptshort, 0, __VA_ARGS__);
	#define AddOptionSet(name, type, value, optname, optnameshort, ...) qConfigType<type>::qConfigHelpOption(qon_mxstr(name), qon_mxstr(type), qon_mxstr(value), optname, optnameshort, preopt, preoptshort, 1, __VA_ARGS__);
	#define AddOptionVec(name, type, optname, optnameshort, ...) qConfigType<type>::qConfigHelpOption(qon_mxstr(name), qon_mxstr(type), NULL, optname, optnameshort, preopt, preoptshort, 2, __VA_ARGS__);
	#define AddSubConfig(name, instance) printf("\t%s\n\n", qon_mxcat(qConfig_subconfig_, name)); \
		if (followSub) qConfigHelp(qon_mxstr(name), 2);
	#define BeginConfig(name, instance) if (subConfig == NULL || *subConfig == 0) { \
		constexpr const char* preopt = ""; \
		constexpr const char preoptshort = 0; \
		printf("\n");
	#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr) \
		const char* qon_mxcat(qConfig_subconfig_, name) = preoptnameshort == 0 ? (qon_mxstr(name) ": --" preoptname "\n\t\t" descr) : (qon_mxstr(name) ": -" qon_mxstr('a') " (--" preoptname ")\n\t\t" descr); \
		if (subConfig == NULL || strcmp(subConfig, followSub == 2 ? qon_mxstr(name) : preoptname) == 0) { \
			constexpr const char* preopt = preoptname; \
			constexpr const char preoptshort = preoptnameshort; \
			printf("\n  %s: (--%s%s%c)\n", descr, preoptname, preoptnameshort == 0 ? "" : " or -", (int) preoptnameshort);
	#define EndConfig() }
	#define AddHelp(cmd, cmdshort) qConfigType<void*>::qConfigHelpOption("help", "help", NULL, cmd, cmdshort, preopt, preoptshort, 3, "Show usage information");
	#define AddHelpAll(cmd, cmdshort) qConfigType<void*>::qConfigHelpOption("help all", "help all", NULL, cmd, cmdshort, preopt, preoptshort, 3, "Show usage info including all subparameters");
	#define AddCommand(cmd, cmdshort, command, help) qConfigType<void*>::qConfigHelpOption("command", "command", NULL, cmd, cmdshort, preopt, preoptshort, 4, help);
	#define AddShortcut(cmd, cmdshort, forward, help, ...) qConfigType<void*>::qConfigHelpOption("shortcut", "shortcut", NULL, cmd, cmdshort, preopt, preoptshort, 4, help);

#elif defined(QCONFIG_PRINT)
	#define AddOption(name, type, default, optname, optnameshort, ...) std::cout << "\t" << qon_mxstr(name) << ": " << tmp.name << "\n";
	#define AddOptionSet(name, type, value, optname, optnameshort, ...) 
	#define AddOptionVec(name, type, optname, optnameshort, ...) 
	#define AddSubConfig(name, instance)
	#define BeginConfig(name, instance) { name& tmp = instance;
	#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr) { name& tmp = parent.instance;
	#define EndConfig() }

#elif defined(QCONFIG_INSTANCE)
	#define AddOption(name, type, default, optname, optnameshort, help, ...) name(default), 
	#define AddVariable(name, type, default) name(default),
	#define AddOptionSet(name, type, value, optname, optnameshort, help, ...)
	#define AddOptionVec(name, type, optname, optnameshort, help, ...) name(),
	#define AddSubConfig(name, instance) instance(),
	#define BeginConfig(name, instance) name instance; name::name() :
	#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr) name::name() :
	#define EndConfig() _qConfigDummy() {}

#elif defined(QCONFIG_EXTERNS)
	#define AddOption(name, type, default, optname, optnameshort, help, ...)
	#define AddOptionSet(name, type, value, optname, optnameshort, help, ...)
	#define AddOptionVec(name, type, optname, optnameshort, help, ...)
	#define AddSubConfig(name, instance)
	#define BeginConfig(name, instance) extern "C" name instance;
	#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr)
	#define EndConfig()
	#undef QCONFIG_EXTERNS
	extern int qConfigParse(int argc, const char** argv, const char* filename = NULL);
	extern void qConfigPrint();
	namespace qConfig {enum qConfigRetVal {qcrOK = 0, qcrError = 1, qcrMinFailure = 2, qcrMaxFailure = 3, qcrHelp = 4, qcrCmd = 5, qcrArgMissing = 6, qcrArgIncomplete = 7};}

#else
	#ifdef QCONFIG_HEADER_GUARD
		#define QCONFIG_HEADER_GUARD_NO_INCLUDE
	#else
		#define QCONFIG_HEADER_GUARD
		#define AddOption(name, type, default, optname, optnameshort, help, ...) type name;
		#define AddVariable(name, type, default) type name;
		#define AddOptionSet(name, type, value, optname, optnameshort, help, ...)
		#define AddOptionVec(name, type, optname, optnameshort, help, ...) std::vector<type> name;
		#define AddSubConfig(name, instance) name instance;
		#ifdef QCONFIG_GPU
			#define BeginConfig(name, instance) struct name {
			#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr) struct name {
			struct qConfigDummy{};
		#else
			#define BeginConfig(name, instance) struct name { name(); name(const name& s); name& operator =(const name& s);
			#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr) struct name { name(); name(const name& s); name& operator =(const name& s);
			struct qConfigDummy{qConfigDummy() {}};
			#define QCONFIG_EXTERNS
		#endif
		#define EndConfig() qConfigDummy _qConfigDummy; };
	#endif
#endif

#ifndef AddHelp
	#define AddHelp(cmd, cmdshort)
#endif
#ifndef AddHelpAll
	#define AddHelpAll(cmd, cmdshort)
#endif
#ifndef AddCommand
	#define AddCommand(cmd, cmdshort, command)
#endif
#ifndef AddShortcut
	#define AddShortcut(cmd, cmdshort, forward, help, ...)
#endif
#ifndef AddVariable
	#define AddVariable(name, type, default)
#endif

#ifndef QCONFIG_HEADER_GUARD_NO_INCLUDE
	#include "qconfigoptions.h"
#endif

#undef AddOption
#undef AddVariable
#undef AddOptionSet
#undef AddOptionVec
#undef AddSubConfig
#undef BeginConfig
#undef BeginSubConfig
#undef EndConfig
#undef AddHelp
#undef AddHelpAll
#undef AddCommand
#undef AddShortcut
#ifdef QCONFIG_COMPARE
#undef QCONFIG_COMPARE
#endif

#ifdef QCONFIG_EXTERNS
	#include "qconfig.h"
#endif
