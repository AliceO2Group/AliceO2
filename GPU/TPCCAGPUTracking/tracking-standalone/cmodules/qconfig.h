#ifdef QCONFIG_PARSE

#define QCONFIG_COMPARE(optname, optnameshort) (argv[i][0] == '-' && ( \
	(preoptshort == 0 && argv[i][1] == optnameshort && argv[i][2] == 0) || \
	(argv[i][1] == '-' && strlen(preopt) == 0 && strcmp(&argv[i][2], optname) == 0) || \
	(preoptshort != 0 && argv[i][1] == preoptshort && argv[i][2] == optnameshort && argv[i][3] == 0) || \
	(argv[i][1] == '-' && strlen(preopt) && strncmp(&argv[i][2], preopt, strlen(preopt)) == 0 && strcmp(&argv[i][2 + strlen(preopt)], optname) == 0) \
	))

#define AddOption(name, type, default, optname, optnameshort, ...) \
	else if (QCONFIG_COMPARE(optname, optnameshort)) \
	{ \
		qConfigType<type>::qAddOption(tmp.name, i, argv, argc, default, __VA_ARGS__); \
	}
	
#define AddOptionSet(name, type, value, optname, optnameshort, ...) \
	else if (QCONFIG_COMPARE(optname, optnameshort)) \
	{ \
		qConfigType<type>::qAddOption(tmp.name, i, nullptr, 0, value, __VA_ARGS__, set(value)); \
	}

#define AddOptionVec(name, type, optname, optnameshort, ...) \
	else if (QCONFIG_COMPARE(optname, optnameshort)) \
	{ \
		qConfigType<type>::qAddOptionVec(tmp.name, i, argv, argc, __VA_ARGS__); \
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
		if (qConfigHelp(arg ? arg : preopt)) return(3); \
	}

#define AddHelpAll(cmd, cmdshort) \
	else if (QCONFIG_COMPARE(cmd, cmdshort)) \
	{ \
		const char* arg = getArg(i, argv, argc); \
		if (qConfigHelp(arg ? arg : "", true)) return(3); \
	}

#define AddCommand(cmd, cmdshort, command, help) \
	else if (QCONFIG_COMPARE(cmd, cmdshort)) \
	{ \
		if (command) return(4); \
	}
#elif defined(QCONFIG_HELP)
#define AddOption(name, type, default, optname, optnameshort, help, ...) printf("\t%s: %s%c%c%s--%s%s, %stype: " qon_mxstr(type) ", default: " qon_mxstr(default) ")\n\t\t%s\n\n", \
	qon_mxstr(name), optnameshort == 0 || preoptshort == 0 ? "" : "-", (int) (optnameshort == 0 ? 0 : preoptshort == 0 ? '-' : preoptshort), (int) (optnameshort == 0 ? 0 : optnameshort), optnameshort == 0 ? "" : " (", preopt, optname, optnameshort == 0 ? "(" : "", help);
#define AddOptionSet(name, type, optname, optnameshort, help, ...) //AddOption(name, type, 0, optname, optnameshort, help, __VA_ARGS__)
#define AddOptionVec(name, type, optname, optnameshort, help, ...) AddOption(name, type, 0, optname, optnameshort, help, __VA_ARGS__)
#define AddSubConfig(name, instance) printf("\t%s\n\n", qon_mxcat(qConfig_subconfig_, name));
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
#define AddHelp(cmd, cmdshort) AddOption(help, help, no, cmd, cmdshort, "Show usage info")
#define AddHelpAll(cmd, cmdshort) AddOption(help, help, no, cmd, cmdshort, "Show usage info including all subparameters")
#define AddCommand(cmd, cmdshort, command, help) AddOption(command, command, no, cmd, cmdshort, help)
#elif defined(QCONFIG_INSTANCE)
#define AddOption(name, type, default, optname, optnameshort, help, ...) name(default), 
#define AddOptionSet(name, type, optname, optnameshort, help, ...)
#define AddOptionVec(name, type, optname, optnameshort, help, ...) name(),
#define AddSubConfig(name, instance) instance(),
#define BeginConfig(name, instance) name instance; name::name() :
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr) name::name() :
#define EndConfig() _qConfigDummy() {}
#elif defined(QCONFIG_EXTERNS)
#define AddOption(name, type, default, optname, optnameshort, help, ...)
#define AddOptionSet(name, type, optname, optnameshort, help, ...)
#define AddOptionVec(name, type, optname, optnameshort, help, ...)
#define AddSubConfig(name, instance)
#define BeginConfig(name, instance) extern name instance;
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr)
#define EndConfig()
#undef QCONFIG_EXTERNS
extern int qConfigParse(int argc, const char** argv, const char* filename = NULL);
#else
#define AddOption(name, type, default, optname, optnameshort, help, ...) type name;
#define AddOptionSet(name, type, optname, optnameshort, help, ...)
#define AddOptionVec(name, type, optname, optnameshort, help, ...) std::vector<type> name;
#define AddSubConfig(name, instance) name instance;
#define BeginConfig(name, instance) struct name { name(); name(const name& s); name& operator =(const name& s);
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr) struct name { name(); name(const name& s); name& operator =(const name& s);
#define EndConfig() qConfigDummy _qConfigDummy; };
#define QCONFIG_EXTERNS
struct qConfigDummy{qConfigDummy() {}};
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

#include "qconfigoptions.h"

#undef AddOption
#undef AddOptionSet
#undef AddOptionVec
#undef AddSubConfig
#undef BeginConfig
#undef BeginSubConfig
#undef EndConfig
#undef AddHelp
#undef AddHelpAll
#undef AddCommand
#ifdef QCONFIG_COMPARE
#undef QCONFIG_COMPARE
#endif

#ifdef QCONFIG_EXTERNS
#include "qconfig.h"
#endif
