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
	} \
	
#define AddOptionSet(name, type, value, optname, optnameshort, ...) \
	else if (QCONFIG_COMPARE(optname, optnameshort)) \
	{ \
		qConfigType<type>::qAddOption(tmp.name, i, nullptr, 0, value, __VA_ARGS__, set(value)); \
	} \

#define AddOptionVec(name, type, optname, optnameshort, ...) \
	else if (QCONFIG_COMPARE(optname, optnameshort)) \
	{ \
		qConfigType<type>::qAddOptionVec(tmp.name, i, argv, argc, __VA_ARGS__); \
	} \

#define AddSubConfig(name, instance)

#define BeginConfig(name, instance) \
	{ \
		constexpr const char* preopt = ""; \
		constexpr const char preoptshort = 0; \
		name& tmp = instance; \
		bool tmpfound = true; \
		if (found); \

#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort) \
	{ \
		constexpr const char* preopt = preoptname; \
		constexpr const char preoptshort = preoptnameshort; \
		name& tmp = parent.instance; \
		bool tmpfound = true; \
		if (found); \

#define EndConfig() \
		else tmpfound = false; \
		if (tmpfound) found = true; \
	} \
	
#elif defined(QCONFIG_INSTANCE)
#define AddOption(name, type, default, optname, optnameshort, help, ...) name(default), 
#define AddOptionSet(name, type, optname, optnameshort, help, ...)
#define AddOptionVec(name, type, optname, optnameshort, help, ...) name(),
#define AddSubConfig(name, instance) instance(),
#define BeginConfig(name, instance) name instance; name::name() :
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort) name::name() :
#define EndConfig() _qConfigDummy() {}
#elif defined(QCONFIG_EXTERNS)
#define AddOption(name, type, default, optname, optnameshort, help, ...)
#define AddOptionSet(name, type, optname, optnameshort, help, ...)
#define AddOptionVec(name, type, optname, optnameshort, help, ...)
#define AddSubConfig(name, instance)
#define BeginConfig(name, instance) extern name instance;
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort)
#define EndConfig()
#undef QCONFIG_EXTERNS
extern int qConfigParse(int argc, const char** argv, const char* filename = NULL);
#else
#define AddOption(name, type, default, optname, optnameshort, help, ...) type name;
#define AddOptionSet(name, type, optname, optnameshort, help, ...)
#define AddOptionVec(name, type, optname, optnameshort, help, ...) std::vector<type> name;
#define AddSubConfig(name, instance) name instance;
#define BeginConfig(name, instance) struct name { name(); name(const name& s); name& operator =(const name& s);
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort) struct name { name(); name(const name& s); name& operator =(const name& s);
#define EndConfig() qConfigDummy _qConfigDummy; };
#define QCONFIG_EXTERNS
struct qConfigDummy{qConfigDummy() {}};
#endif

#include "qconfigoptions.h"

#undef AddOption
#undef AddOptionSet
#undef AddOptionVec
#undef AddSubConfig
#undef BeginConfig
#undef BeginSubConfig
#undef EndConfig
#ifdef QCONFIG_COMPARE
#undef QCONFIG_COMPARE
#endif

#ifdef QCONFIG_EXTERNS
#include "qconfig.h"
#endif
