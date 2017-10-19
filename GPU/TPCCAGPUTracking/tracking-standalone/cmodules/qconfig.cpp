#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <utility>
#include <vector>
#include <functional>
#include <iostream>
#include "qconfig.h"

//Create config instances
#define QCONFIG_INSTANCE
#include "qconfig.h"
#undef QCONFIG_INSTANCE

namespace qConfig {

#define qon_mcat(a, b) a ## b
#define qon_mxcat(a, b) qon_mcat(a, b)
#define qon_mcat3(a, b, c) a ## b ## c
#define qon_mxcat3(a, b, c) qon_mcat3(a, b, c)
#define qon_mstr(a) #a
#define qon_mxstr(a) qon_mstr(a)
#define QCONFIG_SETTING(name, type) \
	struct qon_mxcat3(q, name, _t) {type v; constexpr qon_mxcat3(q, name, _t)(type s) : v(s) {}}; \
	constexpr qon_mxcat3(q, name, _t) name(type v) {return(std::move(qon_mxcat3(q, name, _t)(v)));}

#define QCONFIG_SETTING_TEMPLATE(name) \
	template <typename T> struct qon_mxcat3(q, name, _t) {T v; constexpr qon_mxcat3(q, name, _t)(const T& s) : v(s) {}}; \
	template <typename T> constexpr qon_mxcat3(q, name, _t)<T> name(T v) {return(std::move(qon_mxcat3(q, name, _t)<T>(v)));}

QCONFIG_SETTING(message, const char*)
QCONFIG_SETTING_TEMPLATE(min)
QCONFIG_SETTING_TEMPLATE(max)
QCONFIG_SETTING_TEMPLATE(set)
QCONFIG_SETTING_TEMPLATE(def)

template <typename T> struct qConfigSettings
{
	qConfigSettings() : checkMin(false), checkMax(false), doSet(false), doDefault(false), min(0), max(0), set(0), message(nullptr) {}
	bool checkMin, checkMax;
	bool doSet, doDefault;
	T min, max;
	T set;
	const char* message;
};

template <typename T> int qAddOptionType(qConfigSettings<T>& settings, T& ref, int& i, const char** argv, const int argc, T def, const char* help);
template <typename T> void qAddOptionMessage(qConfigSettings<T>& settings, T& ref);
template <typename T> int qAddOptionMinMax(qConfigSettings<T>& settings, T& ref, const char* arg);

template <typename T> struct qConfigType
{
	//Recursive handling of additional settings
	static inline void qProcessSetting(qConfigSettings<T>& settings, qmin_t<T> minval)
	{
		static_assert(!std::is_same<T, bool>::value, "min option not supported for boolean settings");
		settings.checkMin = true;
		settings.min = minval.v;
	}
	static inline void qProcessSetting(qConfigSettings<T>& settings, qmax_t<T> maxval)
	{
		static_assert(!std::is_same<T, bool>::value, "max option not supported for boolean settings");
		settings.checkMax = true;
		settings.max = maxval.v;
	}
	static inline void qProcessSetting(qConfigSettings<T>& settings, qmessage_t msg)
	{
		settings.message = msg.v;
	}
	static inline void qProcessSetting(qConfigSettings<T>& settings, qset_t<T> set)
	{
		settings.doSet = true;
		settings.set = set.v;
	}
	static inline void qProcessSetting(qConfigSettings<T>& settings, qdef_t<T> set)
	{
		settings.doDefault = true;
		settings.set = set.v;
	}
	
	static inline void qAddOptionSettings(qConfigSettings<T>& settings) {}
	template <typename Arg1, typename... Args> static inline void qAddOptionSettings(qConfigSettings<T>& settings, Arg1&& arg1, Args&&... args)
	{
		qProcessSetting(settings, arg1);
		qAddOptionSettings(settings, args...);
	}
	
	//Main processing function for arguments
	template <typename... Args> static inline int qAddOption(T& ref, int& i, const char** argv, const int argc, T def, const char* help, Args&&... args)
	{
		qConfigSettings<T> settings;
		qAddOptionSettings(settings, args...);
		int retVal = 0;
		int iOrg = i;
		if (settings.doSet) ref = settings.set;
		else if ((retVal = qAddOptionType<T>(settings, ref, i, argv, argc, def, help))) return(retVal);
		if ((retVal = qAddOptionMinMax<T>(settings, ref, iOrg < argc ? argv[iOrg] : ""))) return(retVal);
		qAddOptionMessage<T>(settings, ref);
		return(0);
	}
	
	template <typename... Args> static inline int qAddOptionVec(std::vector<T>& ref, int& i, const char** argv, const int argc, const char* help, Args&&... args)
	{
		int iFirst = i, iLast;
		do
		{
			iLast = i;
			T tmp = 0;
			T def = 0;
			int retVal = qAddOption(tmp, i, argv, argc, def, help, args...);
			if (retVal) return(i == iFirst ? retVal : 0);
			if (i == iFirst || i != iLast) ref.push_back(tmp);
		} while (i != iLast);
		return(0);
	}
	
	template <typename... Args> static inline void qConfigHelpOption(const char* name, const char* type, const char* def, const char* optname, char optnameshort, const char* preopt, char preoptshort, int optionType, const char* help, Args&&... args)
	{
		qConfigSettings<T> settings;
		qAddOptionSettings(settings, args...);
		std::cout << "\t" << name << ": " << (optnameshort == 0 || preoptshort == 0 ? "" : "-") << (char) (optnameshort == 0 ? 0 : preoptshort == 0 ? '-' : preoptshort) << (char) (optnameshort == 0 ? 0 : optnameshort) <<
			(optnameshort == 0 ? "" : " (") << "--" << preopt << optname << (optnameshort == 0 ? " (" : ", ") << "type: " << type;
		if (optionType == 0) std::cout << ", default: " << def;
		if (optionType == 1) std::cout << ", sets " << name << " to " << def;
		if (settings.checkMin) std::cout << ", minimum: " << settings.min;
		if (settings.checkMax) std::cout << ", maximum: " << settings.max;
		std::cout << ")\n\t\t" << help << ".\n";
		if (settings.doDefault) std::cout << "\t\tIf no argument is supplied, " << name << " is set to " << settings.set << ".\n";
		else if (optionType != 1 && std::is_same<T, bool>::value) std::cout << "\t\tIf no argument is supplied, " << name << " is set to true.\n";
		if (optionType == 2) std::cout << "\t\tCan be set multiple times, or can accept multiple arguments.\n";
		std::cout << "\n";
	}
};

inline const char* getArg(int& i, const char** argv, const int argc, bool allowOption = false)
{
	if (i + 1 < argc && argv[i + 1][0] && (allowOption || argv[i + 1][0] != '-')) return(argv[++i]);
	return(nullptr);
}

template <class T> inline int qAddOptionGeneric(T& ref, int& i, const char** argv, const int argc, T def, std::function<T(const char*)> func, bool allowDefault = false)
{
	const char* arg = getArg(i, argv, argc, !allowDefault);
	if (arg)
	{
		ref = func(arg);
		return(0);
	}
	else if (allowDefault)
	{
		ref = def;
		return(0);
	}
	printf("Argument missing for option %s!\n", argv[i]);
	return(qcrArgMissing);
}

//Handling of all supported types
template <> inline int qAddOptionType<bool>(qConfigSettings<bool>& settings, bool& ref, int& i, const char** argv, const int argc, bool def, const char* help)
{
	return qAddOptionGeneric<bool>(ref, i, argv, argc, settings.doDefault ? settings.set : true, [](const char* a)->bool{return atoi(a);}, true);
}
template <> inline int qAddOptionType<int>(qConfigSettings<int>& settings, int& ref, int& i, const char** argv, const int argc, int def, const char* help)
{
	return qAddOptionGeneric<int>(ref, i, argv, argc, settings.set, [](const char* a)->int{return atoi(a);}, settings.doDefault);
}
template <> inline int qAddOptionType<unsigned int>(qConfigSettings<unsigned int>& settings, unsigned int& ref, int& i, const char** argv, const int argc, unsigned int def, const char* help)
{
	return qAddOptionGeneric<unsigned int>(ref, i, argv, argc, settings.set, [](const char* a)->unsigned int{return strtoul(a, nullptr, 0);}, settings.doDefault);
}
template <> inline int qAddOptionType<long long int>(qConfigSettings<long long int>& settings, long long int& ref, int& i, const char** argv, const int argc, long long int def, const char* help)
{
	return qAddOptionGeneric<long long int>(ref, i, argv, argc, settings.set, [](const char* a)->long long int{return strtoll(a, nullptr, 0);}, settings.doDefault);
}
template <> inline int qAddOptionType<unsigned long long int>(qConfigSettings<unsigned long long int>& settings, unsigned long long int& ref, int& i, const char** argv, const int argc, unsigned long long int def, const char* help)
{
	return qAddOptionGeneric<unsigned long long int>(ref, i, argv, argc, settings.set, [](const char* a)->unsigned long long int{return strtoull(a, nullptr, 0);}, settings.doDefault);
}
template <> inline int qAddOptionType<float>(qConfigSettings<float>& settings, float& ref, int& i, const char** argv, const int argc, float def, const char* help)
{
	return qAddOptionGeneric<float>(ref, i, argv, argc, settings.set, [](const char* a)->float{return (float) atof(a);}, settings.doDefault);
}
template <> inline int qAddOptionType<double>(qConfigSettings<double>& settings, double& ref, int& i, const char** argv, const int argc, double def, const char* help)
{
	return qAddOptionGeneric<double>(ref, i, argv, argc, settings.set, [](const char* a)->double{return atof(a);}, settings.doDefault);
}
template <> inline int qAddOptionType<const char*>(qConfigSettings<const char*>& settings, const char*& ref, int& i, const char** argv, const int argc, const char* def, const char* help)
{
	return qAddOptionGeneric<const char*>(ref, i, argv, argc, settings.set, [](const char* a)->const char*{return a;}, settings.doDefault);
}

//Checks and messages for additional settings
template <typename T> inline int qAddOptionMinMax(qConfigSettings<T>& settings, T& ref, const char* arg)
{
	if (settings.checkMin && ref < settings.min)
	{
		std::cout << "Invalid setting for " << arg << ": minimum threshold exceeded (" << ref << " < " << settings.min << ")!\n";
		return(qcrMinFailure);
	}
	if (settings.checkMax && ref > settings.max)
	{
		std::cout << "Invalid setting for " << arg << ": maximum threshold exceeded (" << ref << " > " << settings.max << ")!\n";
		return(qcrMaxFailure);
	}
	return(0);
}
template <> inline int qAddOptionMinMax<bool>(qConfigSettings<bool>& settings, bool& ref, const char* arg)
{
	return(0);
}

template <typename T> inline void qAddOptionMessage(qConfigSettings<T>& settings, T& ref)
{
	if (settings.message) {printf(settings.message, ref); printf("\n");}
}
template <> inline void qAddOptionMessage<bool>(qConfigSettings<bool>& settings, bool& ref)
{
	if (settings.message) {printf(settings.message, ref ? "ON" : "OFF"); printf("\n");}
}

void qConfigHelp(const char* subConfig = NULL, int followSub = 0)
{
	if (followSub < 2) printf("Usage Info:");
#define QCONFIG_HELP
#include "qconfig.h"
#undef QCONFIG_HELP
}

//Create parser for configuration
inline int qConfigParse(int argc, const char** argv, const char* filename)
{
	for (int i = 1;i < argc;i++)
	{
		bool found = false;
#define QCONFIG_PARSE
#include "qconfig.h"
#undef QCONFIG_PARSE
		if (found == false)
		{
			printf("Invalid argument: %s\n", argv[i]);
			return(1);
		}
	}
	return(0);
}

} //end namespace qConfig

//Main parse function called from outside
int qConfigParse(int argc, const char** argv, const char* filename)
{
	return(qConfig::qConfigParse(argc, argv, filename));
}
