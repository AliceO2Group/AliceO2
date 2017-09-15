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

#define qon_mcat(a, b, c) a ## b ## c
#define qon_mxcat3(a, b, c) qon_mcat(a, b, c)
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

template <typename T> struct qConfigSettings
{
	qConfigSettings() : checkMin(false), checkMax(false), doSet(false), min(0), max(0), set(0), message(nullptr) {}
	bool checkMin, checkMax;
	bool doSet;
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
		if ((retVal = qAddOptionMinMax<T>(settings, ref, argv[iOrg]))) return(retVal);
		qAddOptionMessage<T>(settings, ref);
		return(0);
	}
	
	template <typename... Args> static inline int qAddOptionVec(std::vector<T>& ref, int& i, const char** argv, const int argc, const char* help, Args&&... args)
	{
		T tmp = 0;
		T def = 0;
		int retVal = qAddOption(tmp, i, argv, argc, def, help, args...);
		if (retVal) return(retVal);
		ref.push_back(tmp);
		return(0);
	}
};

inline const char* getArg(int& i, const char** argv, const int argc)
{
	if (i + 1 < argc && argv[i + 1][0] && argv[i + 1][0] != '-') return(argv[++i]);
	return(nullptr);
}

template <class T> inline int qAddOptionGeneric(T& ref, int& i, const char** argv, const int argc, T def, std::function<T(const char*)> func, bool allowDefault = false)
{
	const char* arg = getArg(i, argv, argc);
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
	return(1);
}

//Handling of all supported types
template <> inline int qAddOptionType<bool>(qConfigSettings<bool>& settings, bool& ref, int& i, const char** argv, const int argc, bool def, const char* help)
{
	return qAddOptionGeneric<bool>(ref, i, argv, argc, def, [](const char* a)->bool{return atoi(a);}, true);
}
template <> inline int qAddOptionType<int>(qConfigSettings<int>& settings, int& ref, int& i, const char** argv, const int argc, int def, const char* help)
{
	return qAddOptionGeneric<int>(ref, i, argv, argc, def, [](const char* a)->int{return atoi(a);});
}
template <> inline int qAddOptionType<unsigned int>(qConfigSettings<unsigned int>& settings, unsigned int& ref, int& i, const char** argv, const int argc, unsigned int def, const char* help)
{
	return qAddOptionGeneric<unsigned int>(ref, i, argv, argc, def, [](const char* a)->unsigned int{return strtoul(a, nullptr, 0);});
}
template <> inline int qAddOptionType<long long int>(qConfigSettings<long long int>& settings, long long int& ref, int& i, const char** argv, const int argc, long long int def, const char* help)
{
	return qAddOptionGeneric<long long int>(ref, i, argv, argc, def, [](const char* a)->long long int{return strtoll(a, nullptr, 0);});
}
template <> inline int qAddOptionType<unsigned long long int>(qConfigSettings<unsigned long long int>& settings, unsigned long long int& ref, int& i, const char** argv, const int argc, unsigned long long int def, const char* help)
{
	return qAddOptionGeneric<unsigned long long int>(ref, i, argv, argc, def, [](const char* a)->unsigned long long int{return strtoull(a, nullptr, 0);});
}
template <> inline int qAddOptionType<float>(qConfigSettings<float>& settings, float& ref, int& i, const char** argv, const int argc, float def, const char* help)
{
	return qAddOptionGeneric<float>(ref, i, argv, argc, def, [](const char* a)->float{return (float) atof(a);});
}
template <> inline int qAddOptionType<double>(qConfigSettings<double>& settings, double& ref, int& i, const char** argv, const int argc, double def, const char* help)
{
	return qAddOptionGeneric<double>(ref, i, argv, argc, def, [](const char* a)->double{return atof(a);});
}
template <> inline int qAddOptionType<const char*>(qConfigSettings<const char*>& settings, const char*& ref, int& i, const char** argv, const int argc, const char* def, const char* help)
{
	return qAddOptionGeneric<const char*>(ref, i, argv, argc, def, [](const char* a)->const char*{return a;});
}

//Checks and messages for additional settings
template <typename T> inline int qAddOptionMinMax(qConfigSettings<T>& settings, T& ref, const char* arg)
{
	if (settings.checkMin && ref < settings.min)
	{
		std::cout << "Invalid setting for " << arg << ": minimum threshold exceeded (" << ref << " < " << settings.min << ")\n";
		return(1);
	}
	if (settings.checkMax && ref > settings.max)
	{
		std::cout << "Invalid setting for " << arg << ": maximum threshold exceeded (" << ref << " > " << settings.max << ")\n";
		return(2);
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
