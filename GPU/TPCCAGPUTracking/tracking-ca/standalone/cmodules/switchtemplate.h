#ifndef QSWITCHTEMPLATE_H
#define QSWITCHTEMPLATE_H
#define Q_SWITCH_TEMPLATE_BOOL(expr, varname, ...) \
	{ \
		if (expr) \
		{ \
			const int varname = 1; \
			__VA_ARGS__; \
		} \
		else \
		{ \
			const int varname = 0; \
			__VA_ARGS__; \
		} \
	}

#define Q_SWITCH_TEMPLATE_CASE4(val, varname, ...) \
	switch (val) \
	{ \
	case 0: \
	{ \
		const int varname = 0; \
		__VA_ARGS__; \
		break; \
	} \
	case 1: \
	{ \
		const int varname = 1; \
		__VA_ARGS__; \
		break; \
	} \
	case 2: \
	{ \
		const int varname = 2; \
		__VA_ARGS__; \
		break; \
	} \
	case 3: \
	{ \
		const int varname = 3; \
		__VA_ARGS__; \
		break; \
	} \
	}


#endif