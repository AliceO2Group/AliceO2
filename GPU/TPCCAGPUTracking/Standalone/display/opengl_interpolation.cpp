#include <stdio.h>
#include "opengl_backend.h"

void opengl_spline::create(const vecpod<float>& x, const vecpod<float>& y)
{
	fa.clear();
	fb.clear();
	fc.clear();
	fd.clear();
	fx.clear();
	if (x.size() != y.size() || x.size() < 2) return;
	int k = x.size() - 1;
	if (verbose) for (unsigned int i = 0;i < x.size();i++) printf("Point %d: %f --> %f\n", i, x[i], y[i]);
	fa.resize(k + 1);
	fb.resize(k + 1);
	fc.resize(k + 1);
	fd.resize(k + 1);
	fx.resize(k + 1);
	vecpod<float> h(k + 1), alpha(k + 1), l(k + 1), mu(k + 1), z(k + 1);
	for (int i = 0;i <= k;i++) fa[i] = y[i];
	for (int i = 0;i < k;i++) h[i] = x[i + 1] - x[i];
	for (int i = 1;i < k;i++) alpha[i] = 3.f / h[i] * (fa[i + 1] - fa[i]) - 3.f / h[i - 1] * (fa[i] - fa[i - 1]);
	l[0] = l[k] = 1;
	mu[0] = z[0] = z[k] = fc[k] = 0;
	for (int i = 1;i < k;i++)
	{
		l[i] = 2.f * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
		mu[i] = h[i] / l[i];
		z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
	}
	for (int i = k - 1;i >= 0;i--)
	{
		fc[i] = z[i] - mu[i] * fc[i + 1];
		fb[i] = (fa[i + 1] - fa[i]) / h[i] - h[i] / 3.f * (fc[i + 1] + 2.f * fc[i]);
		fd[i] = (fc[i + 1] - fc[i]) / (3.f * h[i]);
	}
	for (int i = 0;i <= k;i++) fx[i] = x[i];
}

float opengl_spline::evaluate(float x)
{
	int base = 0;
	const int k = fx.size() - 1;
	if (k < 0) return(0);
	while (base < k - 1 && x > fx[base + 1]) base++;
	float retVal = fa[base];
	x -= fx[base];
	const float xx = x;
	retVal += fb[base] * x;
	x *= xx;
	retVal += fc[base] * x;
	x *= xx;
	retVal += fd[base] * x;
	if (verbose) printf("Evaluate: %f --> %f (basepoint %d)\n", xx, retVal, base);
	return(retVal);
}
