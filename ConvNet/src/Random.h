#pragma once

#include <cstdint>

namespace Compass
{
	class Random
	{
	private:

	public:
		static void Init();
		static uint32_t RandUInt();
		static int RandInt();
		static double RandDouble();
		static int RandRangeInt(int min, int max);
		static double RandRangeDouble(double min, double max);
	};
}
