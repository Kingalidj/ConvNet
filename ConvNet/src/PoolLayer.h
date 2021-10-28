#pragma once

#include "Layer.h"
#include "Tensor.h"

#include <Atlas/Core/Core.h>

namespace Compass
{

	class PoolLayer : public Layer
	{
	private:
		uint16_t m_Stride, m_KernelSize;

	public:

		PoolLayer(TensorSize inputSize, uint16_t stride, uint16_t kernelSize)
			: Layer(LayerType::POOL, inputSize, { (inputSize.Width - kernelSize) / stride + 1, (inputSize.Height - kernelSize) / stride + 1, inputSize.Depth }),
			m_Stride(stride), m_KernelSize(kernelSize)
		{
			ATL_ASSERT((float(inputSize.Width - kernelSize) / stride) == ((inputSize.Width - kernelSize) / stride), "Incorrect Dimesions");
			ATL_ASSERT((float(inputSize.Height - kernelSize) / stride) == ((inputSize.Height - kernelSize) / stride), "Incorrect Dimesions");
		}

		virtual void Activate(Tensor<float>& tensor) override
		{
			m_Input = tensor;
			Activate();
		}

		void Activate()
		{
			for (uint32_t x = 0; x < m_Output.GetWidth(); x++)
			{
				for (uint32_t y = 0; y < m_Output.GetHeight(); y++)
				{
					for (uint32_t z = 0; z < m_Output.GetDepth(); z++)
					{
						TensorPoint<float> mapped = { x * m_Stride, y * m_Stride, 0 };
						float max = -10000;

						for (uint32_t i = 0; i < m_KernelSize; i++)
						{
							for (uint32_t j = 0; j < m_KernelSize; j++)
							{
								float value = m_Input((uint32_t) mapped.x + i, (uint32_t) mapped.y + j, z);
								if (value > max) max = value;
							}
						}

						m_Output(x, y, z) = max;
					}
				}
			}
		}

		virtual void UpdateWeights() override
		{
		}

		TensorPoint<float> MapToInput(TensorPoint<float> output, int z)
		{
			output.x *= m_Stride;
			output.y *= m_Stride;
			output.z *= z;
			return output;
		}

		int NormalizeRange(float f, int max, bool limMin)
		{
			if (f <= 0) return 0;
			max--;
			if (f >= max) return max;

			if (limMin) return (int) ceil(f);
			else return (int) floor(f);
		}

		Range MapToOutput(uint32_t x, uint32_t y )
		{
			float a = (float) x;
			float b = (float) y;

			return
			{
				NormalizeRange((a - m_KernelSize + 1) / m_Stride, m_Output.GetWidth(), true),
				NormalizeRange((b - m_KernelSize + 1) / m_Stride, m_Output.GetHeight(), true),
				0,
				NormalizeRange(a / m_Stride, m_Output.GetWidth(), false),
				NormalizeRange(b / m_Stride, m_Output.GetHeight(), false),
				(int)m_Output.GetDepth() - 1
			};
		}

		virtual void ComputeGradient(Tensor<float>& nextLayer) override
		{
			for (uint32_t x = 0; x < m_Input.GetWidth(); x++)
			{
				for (uint32_t y = 0; y < m_Input.GetHeight(); y++)
				{
					Range range = MapToOutput(x, y);

					for (uint32_t z = 0; z < m_Input.GetDepth(); z++)
					{
						float sum = 0;
						for (int i = range.MinX; i <= range.MaxX; i++)
						{
							int minX = i * m_Stride;
							for (int j = range.MinY; j <= range.MaxY; j++)
							{
								int minY = j * m_Stride;

								bool isMax = m_Input(x, y, z) == m_Output(i, j, z) ? true : false;
								sum += isMax * nextLayer(i, j, z);
							}
						}
						m_Gradient(x, y, z) = sum;

					}
				}
			}
		}
	};

}

