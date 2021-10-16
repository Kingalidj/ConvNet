#pragma once

#include "layer.h"

#include <vector>

namespace Compass
{

	template<typename T>
	class ConvolutionLayer : public Layer<T>
	{
	private:
		struct TensorPoint
		{
			T x, y, z;
		};

		std::vector<Tensor<T>> m_Kernels;
		uint16_t m_Stride, m_ExtendFilter, m_FilterSize;


	public:
		ConvolutionLayer() = default;

		ConvolutionLayer(TensorSize inputSize, uint16_t stride, uint16_t kernelSize, uint16_t nKernels)
			: Layer(LayerType::CONV, inputSize, { (inputSize.Width - kernelSize) / stride + 1, (inputSize.Height - kernelSize) / stride + 1, nKernels }),
			m_Stride(stride), m_ExtendFilter(kernelSize), m_FilterSize(kernelSize)
		{
			ATL_ASSERT((float(inputSize.Width - kernelSize) / stride) == ((inputSize.Width - kernelSize) / stride), "Incorrect Dimesions");
			ATL_ASSERT((float(inputSize.Height - kernelSize) / stride) == ((inputSize.Height - kernelSize) / stride), "Incorrect Dimesions");

			for (int i = 0; i < nKernels; i++)
			{
				int maxVal = kernelSize * kernelSize * inputSize.Depth;

				Tensor<T> kernel(kernelSize, kernelSize, inputSize.Depth);
				m_Kernels.push_back(kernel);
			}

			for (auto& kernel : m_Kernels)
			{
				RandomizeTensor<T>(kernel, -1, 1);
			}
		}

		void Activate(std::shared_ptr<Tensor<T>> tensor)
		{
			m_Input = tensor;
			Activate();
		}

		void Activate()
		{
			for (int n = 0; n < m_Kernels.size(); n++)
			{
				Tensor<T>& data = m_Kernels[n];

				for (uint32_t x = 0; x < m_Output->GetWidth(); x++)
				{
					for (uint32_t y = 0; y < m_Output->GetWidth(); y++)
					{
						TensorPoint mapped = { (T)x * m_Stride, (T)y * m_Stride, 0 };
						T sum = 0;

						for (uint32_t i = 0; i < m_FilterSize; i++)
						{
							for (uint32_t j = 0; j < m_FilterSize; j++)
							{
								for (uint32_t z = 0; z < m_Input->GetDepth(); z++)
								{
									sum += data(i, j, z) * (*m_Input)((uint32_t) mapped.x + i, (uint32_t) mapped.y + j, z);
								}
								(*m_Output)(x, y, n) = sum;
							}
						}
					}
				}
			}
		}

		std::vector<Tensor<T>>& GetKernels() { return m_Kernels; }
		const std::shared_ptr<Tensor<T>>& GetInput() { return m_Input; }
		const std::shared_ptr<Tensor<T>>& GetOutput() { return m_Output; }

	};

}
