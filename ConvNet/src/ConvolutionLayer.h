#pragma once

#include "Layer.h"

#include <vector>

namespace Compass
{

	class ConvolutionLayer : public Layer
	{
	private:

		std::vector<Tensor<float>> m_Kernels;
		std::vector<Tensor<Gradient>> m_KernelGradient;

		uint16_t m_Stride, m_KernelSize;

	public:
		ConvolutionLayer() = default;

		ConvolutionLayer(TensorSize inputSize, uint16_t stride, uint16_t kernelSize, uint16_t nKernels)
			: Layer(LayerType::CONVOLUTION, inputSize, { (inputSize.Width - kernelSize) / stride + 1, (inputSize.Height - kernelSize) / stride + 1, nKernels }),
			m_Stride(stride), m_KernelSize(kernelSize)
		{
			ATL_ASSERT((float(inputSize.Width - kernelSize) / stride) == ((inputSize.Width - kernelSize) / stride), "Incorrect Dimesions");
			ATL_ASSERT((float(inputSize.Height - kernelSize) / stride) == ((inputSize.Height - kernelSize) / stride), "Incorrect Dimesions");

			for (int i = 0; i < nKernels; i++)
			{
				int maxVal = kernelSize * kernelSize * inputSize.Depth;

				Tensor<float> kernel(kernelSize, kernelSize, inputSize.Depth, 0.0f);
				m_Kernels.push_back(kernel);
			}

			for (auto& kernel : m_Kernels)
			{
				RandomizeTensor<float>(kernel, 0, 1);
			}

			for (int i = 0; i < nKernels; i++)
			{
				Tensor<Gradient> grad(kernelSize, kernelSize, m_Input.GetDepth(), Gradient({ 0.0f, 0.0f }));
				m_KernelGradient.push_back(grad);
			}
		}

		virtual void Activate(Tensor<float>& tensor) override
		{
			m_Input = tensor;
			Activate();
		}

		void Activate() 
		{
			for (int n = 0; n < m_Kernels.size(); n++)
			{
				Tensor<float>& data = m_Kernels[n];

				for (uint32_t x = 0; x < m_Output.GetWidth(); x++)
				{
					for (uint32_t y = 0; y < m_Output.GetWidth(); y++)
					{
						TensorPoint<float> mapped = { x * m_Stride, y * m_Stride, 0 };
						float sum = 0;

						for (uint32_t i = 0; i < m_KernelSize; i++)
						{
							for (uint32_t j = 0; j < m_KernelSize; j++)
							{
								for (uint32_t z = 0; z < m_Input.GetDepth(); z++)
								{
									sum += data(i, j, z) * m_Input((uint32_t) mapped.x + i, (uint32_t) mapped.y + j, z);
								}
								m_Output(x, y, n) = sum;
							}
						}
					}
				}
			}
		}

		virtual void UpdateWeights() override
		{
			for (uint32_t a = 0; a < m_Kernels.size(); a++)
			{
				for (uint32_t x = 0; x < m_KernelSize; x++)
				{
					for (uint32_t y = 0; y < m_KernelSize; y++)
					{
						for (uint32_t z = 0; z < m_Input.GetDepth(); z++)
						{
							float& w = m_Kernels[a](x, y, z);
							Gradient& grad = m_KernelGradient[a].GetData(x, y, z);
							w = UpdateWeight(w, grad);
							UpdateGradient(grad);
						}
					}
				}
			}
		}

		int NormalizeRange(float f, int max, bool limMin)
		{
			if (f <= 0) return 0;
			max--;
			if (f >= max) return max;

			if (limMin) return ceil(f);
			else return floor(f);
		}

		Range MapToOutput(uint32_t x, uint32_t y )
		{
			float a = x;
			float b = y;

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
			for (uint32_t k = 0; k < m_KernelGradient.size(); k++)
			{
				for (uint32_t x = 0; x < m_KernelSize; x++)
				{
					for (uint32_t y = 0; y < m_KernelSize; y++)
					{
						for (uint32_t z = 0; z < m_Input.GetDepth(); z++)
						{
							m_KernelGradient[k].GetData(x, y, z).Grad = 0;
						}
					}
				}
			}

			for (uint32_t x = 0; x < m_Input.GetWidth(); x++)
			{
				for (uint32_t y = 0; y < m_Input.GetHeight(); y++)
				{
					Range range = MapToOutput(x, y);

					for (uint32_t z = 0; z < m_Input.GetDepth(); z++)
					{
						float sum = 0;
						for (int i = range.MinX; i < range.MaxX; i++)
						{
							int minX = i * m_Stride;
							for (int j = range.MinY; j < range.MaxY; j++)
							{
								int minY = j * m_Stride;

								for (int k = range.MinZ; k < range.MaxZ; k++)
								{
									int w = m_Kernels[k].GetData(x - minX, y - minY, z);
									sum += w * nextLayer.GetData(i, j, k);
									m_KernelGradient[k].GetData(x - minX, y - minY, z).Grad += m_Input.GetData(x, y, z) * nextLayer.GetData(i, j, k);
								}
							}
						}
						m_Gradient.GetData(x, y, z) = sum;
					}
				}
			}
		}

		std::vector<Tensor<float>>& GetKernels() { return m_Kernels; }

	};

}
