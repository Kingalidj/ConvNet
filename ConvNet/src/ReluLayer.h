#pragma once

#include "Layer.h"

namespace Compass
{
	class ReluLayer : public Layer
	{
	private:

	public:
		ReluLayer(TensorSize size)
			: Layer(LayerType::RELU, size, size)
		{
		}

		virtual void Activate(std::shared_ptr<Tensor<float>> tensor) override
		{
			m_Input = tensor;
			Activate();
		}

		void Activate() 
		{
			for (uint32_t x = 0; x < m_Input->GetWidth(); x++)
			{
				for (uint32_t y = 0; y < m_Input->GetHeight(); y++)
				{
					for (uint32_t z = 0; z < m_Input->GetDepth(); z++)
					{
						float value = (*m_Input)(x, y, z);
						if (value < 0) value = 0;
						(*m_Output)(x, y, z) = value;
					}
				}
			}
		}

		virtual void UpdateWeights() override
		{
		}

		virtual void ComputeGradient(std::shared_ptr<Tensor<float>> nextLayer) override
		{
			for (uint32_t x = 0; x < m_Input->GetHeight(); x++)
			{
				for (uint32_t y = 0; y < m_Input->GetWidth(); y++)
				{
					for (int z = 0; z < m_Input->GetDepth(); z++)
					{
						(*m_Gradient)(x, y, z) = ((*m_Input)(x, y, z) < 0) ? 0 : (*nextLayer)(x, y, z);
					}
				}
			}
		}

	};
}
