#pragma once

#include "Layer.h"

namespace Compass
{


	class FullyConnectedLayer : public Layer
	{
	private:
		Tensor<float> m_Weights;
		Tensor<float> m_InputVector;
		Tensor<Gradient> m_GradientVector;

		float ActivationFunc(float value)
		{
			return (1.0f / (1.0f + std::exp(-value)));
		}

		float DerivativeFunc(float value)
		{
			float sig = (1.0f / (1.0f + std::exp(-value)));
			return sig * (1 - sig);
		}


	public:
		FullyConnectedLayer(TensorSize inputSize, uint32_t outputSize)
			: Layer(LayerType::FULLY_CONNECTED, inputSize, { outputSize, 1, 1 }),
			m_Weights(Tensor<float>(inputSize.Width * inputSize.Height * inputSize.Depth, outputSize, 1))
		{
			m_InputVector = Tensor<float>(outputSize, 1, 1);
			m_GradientVector = Tensor<Gradient>(outputSize, 1, 1);

			int maxVal = inputSize.Width * inputSize.Height * inputSize.Depth;
			RandomizeTensor(m_InputVector, 0.0f, 2.19722f / maxVal);
		}

		virtual void Activate(Tensor<float>& tensor) override
		{
			m_Input = tensor;
			Activate();
		}

		void Activate()
		{
			for (uint32_t n = 0; n < m_Output.GetWidth(); n++)
			{
				float value = 0;

				for (uint32_t x = 0; x < m_Input.GetWidth(); x++)
				{
					for (uint32_t y = 0; y < m_Input.GetHeight(); y++)
					{
						for (uint32_t z = 0; z < m_Input.GetDepth(); z++)
						{
							int w = z * (m_Input.GetWidth() * m_Input.GetHeight()) + y * m_Input.GetWidth() + x;
							value += m_Input(x, y, z) * m_Weights(w, n, 0);
						}
					}	
				}
				m_InputVector(n, 0, 0) = value;

				m_Output(n, 0, 0) = ActivationFunc(value);
			}
		}

		virtual void UpdateWeights()
		{
			for (uint32_t n = 0; n < m_Output.GetWidth(); n++)
			{
				auto& grad = m_GradientVector(n, 0, 0);

				for (uint32_t x = 0; x < m_Input.GetWidth(); x++)
				{
					for (uint32_t y = 0; y < m_Input.GetHeight(); y++)
					{
						for (uint32_t z = 0; z < m_Input.GetDepth(); z++)
						{
							int w = z * (m_Input.GetWidth() * m_Input.GetHeight()) + y * m_Input.GetWidth() + x;
							float& weight = m_Weights(w, n, 0);

							weight = Compass::UpdateWeight(weight, grad, m_Input(x, y, z));
						}
					}
				}

				Compass::UpdateGradient(grad);
			}
		}

		virtual void ComputeGradient(Tensor<float>& nextLayer)
		{
			m_Gradient.Fill(0);

			for (uint32_t n = 0; n < m_Output.GetWidth(); n++)
			{
				auto& grad = m_GradientVector(n, 0, 0);
				grad.Grad = nextLayer(n, 0, 0) * DerivativeFunc(m_InputVector(n, 0, 0));

				for (uint32_t x = 0; x < m_Input.GetWidth(); x++)
				{
					for (uint32_t y = 0; y < m_Input.GetHeight(); y++)
					{
						for (uint32_t z = 0; z < m_Input.GetDepth(); z++)
						{
							int w = z * (m_Input.GetWidth() * m_Input.GetHeight()) + y * m_Input.GetWidth() + x;
							m_Gradient(x, y, z) += grad.Grad * m_Weights(w, n, 0);
						}
					}
				}
			}
		}

	};

}
