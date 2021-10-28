#pragma once

#include "ConvolutionLayer.h"
#include "PoolLayer.h"
#include "ReluLayer.h"
#include "FullyConnectedLayer.h"

namespace Compass
{

	class ConvNet
	{
	private:
		std::vector<std::shared_ptr<Layer>> m_Layers;

	public:
		ConvNet()
		{
			std::shared_ptr<ConvolutionLayer> layer1 = std::make_shared<ConvolutionLayer>(TensorSize({ 28, 28, 1}), 1, 5, 2);
			std::shared_ptr<ReluLayer> layer2 = std::make_shared<ReluLayer>(layer1->GetOutput().GetSize());
			std::shared_ptr<PoolLayer> layer3 = std::make_shared<PoolLayer>(layer2->GetOutput().GetSize(), 2, 2);
			std::shared_ptr<FullyConnectedLayer> layer4 = std::make_shared<FullyConnectedLayer>(layer3->GetOutput().GetSize(), 10);

			m_Layers.push_back(layer1);
			m_Layers.push_back(layer2);
			m_Layers.push_back(layer3);
			m_Layers.push_back(layer4);
		}

		std::vector<std::shared_ptr<Layer>>& GetLayers() { return m_Layers; }

		void Forward(Tensor<float>& data)
		{
			for (int i = 0; i < m_Layers.size(); i++)
			{
				if (i == 0) m_Layers[i]->Activate(data);
				else m_Layers[i]->Activate(m_Layers[i - 1]->GetOutput());
			}
		}
		
		void BackProp(Tensor<float>& gradient)
		{
			for (int i = m_Layers.size() - 1; i >= 0; i--)
			{
				if (i == m_Layers.size() - 1) m_Layers[i]->ComputeGradient(gradient);
				else m_Layers[i]->ComputeGradient(m_Layers[i + 1]->GetGradient());
			}
		}

		float Train(Tensor<float>& data, Tensor<float>& expected)
		{
			Forward(data);
			Tensor<float> grads = m_Layers.back()->GetOutput() - expected;
			BackProp(expected);

			for (auto& layer : m_Layers) layer->UpdateWeights();

			float error = 0;

			for (uint32_t i = 0; i < grads.GetWidth() * grads.GetHeight() * grads.GetDepth(); i++)
			{
				float f = expected.GetValuePointer()[i];
				if (f > 0.5)
				{
					error += abs(grads.GetValuePointer()[i]);
				}
			}

			return error * 100;
		}
	};

}
