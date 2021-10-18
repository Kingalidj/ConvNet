#pragma once

#include <Atlas.h>
#include "TensorFrameBuffer.h"
#include "ConvNet.h"

class DrawLayer : public Atlas::Layer
{
private:
	Compass::ConvNet m_CNN;
	std::vector<Compass::TensorFrameBuffer> m_TensorFramebuffers;

	std::vector<int> m_Lables;
	std::vector<std::shared_ptr<Compass::Tensor<float>>> m_TrainImages;
	std::vector<int> m_TrainLabels;

	int m_TrainCount = 0;
	//Compass::PoolLayer m_Layer;
	//std::vector<Compass::TensorFrameBuffer<float>> m_Textures;
	//std::shared_ptr<Compass::Tensor<float>> m_Input = std::make_shared<Compass::Tensor<float>>(28, 28, 1);

public:

	DrawLayer();
	virtual void OnAttach() override;
	virtual void OnDetach() override;

	void OnUpdate(Atlas::Timestep ts) override;
	virtual void OnImGuiRender() override;
	void OnEvent(Atlas::Event& e) override;
};

