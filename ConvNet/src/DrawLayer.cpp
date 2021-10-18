#include "DrawLayer.h"

#include <ImGui/imgui.h>

#include <fstream>

using namespace Atlas;

uint8_t* ReadFile(const char* filepath)
{
	std::ifstream file(filepath, std::ios::binary | std::ios::ate);
	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);

	if (size == -1)
		return nullptr;

	uint8_t* buffer = new uint8_t[size];
	file.read((char*)buffer, size);
	return buffer;
}

DrawLayer::DrawLayer()
	//: m_Layer(Compass::PoolLayer({ 28, 28, 1 }, 1, 5))
	: m_CNN()
{
	uint8_t* images = ReadFile("assets/train-images.idx3-ubyte");
	uint8_t* trainLabels = ReadFile("assets/train-labels.idx1-ubyte");

	for (int i = 0; i < 100; i++)
	{
		std::shared_ptr<Compass::Tensor<float>> input = std::make_shared<Compass::Tensor<float>>(28, 28, 1);
	  //case_t c{ tensor_t<float>(28, 28, 1), tensor_t<float>(10, 1, 1) };

		uint8_t* img = images + 16 + i * (28 * 28);
		uint8_t* label = trainLabels + 8 + i;

		for (int x = 0; x < 28; x++)
			for (int y = 0; y < 28; y++)
	  		input->GetData(x, y, 0) = img[x + y * 28] / 255.0f;
	  		//c.data(x, y, 0) = img[x + y * 28] / 255.f;

		m_TrainImages.push_back(input);
		m_TrainLabels.push_back(*label);

	  //for (int b = 0; b < 10; b++)
	  	//c.out(b, 0, 0) = *label == b ? 1.0f : 0.0f;

	  //cases.push_back(c);
	}

	//m_TensorFramebuffers.push_back({ *(m_CNN.GetLayers()[3]->GetOutput()) });

	//for (auto& layer : m_CNN.GetLayers()) if (layer->GetOutput()) m_TensorFramebuffers.push_back({ *(layer->GetOutput()) });

	//m_InputTexture = { *m_Input };

	//m_OutputTexture = { *m_Layer.GetOutput() };

	//delete[] images;
}

void DrawLayer::OnAttach()
{
	m_TensorFramebuffers.push_back({ m_CNN.GetLayers()[0]->GetInput() });
	m_TensorFramebuffers.push_back({ m_CNN.GetLayers()[1]->GetOutput() });
	m_TensorFramebuffers.push_back({ m_CNN.GetLayers()[2]->GetOutput() });
	m_TensorFramebuffers.push_back({ m_CNN.GetLayers()[3]->GetOutput() });
	//Compass::Tensor<float> t1 = Compass::Tensor<float>(10, 10, 10);
	//Compass::RandomizeTensor<float>(t1, -1, 1);

	//for (auto& tensor : m_Layer.GetKernels())
	//{
	//	m_Textures.push_back({ tensor });
	//}

	//tFB = { t1 };

	//layer.Activate();
}

void DrawLayer::OnDetach()
{
}

void DrawLayer::OnUpdate(Timestep ts)
{
}


void DrawLayer::OnImGuiRender()
{
	Compass::Tensor<float> expected(10, 1, 1);
	expected.Fill(0);
	expected.GetData(m_TrainCount % 100, 0, 0) = 1;
	m_CNN.Train(m_TrainImages[m_TrainCount % 100], std::shared_ptr<Compass::Tensor<float>>(&expected, [](Compass::Tensor<float>* t) {}));

	ImGui::Begin("ConvNet Viewer");

	for (int i = 0; i < m_TensorFramebuffers.size(); i++)
	{
		auto& layer = m_TensorFramebuffers[i];
		layer.Update();
		ImGui::Image((void*)(std::size_t)layer.GetRendererID(), ImVec2(300, 300));
	}

	ImGui::End();

	/*
	m_Layer.Activate(m_Input);


	//for (auto& tensor : m_Layer.GetKernels()) { Compass::RandomizeTensor<float>(tensor, -1, 1); }
	m_Layer.Activate(m_Input);

	m_OutputTexture.Update();
	ImGui::Image((void*)(std::size_t)m_OutputTexture.GetRendererID(), ImVec2(300, 300));


	//for (auto& tFB : m_Textures)
	//{
	//	tFB.Update();
	//	ImGui::Image((void*)(std::size_t)tFB.GetRendererID(), ImVec2(100, 100));
	//}

	ImGui::End();
	*/


}

void DrawLayer::OnEvent(Event& e)
{
}
