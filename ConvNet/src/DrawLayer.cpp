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
		Compass::Tensor<float> input(28, 28, 1);
		//case_t c{ tensor_t<float>(28, 28, 1), tensor_t<float>(10, 1, 1) };

		uint8_t* img = images + 16 + i * (28 * 28);
		uint8_t* label = trainLabels + 8 + i;

		for (int x = 0; x < 28; x++)
			for (int y = 0; y < 28; y++)
				input.GetData(x, y, 0) = img[x + y * 28] / 255.0f;
		//c.data(x, y, 0) = img[x + y * 28] / 255.f;

		m_TrainImages.push_back(input);
		m_TrainLabels.push_back(*label);

		//for (int b = 0; b < 10; b++)
		  //c.out(b, 0, 0) = *label == b ? 1.0f : 0.0f;

		//cases.push_back(c);
	}

	//m_CNN.Forward(m_TrainImages[0]);

	//m_TensorFramebuffers.push_back({ *(m_CNN.GetLayers()[3]->GetOutput()) });

	//m_InputTexture = { *m_Input };

	//m_OutputTexture = { *m_Layer.GetOutput() };

	delete[] images;
}

void DrawLayer::OnAttach()
{
	m_TensorFramebuffers.push_back({ m_CNN.GetLayers()[0]->GetInput() });
	m_TensorFramebuffers.push_back({ m_CNN.GetLayers()[1]->GetOutput() });
	m_TensorFramebuffers.push_back({ m_CNN.GetLayers()[2]->GetOutput() });

	auto& convLayer = dynamic_cast<Compass::ConvolutionLayer&> (*m_CNN.GetLayers()[0]);
	for (auto& kernel : convLayer.GetKernels())
	{
		m_KernelFramebuffers.push_back({ kernel });
	}
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
	for (int i = 0; i < 10; i++)
	{

		Compass::Tensor<float> expected(10, 1, 1);
		expected.Fill(0);
		expected.GetData(m_TrainLabels[m_TrainCount % 100], 0, 0) = 1;
		ATL_TRACE(m_CNN.Train(m_TrainImages[m_TrainCount % 100], expected));

		m_TrainCount++;
	}

	m_TensorFramebuffers[0].Update(m_CNN.GetLayers()[0]->GetInput());
	m_TensorFramebuffers[1].Update(m_CNN.GetLayers()[1]->GetOutput());
	m_TensorFramebuffers[2].Update(m_CNN.GetLayers()[2]->GetOutput());

	ImGui::Begin("ConvNet Viewer");

	for (int i = 0; i < m_TensorFramebuffers.size(); i++)
	{
		auto& layer = m_TensorFramebuffers[i];
		ImGui::Image((void*)(std::size_t)layer.GetRendererID(), ImVec2(300, 300));
	}

	auto& convLayer = dynamic_cast<Compass::ConvolutionLayer&> (*m_CNN.GetLayers()[0]);
	for (int i = 0; i < m_KernelFramebuffers.size(); i++)
	{
		m_KernelFramebuffers[i].Update(convLayer.GetKernels()[i]);
	}

	for (int i = 0; i < m_KernelFramebuffers.size(); i++)
	{
		ImGui::Image((void*)(std::size_t)m_KernelFramebuffers[i].GetRendererID(), ImVec2(200, 200));
	}


	ImGui::End();

}

void DrawLayer::OnEvent(Event& e)
{
}
