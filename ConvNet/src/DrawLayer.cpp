#include "DrawLayer.h"

#include <ImGui/imgui.h>

#include <fstream>

using namespace Atlas;

uint8_t* readFile(const char* filepath)
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
	: m_Layer(Compass::ConvolutionLayer<float>({ 28, 28, 1 }, 1, 4, 10))
{
	uint8_t* images = readFile("assets/train-images.idx3-ubyte");

	for (int i = 0; i < 1; i++)
	{
		//case_t c{ tensor_t<float>(28, 28, 1), tensor_t<float>(10, 1, 1) };

		uint8_t* img = images + 16 + i * (28 * 28);
		//uint8_t* label = train_labels + 8 + i;

		for (int x = 0; x < 28; x++)
			for (int y = 0; y < 28; y++)
				m_Input.GetData(x, y, 0) = img[x + y * 28] / 255.0f;
				//c.data(x, y, 0) = img[x + y * 28] / 255.f;

		//for (int b = 0; b < 10; b++)
			//c.out(b, 0, 0) = *label == b ? 1.0f : 0.0f;

		//cases.push_back(c);
	}

	m_InputTexture = { m_Input };
	m_Layer.Activate(m_Input);

	delete[] images;
}

void DrawLayer::OnAttach()
{
	//Compass::Tensor<float> t1 = Compass::Tensor<float>(10, 10, 10);
	//Compass::RandomizeTensor<float>(t1, -1, 1);

	for (auto& tensor : m_Layer.GetKernels())
	{
		m_Textures.push_back({ tensor });
	}

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
	ImGui::Begin("ConvNet Viewer");
	ImGui::Image((void*)(std::size_t)m_InputTexture.GetRendererID(), ImVec2(300, 300));


	for (auto& tFB : m_Textures)
	{
		ImGui::Image((void*)(std::size_t)tFB.GetRendererID(), ImVec2(100, 100));
	}

	ImGui::End();


}

void DrawLayer::OnEvent(Event& e)
{
}
