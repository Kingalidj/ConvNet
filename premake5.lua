include "Dependencies.lua"

workspace "ConvNet"
	architecture "x86_64"
	startproject "ConvNet"

	configurations
	{
		"Debug",
		"Release",
		"Dist"
	}

	flags
	{
		"MultiProcessorCompile"
	}

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

group "Dependencies"
	include "ConvNet/vendor/Atlas/Atlas/vendor/GLFW"
	include "ConvNet/vendor/Atlas/Atlas/vendor/Glad"
	include "ConvNet/vendor/Atlas/Atlas/vendor/imgui"
group ""

include "ConvNet/vendor/Atlas/Atlas"
include "ConvNet"
