// ----------------------------------------------------------------------------
// -                       Open3DV: www.open3dv.org                           -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "Config.h"
#include "PointCloudIO.h"

#include <Core/Console.h>
#include <External/rply/rply.h>

namespace three{

namespace {

struct PLYReaderState {
	PointCloud *pointcloud_ptr;
	long vertex_index;
	long vertex_num;
	long normal_index;
	long normal_num;
	long color_index;
	long color_num;
};

int ReadVertexCallback(p_ply_argument argument)
{
	PLYReaderState *state_ptr;
	long index;
	ply_get_argument_user_data(argument, 
			reinterpret_cast<void **>(&state_ptr), &index);
	double value = ply_get_argument_value(argument);
	state_ptr->pointcloud_ptr->points_[state_ptr->vertex_index](index) = value;
	if (index == 2) {	// reading 'z'
		state_ptr->vertex_index++;
		if (state_ptr->vertex_index > state_ptr->vertex_num) {
			return 0;
		}
	}
	return 1;
}

int ReadNormalCallback(p_ply_argument argument)
{
	PLYReaderState *state_ptr;
	long index;
	ply_get_argument_user_data(argument, 
			reinterpret_cast<void **>(&state_ptr), &index);
	double value = ply_get_argument_value(argument);
	state_ptr->pointcloud_ptr->normals_[state_ptr->normal_index](index) = value;
	if (index == 2) {	// reading 'z'
		state_ptr->normal_index++;
		if (state_ptr->normal_index > state_ptr->normal_num) {
			return 0;
		}
	}
	return 1;
}

int ReadColorCallback(p_ply_argument argument)
{
	PLYReaderState *state_ptr;
	long index;
	ply_get_argument_user_data(argument, 
			reinterpret_cast<void **>(&state_ptr), &index);
	double value = ply_get_argument_value(argument);
	state_ptr->pointcloud_ptr->colors_[state_ptr->color_index](index) =
			value / 255.0;
	if (index == 2) {	// reading 'z'
		state_ptr->color_index++;
		if (state_ptr->color_index > state_ptr->color_num) {
			return 0;
		}
	}
	return 1;
}

}	// unnamed namespace

bool ReadPointCloudFromPLY(
		const std::string &filename,
		PointCloud &pointcloud)
{
	p_ply ply_file = ply_open(filename.c_str(), NULL, 0, NULL);
	if (!ply_file) {
		PrintDebug("Read PLY failed: unable to open file.\n");
		return false;
	}
	if (!ply_read_header(ply_file)) {
		PrintDebug("Read PLY failed: unable to parse header.\n");
		return false;
	}

	PLYReaderState state;
	state.pointcloud_ptr = &pointcloud;
	state.vertex_num = ply_set_read_cb(ply_file, "vertex", "x", 
			ReadVertexCallback, &state, 0);
	ply_set_read_cb(ply_file, "vertex", "y",  ReadVertexCallback, &state, 1);
	ply_set_read_cb(ply_file, "vertex", "z",  ReadVertexCallback, &state, 2);

	state.normal_num = ply_set_read_cb(ply_file, "vertex", "nx", 
			ReadNormalCallback, &state, 0);
	ply_set_read_cb(ply_file, "vertex", "ny",  ReadNormalCallback, &state, 1);
	ply_set_read_cb(ply_file, "vertex", "nz",  ReadNormalCallback, &state, 2);

	state.color_num = ply_set_read_cb(ply_file, "vertex", "red", 
			ReadColorCallback, &state, 0);
	ply_set_read_cb(ply_file, "vertex", "green",  ReadColorCallback, &state, 1);
	ply_set_read_cb(ply_file, "vertex", "blue",  ReadColorCallback, &state, 2);

	if (state.vertex_num <= 0) {
		PrintDebug("Read PLY failed: number of vertex <= 0.\n");
		return false;
	}

	state.vertex_index = 0;
	state.normal_index = 0;
	state.color_index = 0;

	pointcloud.Clear();
	pointcloud.points_.resize(state.vertex_num);
	pointcloud.normals_.resize(state.normal_num);
	pointcloud.colors_.resize(state.color_num);

	if (!ply_read(ply_file)) {
		PrintDebug("Read PLY failed: unable to read file.\n");
		return false;
	}

	ply_close(ply_file);
	return true;
}

bool WritePointCloudToPLY(
		const std::string &filename,
		const PointCloud &pointcloud,
		const bool write_ascii/* = false*/)
{
	if (pointcloud.HasPoints() == false) {
		PrintDebug("Write PLY failed: point cloud has 0 points.\n");
		return false;
	}
	
	p_ply ply_file = ply_create(filename.c_str(),
			write_ascii ? PLY_ASCII : PLY_LITTLE_ENDIAN, NULL, 0, NULL);
	if (!ply_file) {
		PrintDebug("Write PLY failed: unable to open file.\n");
		return false;
	}
	ply_add_comment(ply_file, "Created by Open3DV");
	ply_add_element(ply_file, "vertex",
			static_cast<long>(pointcloud.points_.size()));
	ply_add_property(ply_file, "x", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
	ply_add_property(ply_file, "y", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
	ply_add_property(ply_file, "z", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
	if (pointcloud.HasNormals()) {
		ply_add_property(ply_file, "nx", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
		ply_add_property(ply_file, "ny", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
		ply_add_property(ply_file, "nz", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
	}
	if (pointcloud.HasColors()) {
		ply_add_property(ply_file, "red", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
		ply_add_property(ply_file, "green", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
		ply_add_property(ply_file, "blue", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
	}
	if (!ply_write_header(ply_file)) {
		PrintDebug("Write PLY failed: unable to write header.\n");
		return false;
	}
	
	for (size_t i = 0; i < pointcloud.points_.size(); i++) {
		const Eigen::Vector3d &point = pointcloud.points_[i];
		ply_write(ply_file, point(0));
		ply_write(ply_file, point(1));
		ply_write(ply_file, point(2));
		if (pointcloud.HasNormals()) {
			const Eigen::Vector3d &normal = pointcloud.normals_[i];
			ply_write(ply_file, normal(0));
			ply_write(ply_file, normal(1));
			ply_write(ply_file, normal(2));
		}
		if (pointcloud.HasColors()) {
			const Eigen::Vector3d &color = pointcloud.colors_[i];
			ply_write(ply_file, color(0) * 255.0);
			ply_write(ply_file, color(1) * 255.0);
			ply_write(ply_file, color(2) * 255.0);
		}
	}

	ply_close(ply_file);
	return true;
}

}	// namespace three