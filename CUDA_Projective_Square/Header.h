int i_l = 0, i_d = 0, i_u = 0;
int p_base_l = -1, p_base_u = -1;
int now_base_l = 0, next_base_l = color_vertices_num[layer][1];
int r_base_u = -1, c_base_u = 0;

std::map<std::pair<int, int>, int> block_offset;
for (auto iter : cooMap)
{
	int r = iter.first.first, c = iter.first.second;
	TYPE *v = iter.second;
	if (r == c)
	{
		MF_D_rowInd[i_d] = r;
		MF_D_colInd[i_d] = c;
		for (int i = 0; i < 9; i++)
			MF_D_Val[i_d * 9 + i] = v[i];
		block_offset[iter.first] = off_d + i_d;
		i_d++;
	}

	if (r < c)
	{
		while (r >= c_base_u)
		{
			p_base_u++;
			MF_GS_U_Ptr[p_base_u] = i_u;
			r_base_u = color_vertices_num[layer][p_base_u];
			c_base_u = color_vertices_num[layer][p_base_u + 1];
		}
		MF_U_rowInd[i_u] = r;
		MF_U_colInd[i_u] = c;
		MF_GS_U_rowInd[i_u] = r - r_base_u;
		MF_GS_U_colInd[i_u] = c - c_base_u;
		for (int i = 0; i < 9; i++)
			MF_U_Val[i_u * 9 + i] = v[i];
		block_offset[iter.first] = off_u + i_u;
		i_u++;
	}

	if (r > c)
	{
		while (r >= next_base_l)
		{
			p_base_l++;
			MF_GS_L_Ptr[p_base_l] = i_l;
			now_base_l = color_vertices_num[layer][p_base_l + 1];
			next_base_l = color_vertices_num[layer][p_base_l + 2];
		}
		MF_L_rowInd[i_l] = r;
		MF_L_colInd[i_l] = c;
		MF_GS_L_rowInd[i_l] = r - now_base_l;
		MF_GS_L_colInd[i_l] = c;
		for (int i = 0; i < 9; i++)
			MF_L_Val[i_l * 9 + i] = v[i];
		block_offset[iter.first] = off_l + i_l;
		i_l++;
	}
}
for (; p_base_u < colors_num[layer] - 1;)
	MF_GS_U_Ptr[++p_base_u] = i_u;
for (; p_base_l < colors_num[layer] - 1;)
	MF_GS_L_Ptr[++p_base_l] = i_l;
build_constraints(block_offset);
		}