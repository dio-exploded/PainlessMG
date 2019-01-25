#ifndef __FILE_IO_BMP_IO_H__
#define __FILE_IO_BMP_IO_H__

#include <stdio.h>
#include <windows.h>
#include "MY_MATH.h"
// We only consider 24 bits RGB color BMP files.
// The alpha channel is simply ignored.
// The real storage sequence in BITMAP is BGR, not RGB.


/*
template <class T>
bool BMP_Read(const char *filename, ARRAY_2D<SPECTRUM<T> >* &pixels, ARRAY_2D<T>* &alpha)
{
#ifdef __WIN32__
	HBITMAP bmp_handle = (HBITMAP)LoadImageA(NULL, filename, IMAGE_BITMAP, 0, 0, LR_LOADFROMFILE | LR_CREATEDIBSECTION | LR_DEFAULTSIZE);
	if(bmp_handle==INVALID_HANDLE_VALUE) return false;
	BITMAP bitmap;
	GetObject(bmp_handle, sizeof(BITMAP), (LPSTR)&bitmap);

	int bit_depth=bitmap.bmPlanes * bitmap.bmBitsPixel;
	if(bit_depth!=8 && bit_depth!=24 && bit_depth!=32)
	{printf("Error: File (%s)'s image depth is not supported %d.\n", filename, bit_depth); DeleteObject(bmp_handle); return false;}
	//printf("plane, %d; bits %d\n", bitmap.bmPlanes, bitmap.bmBitsPixel);

    if(pixels) delete pixels;
    if(alpha) delete alpha;
	pixels=new ARRAY_2D<SPECTRUM<T> > (bitmap.bmWidth, bitmap.bmHeight);
	alpha=0;

	unsigned char *ptr=(unsigned char *)(bitmap.bmBits);
	for(int j=0; j<pixels->size.y; j++)
	{
		unsigned char *line_ptr=ptr;
		for(int i=0; i<pixels->size.x; i++) 
		{
			if(bit_depth==8)
			{
				(*pixels)(i,j)=SPECTRUM<T>(line_ptr[0], line_ptr[0], line_ptr[0])*INV_255;
				line_ptr++;
			}
			if(bit_depth==24)
			{
				(*pixels)(i,j)=SPECTRUM<T>(line_ptr[2], line_ptr[1], line_ptr[0])*INV_255;
				line_ptr+=3;
			}
			if(bit_depth==32)
			{
				(*pixels)(i,j)=SPECTRUM<T>(line_ptr[2], line_ptr[1], line_ptr[0])*INV_255;
				line_ptr+=4;
			}
		}
		ptr+=bitmap.bmWidthBytes;
	}

	DeleteObject(bmp_handle);
	return true;
#endif
	return false;
}*/

bool BMP_Write(const char *filename, float *pixels, int width, int height)
{
	//Preparing the BITMAP data structure from IMAGE object.
	HDC dc= GetDC(NULL);
	BITMAPINFO info;
	ZeroMemory( &info.bmiHeader, sizeof(BITMAPINFOHEADER) );
	info.bmiHeader.biWidth=width;
	info.bmiHeader.biHeight=height;
	info.bmiHeader.biPlanes=1;
	info.bmiHeader.biBitCount=24;
	info.bmiHeader.biSizeImage=0;
	info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
	info.bmiHeader.biClrUsed= 0;
	info.bmiHeader.biClrImportant= 0;
	VOID *pvBits;
	HBITMAP bmp_handle=CreateDIBSection( dc, &info, DIB_RGB_COLORS, &pvBits,NULL,0 );
	BITMAP bitmap;
	GetObject(bmp_handle, sizeof(BITMAP), (LPSTR)&bitmap);

	unsigned char *ptr=(unsigned char *)(bitmap.bmBits);
	for(int j=0; j<height; j++)
	{
		unsigned char *line_ptr=ptr;
		for(int i=0; i<width; i++) 
		{
			line_ptr[2]=(unsigned char)CLAMP(pixels[(j*width+i)*3+0]*255, 0, 255);
			line_ptr[1]=(unsigned char)CLAMP(pixels[(j*width+i)*3+1]*255, 0, 255);
			line_ptr[0]=(unsigned char)CLAMP(pixels[(j*width+i)*3+2]*255, 0, 255);
			line_ptr+=3;
		}
		ptr+=bitmap.bmWidthBytes;
	}

    //Decide the data size.
	WORD wBitCount=24;
	DWORD dwPaletteSize=0, dwBmBitsSize, dwDIBSize, dwWritten;
	LPBITMAPINFOHEADER lpbi;          
	dwBmBitsSize = ((bitmap.bmWidth *  wBitCount+31)/32)* 4 *bitmap.bmHeight ;

	//Preparing the palette and the pixel data.
	HANDLE hDib  = GlobalAlloc(GHND,dwBmBitsSize+ dwPaletteSize+sizeof(BITMAPINFOHEADER));
	lpbi = (LPBITMAPINFOHEADER)GlobalLock(hDib);
	*lpbi = info.bmiHeader;
	HANDLE hPal,hOldPal=NULL;
	hPal = GetStockObject(DEFAULT_PALETTE);
	if (hPal){hOldPal = SelectPalette(dc, (HPALETTE)hPal, FALSE);RealizePalette(dc);}
	GetDIBits(dc, bmp_handle, 0, (UINT) bitmap.bmHeight, (LPSTR)lpbi + sizeof(BITMAPINFOHEADER) +dwPaletteSize,(BITMAPINFO *)lpbi, DIB_RGB_COLORS);
	if (hOldPal){SelectPalette(dc, (HPALETTE)hOldPal, TRUE);RealizePalette(dc);	ReleaseDC(NULL, dc);}

	//Start writing the file.
	HANDLE fh = CreateFileA(filename, GENERIC_WRITE, NULL, NULL,CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	if (fh == INVALID_HANDLE_VALUE) return false;
	BITMAPFILEHEADER bmfHdr;        
	bmfHdr.bfType = 0x4D42;//"BM"
	dwDIBSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + dwPaletteSize + dwBmBitsSize;  
	bmfHdr.bfSize = dwDIBSize;
	bmfHdr.bfReserved1 = 0;
	bmfHdr.bfReserved2 = 0;
	bmfHdr.bfOffBits = (DWORD)sizeof (BITMAPFILEHEADER) + (DWORD)sizeof(BITMAPINFOHEADER) + dwPaletteSize;
	WriteFile(fh, (LPSTR)&bmfHdr, sizeof(BITMAPFILEHEADER), &dwWritten, NULL);
	WriteFile(fh, (LPSTR)lpbi, dwDIBSize, &dwWritten, NULL);
	CloseHandle(fh);

    //Deallocation.
	GlobalUnlock(hDib);
	GlobalFree(hDib);
	DeleteObject(bmp_handle);
	return true;
}


#endif //__FILE_IO_BMP_IO_H__