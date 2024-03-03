#include "Histogram.h"
#include"wx/wx.h"
Histogram::Histogram(long int max, long int* fr, long int* fg, long int* fb) 
{
	this->max_frec = max;
	this->frec_r = fr;
	this->frec_g = fg;
	this->frec_b = fb;
	wxLogMessage("Valor max %li", this->max_frec);
}
