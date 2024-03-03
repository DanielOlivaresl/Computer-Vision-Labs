#pragma once
#include "wx/wx.h"

// ******* things to do *********
// - Add a general event handler that only changes the argument Convert(GRAY,RGB,HSV,HSL)

/// <summary>
///  A general functions that is the super method and the argumentss are the methods.
/// </summary>

class cMain : public wxMDIParentFrame
{
public:
	cMain(); // all the init of the GUI happens here...
	~cMain();
private:
	wxToolBar* toolBar = nullptr; // Implemented
	wxMenuBar* menuBar = nullptr; // Implemented

	//---------------------------------- Image buttuns events-------------------------------------------------------------------------
	// Every function here is the first step in the chain of process to aplly the conert functions, here is the click event 
	// -------------------------------------------------------------------------------------------------------------------------------
	// --------------CONVERT PROCEDURES--------------------------------
	void ToGray(wxCommandEvent& event);// Implemented       //
	void ToRGB(wxCommandEvent& event); // not yet...       //
	void ToHSV(wxCommandEvent& event); // not yet ...     //
	void ToHSL(wxCommandEvent& event); // not yet ...    //
	//------------------------------------------------------



	// APP HANDLERS-----------------
	void InMenuOpenNew(wxCommandEvent& event);// Implemented
	void InMenuSave(wxCommandEvent& event);// Implemented
	void InMenuExit(wxCommandEvent& event);// Implemented
	void InInfo(wxCommandEvent& event);
	// ------------------------------------
	wxDECLARE_EVENT_TABLE();
};
