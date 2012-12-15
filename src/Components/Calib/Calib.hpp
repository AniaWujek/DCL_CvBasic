/*!
 * \file
 * \brief 
 * \author Tomasz Kornuta [tkornuta@ia.pw.edu.pl]
 */

#ifndef CALIB_HPP_
#define CALIB_HPP_

#include "Component_Aux.hpp"
#include "Component.hpp"
#include "Panel_Empty.hpp"
#include "DataStream.hpp"
#include "Property.hpp"
#include "EventHandler2.hpp"
#include "Types/Objects3D/Chessboard.hpp"


namespace Processors {
namespace Calib {

/*!
 * \class Calib
 * \brief Calib processor class.
 *
 * Camera calibration based on chessboard corners
 */
class Calib: public Base::Component {
public:
	/*!
	 * Constructor.
	 */
	Calib(const std::string & name = "Calib");

	/*!
	 * Destructor
	 */
	virtual ~Calib();

	/*!
	 * Prepare components interface (register streams and handlers).
	 * At this point, all properties are already initialized
	 * (default values or the ones loaded from the configuration file).
	 */
	void prepareInterface();

protected:

	/*!
	 * Connects source to given device.
	 */
	bool onInit();

	/*!
	 * Disconnect source from device, closes streams, etc.
	 */
	bool onFinish();

	/*!
	 * Start component
	 */
	bool onStart();

	/*!
	 * Stop component
	 */
	bool onStop();

	// Data streams
	Base::DataStreamIn< Types::Objects3D::Chessboard > in_chessboard;

	// Handlers

	// Handler activated when datastream chessboard is present.
	Base::EventHandler2 h_process_chessboard;

	// Handler activated a calibration computations should be performed.
	Base::EventHandler2 h_perform_calibration;


	// Adds received chessboard observation to calibration set.
	void process_chessboard();

	// Performs the calibration.
	void perform_calibration();

private:
   // The vector of vectors of the object point projections on the calibration pattern views, one vector per a view.
	vector<vector<cv::Point2f> > imagePoints;

	// The vector of vectors of points on the calibration pattern in its coordinate system, one vector per view.
	vector<vector<cv::Point3f> > objectPoints;

};

} //: namespace Calib
} //: namespace Processors

/*
 * Register processor component.
 */
REGISTER_COMPONENT("Calib", Processors::Calib::Calib)

#endif /* CALIB_HPP_ */