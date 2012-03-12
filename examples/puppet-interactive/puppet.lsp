(system
 (gravity 0 0 -9.8)
 (damping 2.0)
 (tx "TorsoX" ty "TorsoY" tz "TorsoZ" rz "TorsoPsi" ry "TorsoTheta" rx "TorsoPhi" (Name "Torso") 
     (tz -1.5 (Mass 25 5 5 5))
     (tx -1.011 tz 0.658 (Name "Right Torso Hook"))
     (tx  1.011 tz 0.658 (Name "Left Torso Hook"))
     (tz 0.9 (Name "Head") (tz 0.5 (Mass 1 1 1 1))
         (tz 0.9 (Name "Head Hook")))
     (tx 1.3 tz 0.4
	 (rz "LShoulderPsi" ry "LShoulderTheta" rx "LShoulderPhi" (Name "Left Shoulder")
	     (tz -0.95 (Name "Left Humerus") (Mass 5 1 1 1))
	     (tz -1.9 
		 (rx "LElbowTheta" (Name "Left Elbow")
		     (tz -1.00 (Name "Left Radius") (Mass 4 1 1 1))
		     (tx 0.14 ty -0.173 tz -2.001 (Name "Left Finger"))))))
     (tx -1.3 tz 0.4 
	 (rz "RShoulderPsi" ry "RShoulderTheta" rx "RShoulderPhi" (Name "Right Shoulder")
	     (tz -0.95 (Name "Right Humerus") (Mass 5 1 1 1))
	     (tz -1.9 
		 (rx "RElbowTheta" (Name "Right Elbow")
		     (tz -1.00 (Name "Right Radius") (Mass 4 1 1 1))
		     (tx -0.14 ty -0.173 tz -2.001 (Name "Right Finger"))))))
     (tx 0.5 tz -3.0 
	 (rz "LHipPsi" ry "LHipTheta" rx "LHipPhi" (Name "Left Hip")
	     (tz -1.5 (Name "Left Femur") (Mass 5 1 1 1))
	     (ty -0.322 tz -2.59 (Name "Left Knee Hook"))
	     (tz -3.0 
		 (rx "LKneeTheta" (Name "Left Knee")
		     (tz -1.5 (Name "Left Tibia") (Mass 50 1 1 1))))))
     (tx -0.5 tz -3.0 
	  (rz "RHipPsi" ry "RHipTheta" rx "RHipPhi" (Name "Right Hip")
	      (tz -1.5 (Name "Right Femur") (Mass 5 1 1 1))
	      (ty -0.322 tz -2.59 (Name "Right Knee Hook"))
	      (tz -3.0 
		  (rx "RKneeTheta" (Name "Right Knee")
		      (tz -1.5 (Name "Right Tibia") (Mass 4 1 1 1)))))))

  (tz 10 (Name "String Plane")
      (tx (k "StringPlatformX")
	  (ty (k "StringPlatformY") (Name "String Platform")
	      (tx 1 (Name "Left Torso Spindle"))
	      (tx -1 (Name "Right Torso Spindle"))
	      (tx (k "LArmStringX") ty (k "LArmStringY") (Name "Left Arm Spindle"))
	      (tx (k "RArmStringX") ty (k "RArmStringY") (Name "Right Arm Spindle"))
	      (tx (k "LLegStringX") ty (k "LLegStringY") (Name "Left Leg Spindle"))
	      (tx (k "RLegStringX") ty (k "RLegStringY") (Name "Right Leg Spindle")))))
  
  (string-constraint "Head Hook" "String Platform" 12)
  ;(string-constraint "Left Torso Hook" "Left Torso Spindle" 13)
  ;(string-constraint "Right Torso Hook" "Right Torso Spindle" 13)
  (string-constraint "Left Finger" "Left Arm Spindle" "LArmStringL")
  (string-constraint "Right Finger" "Right Arm Spindle" "RArmStringL")
  (string-constraint "Left Knee Hook" "Left Leg Spindle" "LLegStringL")
  (string-constraint "Right Knee Hook" "Right Leg Spindle" "RLegStringL"))
