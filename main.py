import cv2
import numpy as np
import time
from ultralytics import YOLO

class BallTracker3D:
    def __init__(self, model_path, ball_diameter_m, focal_length_px, camera_center, target_y_m):
        # Load the trained YOLO model (e.g., trained on a neon ball)
        self.model = YOLO(model_path)
        
        # Physical and Camera Parameters
        self.W_real = ball_diameter_m  # Real-world diameter of the ball in meters
        self.f = focal_length_px       # Camera focal length in pixels
        self.cx, self.cy = camera_center # Optical center of the camera frame (pixels)
        self.target_y = target_y_m     # The physical height of the trash bin opening
        
        # Tracking history
        self.history = []  # Stores tuples of (time, X, Y, Z)
        
    def estimate_3d_position(self, bbox):
        """Converts 2D bounding box to 3D physical coordinates."""
        x1, y1, x2, y2 = bbox
        
        # 1. Pixel dimensions
        w_px = x2 - x1
        h_px = y2 - y1
        d_px = max(w_px, h_px) # Use the largest dimension for stability
        
        # Center of the bounding box
        u = x1 + w_px / 2.0
        v = y1 + h_px / 2.0
        
        # 2. Monocular Depth Estimation (Z-axis)
        # Z = (Real Width * Focal Length) / Pixel Width
        Z = (self.W_real * self.f) / d_px
        
        # 3. Horizontal and Vertical Position (X, Y axes)
        # Assuming camera is at (0,0,0) and Y goes UP (inverted from image Y)
        X = ((u - self.cx) * Z) / self.f
        Y = ((self.cy - v) * Z) / self.f
        
        return X, Y, Z
        
    def predict_landing_spot(self):
        """Fits a trajectory to the historical points and calculates the landing coordinate."""
        if len(self.history) < 4:
            return None # Need enough points to fit a reliable parabola
            
        # Unpack history
        times = np.array([pt[0] for pt in self.history])
        X_vals = np.array([pt[1] for pt in self.history])
        Y_vals = np.array([pt[2] for pt in self.history])
        Z_vals = np.array([pt[3] for pt in self.history])
        
        # Time relative to the first tracked point
        t = times - times[0]
        
        # Fit X and Z to linear equations (constant velocity horizontally)
        # X(t) = v_x * t + X_0
        fit_X = np.polyfit(t, X_vals, 1) 
        fit_Z = np.polyfit(t, Z_vals, 1)
        
        # Fit Y to a quadratic equation (gravity parabola)
        # Y(t) = a * t^2 + b * t + c  (where a is roughly -0.5 * 9.81)
        fit_Y = np.polyfit(t, Y_vals, 2)
        a, b, c = fit_Y
        
        # We want to find the time (t_land) when Y(t) = target_y
        # a*t^2 + b*t + (c - target_y) = 0
        c_adjusted = c - self.target_y
        discriminant = b**2 - 4 * a * c_adjusted
        
        if discriminant < 0:
            return None # The ball never reaches the target height
            
        # Quadratic formula for roots
        t_root1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t_root2 = (-b - np.sqrt(discriminant)) / (2 * a)
        
        # The landing time is the future root (t > current time)
        current_t = t[-1]
        t_land = max(t_root1, t_root2)
        
        if t_land < current_t:
            return None # Object has already passed the target height
            
        # Calculate X and Z at the landing time
        X_land = fit_X[0] * t_land + fit_X[1]
        Z_land = fit_Z[0] * t_land + fit_Z[1]
        
        return X_land, Z_land

# --- Execution Example ---
if __name__ == "__main__":
    # Example Camera Setup (Logitech C920 at 1080p is roughly f=1050 px)
    tracker = BallTracker3D(
        model_path="neon_ball_yolo.pt", 
        ball_diameter_m=0.065,   # 6.5 cm tennis ball
        focal_length_px=1050.0,  
        camera_center=(960, 540), # Center of a 1920x1080 frame
        target_y_m=-0.5          # Bin is 0.5 meters below the camera
    )
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Run YOLO detection
        results = tracker.model.predict(frame, conf=0.5, verbose=False)
        
        for box in results[0].boxes:
            bbox = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
            
            # 2. Convert to 3D Coordinates
            X, Y, Z = tracker.estimate_3d_position(bbox)
            tracker.history.append((time.time(), X, Y, Z))
            
            # Keep only the last 15 frames for the physics curve fit
            if len(tracker.history) > 15:
                tracker.history.pop(0)
                
            # 3. Predict the landing spot
            landing = tracker.predict_landing_spot()
            
            if landing:
                X_land, Z_land = landing
                print(f"Predicted Landing: X={X_land:.2f}m, Z={Z_land:.2f}m")
                
                # Draw visual cues on the frame
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"Z: {Z:.2f}m", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Land: {X_land:.2f}m, {Z_land:.2f}m", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
